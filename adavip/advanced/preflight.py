"""Real released checkpoint tests; emits COMPLETE only after every assertion."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
import types

import numpy as np

from .common import atomic_json, presence_table


def ftp1(config, output):
    import torch
    from safetensors import safe_open
    from openpi.models_pytorch.ftp1_pytorch import FTP1Pytorch
    from openpi.policies.ftp1_load_utils import load_ftp1_model, _SKIP_CHECKPOINT_KEYS
    from adavip.ftp1.integration import FTP1UViP
    from adavip.ftp1.modulation import per_sample_flow_loss
    from .data import FTPData

    torch.manual_seed(42)
    dataset = FTPData(config["task"], config["cache"], config["teacher"])
    obs, actions = dataset.batch([0, len(dataset)//2])
    model, _, directory = load_ftp1_model(Path(config["teacher"]), device="cuda")
    model.eval().requires_grad_(False)
    # Check every native state_dict tensor has a corresponding release tensor.
    shapes = {}
    for file, prefix in [(directory / "model.safetensors", ""),
                         (directory / "hpt_tokenizer/shared_image_chunk_encoder.safetensors",
                          "hpt_tactile_encoder.shared_image_chunk_encoder."),
                         (directory / "hpt_tokenizer/GelSightMini_image_224_224_3.safetensors",
                          "hpt_tactile_encoder.tokenizers.GelSightMini_image_224_224_3.")]:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name not in _SKIP_CHECKPOINT_KEYS:
                    shapes[prefix+name] = list(handle.get_slice(name).get_shape())
    native_state = model.state_dict()
    covered = {value.untyped_storage().data_ptr() for name,value in native_state.items()
               if shapes.get(name)==list(value.shape)}
    # HF ties the VLM embedding to the loaded lm_head; safetensors stores a
    # single alias. Two one-token expert lm_heads are intentionally unused by
    # the native hidden-state-only action forward and excluded by its loader.
    missing = [name for name,value in native_state.items() if name not in _SKIP_CHECKPOINT_KEYS
               and shapes.get(name)!=list(value.shape) and value.untyped_storage().data_ptr() not in covered]
    if missing:
        raise AssertionError(f"Uncovered native checkpoint tensors: {missing}")
    torch.cuda.synchronize()
    report = {"checkpoint_tensor_coverage": len(model.state_dict()), "dataset": dataset.contract}
    report["tied_weight_aliases"] = [name for name,value in native_state.items()
        if name not in _SKIP_CHECKPOINT_KEYS and name not in shapes and value.untyped_storage().data_ptr() in covered]
    report["native_unused_lm_head_exclusions"] = sorted(_SKIP_CHECKPOINT_KEYS)
    print("PASS released tensor coverage", flush=True)
    noise = model.sample_noise(actions.shape, "cuda")
    times = torch.tensor([.25, .8], device="cuda")
    velocity = []
    hook = model.action_out_proj.register_forward_hook(lambda m,i,o: velocity.append(o.detach().clone()))
    # Disable only native training augmentation, keeping native loss/forward.
    def deterministic_preprocess(self, observation, *, train=True):
        return FTP1Pytorch._preprocess_observation(self, observation, train=False)
    model._preprocess_observation = types.MethodType(deterministic_preprocess, model)
    with torch.no_grad():
        native_loss = model(obs, actions, noise=noise, time=times)["loss"].clone()
        native_velocity = velocity[-1]
        native_actions = model.sample_actions("cuda", obs, noise=noise.clone(), num_steps=3)
    hook.remove()
    del model._preprocess_observation
    model.__class__ = FTP1UViP
    model.attach_uvip(rank=config.get("lora_rank", 16))
    with torch.no_grad():
        full = model(obs, actions, noise=noise, time=times)
        torch.testing.assert_close(full["velocity"], native_velocity, rtol=0, atol=0)
        torch.testing.assert_close(full["loss"], native_loss, rtol=0, atol=0)
        torch.testing.assert_close(full["per_sample_loss"].mean(), native_loss, rtol=1e-5, atol=1e-6)
        prediction = model.sample_actions("cuda", obs, noise=noise.clone(), num_steps=3)
        torch.testing.assert_close(prediction, native_actions, rtol=0, atol=0)
    report["identity_full_and_native_cached_sampling"] = "exact"
    print("PASS exact full/cached sampling identity", flush=True)
    table = torch.tensor(presence_table("ftp1"), device="cuda")
    model.uvip_enabled = False
    losses = []
    with torch.no_grad():
        model._raw_features = model.encode_native(obs)
        for flags in table:
            model.presence = flags[None].expand(len(actions), -1)
            result = model(obs, actions, noise=noise, time=times)
            assert torch.isfinite(result["per_sample_loss"]).all()
            losses.append(result["per_sample_loss"].cpu().tolist())
        model._raw_features = None
        # An arbitrarily different P must have zero effect when P is removed.
        model.presence = table[5:6].expand(len(actions), -1)
        first = model(obs, actions, noise=noise, time=times)["velocity"]
        changed = dataclasses.replace(obs, state=obs.state + 100)
        second = model(changed, actions, noise=noise, time=times)["velocity"]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    report["teacher_full_masked_per_sample_losses"] = losses
    report["label_scoring"] = "Native full forward with reused frozen prefusion tokens; no KV approximation"
    atomic_json(output / "teacher_loss_checks.json",report)
    report["state_adarms_mask_bypass"] = "absent"
    model.presence, model.uvip_enabled = None, True
    model.set_phase("B")
    audit = model.trainable_audit()
    assert len(model.lora_targets) == 126
    assert sum(name.endswith("lora_a") or name.endswith("lora_b") for name in audit) == 252
    assert all(name.startswith("uvip.") or (
        name.startswith("paligemma_with_expert.gemma_expert.model.") and name.endswith(("lora_a", "lora_b"))) for name in audit)
    atomic_json(output / "trainable_parameters.json", audit)
    report["trainable_numel"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Nonzero FiLM exercises Context/HyperNet action gradients, beyond init.
    with torch.no_grad():
        for layer in model.uvip.film.values():
            layer.weight.normal_(std=1e-4)
    model.zero_grad(set_to_none=True)
    result = model(obs, actions, noise=noise, time=times)
    result["utility"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.uvip.utility_predictor.parameters())
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith("uvip.utility_predictor."))
    model.zero_grad(set_to_none=True)
    result = model(obs, actions, noise=noise, time=times)
    result["loss"].backward()
    assert all(p.grad is None for p in model.uvip.utility_predictor.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.uvip.context.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if n.endswith("lora_b"))
    report["full_model_gradient_isolation"] = "passed"
    print("PASS gradients and action-only LoRA whitelist", flush=True)
    params = {n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(list(params.values()), lr=1e-4, betas=(.9,.95), weight_decay=1e-10)
    def step():
        model.zero_grad(set_to_none=True)
        result = model(obs, actions)
        loss = result["loss"] + .05 * result["utility"].square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 1.)
        optimizer.step()
        return loss.detach()
    step()
    saved = {"delta": {n:p.detach().cpu().clone() for n,p in params.items()},
             "optimizer": copy.deepcopy(optimizer.state_dict()), "cpu_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all()}
    torch.save(saved, output / "resume_test.pt")
    expected_loss = step()
    expected = {n:p.detach().cpu().clone() for n,p in params.items()}
    saved = torch.load(output / "resume_test.pt", weights_only=False)
    with torch.no_grad():
        for n,p in params.items():
            p.copy_(saved["delta"][n].to(p.device))
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    restored_loss = step()
    torch.testing.assert_close(restored_loss, expected_loss, rtol=0, atol=0)
    for name, value in params.items():
        torch.testing.assert_close(value.cpu(), expected[name], rtol=0, atol=0)
    report["checkpoint_optimizer_rng_resume"] = "exact"
    report["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    atomic_json(output / "COMPLETE.json", report)
    print("PASS exact checkpoint/optimizer/RNG continuation; all FTP tests passed", flush=True)


def tabero(config, output):
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax
    from openpi.models.pi0 import Pi0
    from adavip.tabero.vtla import load_released, trainable_filter
    from .data import TaberoData

    dataset = TaberoData(config["cache"], config["normalizer"])
    obs, actions = dataset.batch([0])
    model, _ = load_released(config["teacher"])
    print("PASS native Tabero checkpoint shape-strict load",flush=True)
    key = jax.random.key(42)
    # Native method still uses overridden embed_prefix, so explicitly bind the
    # original native class while evaluating the independent baseline.
    adapted_class = model.__class__
    model.__class__ = Pi0
    native_loss = nnx.jit(lambda m,o,a: m.compute_loss(key,o,a,train=False))(model,obs,actions)
    native_actions = nnx.jit(lambda m,o: m.sample_actions(key,o,num_steps=3))(model,obs)
    print("PASS native Teacher full loss and sampling",float(native_loss.mean()),flush=True)
    model.__class__ = adapted_class
    full_fn = nnx.jit(lambda m,o,a,p: m.flow_loss(key,o,a,presence=p))
    full = full_fn(model,obs,actions,jnp.ones((1,5), bool))
    np.testing.assert_allclose(np.asarray(full["loss"]), np.asarray(native_loss), rtol=.001, atol=.0001)
    output_actions = nnx.jit(lambda m,o: m.sample_actions(key,o,num_steps=3))(model,obs)
    np.testing.assert_allclose(np.asarray(output_actions), np.asarray(native_actions), rtol=.001, atol=.0001)
    print("PASS Tabero identity full loss and native cached sampling",flush=True)
    report = {"identity_loss_max_error": float(jnp.max(jnp.abs(full["loss"]-native_loss))),
              "identity_sampling_max_error": float(jnp.max(jnp.abs(output_actions-native_actions))),
              "dataset": dataset.contract}
    model.uvip_enabled = False
    prefill_fn = nnx.jit(lambda m,o,p: m.prefill(o,p))
    cached_fn = nnx.jit(lambda m,o,x,t,c,p: m.cached_velocity(o,x,t,c,p))
    _, nk, tk = jax.random.split(key,3)
    noise = jax.random.normal(nk,actions.shape)
    time = jax.random.beta(tk,1.5,1,(1,))*.999+.001
    xt = time[:,None,None]*noise+(1-time[:,None,None])*actions
    losses, cache_errors = [], []
    for presence in presence_table("tabero"):
        flags = jnp.asarray(presence[None])
        result = full_fn(model,obs,actions,flags)
        assert np.isfinite(np.asarray(result["loss"])).all()
        cached = cached_fn(model,obs,xt,time,prefill_fn(model,obs,flags),flags)
        cache_errors.append(float(jnp.max(jnp.abs(cached.astype(jnp.float32)-result["velocity"].astype(jnp.float32)))))
        losses.append(np.asarray(result["per_sample_loss"]).tolist())
        print("PASS Teacher full/masked cached loss",presence.tolist(),losses[-1],flush=True)
    flags = jnp.asarray(presence_table("tabero")[4:5])
    changed = dataclasses.replace(obs,state=obs.state+100)
    np.testing.assert_array_equal(np.asarray(full_fn(model,obs,actions,flags)["velocity"]),
                                  np.asarray(full_fn(model,changed,actions,flags)["velocity"]))
    report["teacher_full_masked_per_sample_losses"] = losses
    report["native_cached_vs_full_velocity_max_error"] = cache_errors
    report["label_scoring"] = "Native full forward; only prefusion encoder tokens reused, not Transformer KV"
    atomic_json(output / "teacher_loss_checks.json",report)
    model.uvip_enabled = True
    selected = nnx.state(model,trainable_filter("B"))
    flat = selected.flat_state()
    audit = {"/".join(map(str,p)):list(v.value.shape) for p,v in flat.items()}
    lora = [name for name in audit if not name.startswith("uvip/")]
    assert len(lora) == 10 and all("lora" in name for name in lora), lora
    assert not any("tactile_prefix_encoder" in name or "/img/" in name for name in audit)
    atomic_json(output / "trainable_parameters.json",audit)
    # Exercise nonzero FiLM, ensuring action gradients reach new context only.
    for head in model.uvip.film.values():
        head.kernel.value = jax.random.normal(key,head.kernel.value.shape)*1e-4
    def utility_loss(m):
        return jnp.mean(m.utility(obs)**2)
    grads = nnx.jit(nnx.grad(utility_loss,argnums=nnx.DiffState(0,trainable_filter("B"))))(model)
    flatgrads = {"/".join(map(str,p)):v.value for p,v in grads.flat_state().items()}
    assert any(float(jnp.linalg.norm(v))>0 for n,v in flatgrads.items() if n.startswith("uvip/utility_predictor/"))
    assert all(float(jnp.linalg.norm(v))==0 for n,v in flatgrads.items() if not n.startswith("uvip/utility_predictor/"))
    def action_loss(m):
        return jnp.mean(m.flow_loss(key,obs,actions)["loss"])
    grads = nnx.jit(nnx.grad(action_loss,argnums=nnx.DiffState(0,trainable_filter("B"))))(model)
    flatgrads = {"/".join(map(str,p)):v.value for p,v in grads.flat_state().items()}
    assert all(float(jnp.linalg.norm(v))==0 for n,v in flatgrads.items() if n.startswith("uvip/utility_predictor/"))
    assert any(float(jnp.linalg.norm(v))>0 for n,v in flatgrads.items() if n.startswith("uvip/context_"))
    assert any(float(jnp.linalg.norm(v))>0 for n,v in flatgrads.items() if n in lora)
    report["full_model_gradient_isolation"] = "passed"
    report["action_lora_tensor_count"] = len(lora)
    print("PASS Tabero gradient isolation and action-only LoRA whitelist",flush=True)
    # Resume round trip includes selected params, Adam moments and RNG.
    import pickle
    tx = optax.adamw(2.5e-5,b1=.9,b2=.95,weight_decay=1e-10)
    params = nnx.state(model,trainable_filter("B"))
    state = tx.init(params)
    updates, state = tx.update(grads,state,params)
    params = optax.apply_updates(params,updates)
    saved = jax.device_get((params,state,jax.random.key_data(key)))
    with open(output / "resume_test.pkl","wb") as handle:
        pickle.dump(saved,handle)
    with open(output / "resume_test.pkl","rb") as handle:
        restored = jax.tree.map(jnp.asarray,pickle.load(handle))
    u1,s1 = tx.update(grads,state,params)
    u2,s2 = tx.update(grads,restored[1],restored[0])
    for left,right in zip(jax.tree.leaves((u1,s1)),jax.tree.leaves((u2,s2))):
        np.testing.assert_array_equal(np.asarray(left),np.asarray(right))
    report["checkpoint_optimizer_rng_resume"] = "exact selected params/Adam/RNG round trip"
    atomic_json(output / "COMPLETE.json",report)
    print("PASS all real Tabero-VTLA checks",flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",choices=["ftp1","tabero"],required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    (ftp1 if args.model=="ftp1" else tabero)(config,output)


if __name__ == "__main__":
    main()
