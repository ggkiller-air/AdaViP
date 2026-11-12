"""Bounded real-data H200 A/B/C update, affinity, gradient and checkpoint gates."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
from adavip.manifeel.stage2_ablation_workspace import (
    initialize_ablation_worker, parameter_report, tensor_fingerprint, validate_gradient_routes)
from adavip.manifeel.stage2_main_ablation import MainAblationTrainer, set_protocol_step
from scripts.manifeel.train_stage2_t2 import set_stage_trainability, update_ema
from scripts.manifeel.train_stage2_t2 import rng_state


class RepeatedSmokeWindows(Dataset):
    """Cycle real windows from every task; smoke never modifies formal sampling."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return 32 * 120

    def __getitem__(self, index):
        return self.samples[index % len(self.samples)]


def smoke_worker(worker_id):
    initialize_ablation_worker(worker_id)
    print(json.dumps({"event": "smoke_worker_affinity", "worker": worker_id,
                      "pid": os.getpid(), "cores": sorted(os.sched_getaffinity(0))}), flush=True)


def optimizer_fingerprint(optimizer):
    values = {f"{index}/{name}": value if torch.is_tensor(value) else torch.as_tensor(value)
              for index, state in optimizer.state_dict()["state"].items() for name, value in state.items()}
    return tensor_fingerprint(values)


def training_fingerprints(trainer):
    """Exact stored state fingerprints, distinct from subsequent GPU arithmetic."""
    return {"model": tensor_fingerprint(trainer.policy.state_dict()),
            "ema": tensor_fingerprint(trainer.ema.state_dict()),
            "optimizer": optimizer_fingerprint(trainer.optimizer),
            "scheduler": trainer.scheduler.state_dict()}


def assert_rng_equal(first, second):
    assert first["python"] == second["python"]
    assert first["numpy"][0] == second["numpy"][0]
    assert np.array_equal(first["numpy"][1], second["numpy"][1])
    assert first["numpy"][2:] == second["numpy"][2:]
    assert torch.equal(first["torch"], second["torch"])
    assert len(first["cuda"]) == len(second["cuda"])
    assert all(torch.equal(a, b) for a, b in zip(first["cuda"], second["cuda"]))


def run_smoke(policy, dataset, output, contract, cfg, args):
    """Run 100 B/C updates plus A checks; only emit PASS after exact recovery/inference."""
    output = Path(output)
    exact = policy.ablation == "C_teacher_residual"
    tolerance = dict(rtol=0, atol=0) if exact else dict(rtol=1e-6, atol=1e-8)
    report = {"ablation": policy.ablation, "device": torch.cuda.get_device_name(0),
              "trainer_pid": os.getpid(), "trainer_cores": sorted(os.sched_getaffinity(0)),
              "batch_size": 32, "workers": args.workers, "phases": {}}
    # Metadata already verified by the entrypoint. Map the original decoded data;
    # production separately verifies full-array hashes while loading all RAM.
    manifest = json.loads((args.cache / "manifest.json").read_text())
    groups = [{"data": {}} for _ in dataset.task_specs]
    for item in manifest["arrays"]:
        task = int(item["task_index"])
        values = np.load(args.cache / item["filename"], mmap_mode="r", allow_pickle=False)
        groups[task]["data"][item["key"]] = SourceIndexedArray(values, np.arange(len(values)), len(values))
    dataset.source.groups = groups
    dataset.source.verified_ram_cache = True
    selected = np.concatenate([np.flatnonzero(dataset.task_indices == task)[:8]
                               for task in range(len(dataset.task_specs))])
    samples = [dataset[int(index)] for index in selected]
    loader = DataLoader(RepeatedSmokeWindows(samples), batch_size=32, num_workers=args.workers,
                        pin_memory=True, persistent_workers=bool(args.workers),
                        worker_init_fn=smoke_worker, generator=torch.Generator().manual_seed(700042))
    batches = iter(loader)

    def next_batch():
        batch = next(batches)
        return {key: ({k: v.cuda(non_blocking=True) for k, v in value.items()} if key == "obs"
                      else value.cuda(non_blocking=True)) for key, value in batch.items()}

    trainer = MainAblationTrainer(policy, dataset, output / "check", contract, cfg, lambda: 0,
                                  workers=0)
    trainer.output.mkdir(parents=True, exist_ok=True)
    trainer.stats = trainer._new_stats()
    batch = next_batch()
    report["routes"] = validate_gradient_routes(policy, batch)
    teacher_encoder = tensor_fingerprint(policy.obs_encoder.state_dict())

    def update(batch, stage):
        trainer.optimizer.zero_grad(set_to_none=True)
        result = policy.training_losses(batch, stage)
        loss = result["utility_loss"] if stage == "A" else result["action_loss"] + .05 * result["utility_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite smoke loss")
        loss.backward()
        trainer.observe_before_clip()
        norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                             1.0, error_if_nonfinite=True)
        trainer.observe_clip(norm)
        trainer.optimizer.step()
        trainer.scheduler.step()
        update_ema(trainer.ema, policy, .9999)
        trainer.global_step += 1
        trainer.batch_cursor += 1
        return {key: float(result[key]) for key in ("action_loss", "utility_loss")}

    stages = [("A", 1, 0, 2)] if policy.utility_predictor is not None else []
    stages += [("B", 2, 2040, 50), ("C", 7, 12240, 50)]
    for stage, epoch, step, updates in stages:
        set_stage_trainability(policy, stage)
        policy.train()
        trainer.epoch, trainer.batch_cursor = epoch, 0
        set_protocol_step(trainer, step)
        trainer.stats = trainer._new_stats()
        before_roles = {name: tensor_fingerprint(module.state_dict()) for name, module in
                        (("unet", policy.model), ("utility", policy.utility_predictor),
                         ("attention", policy.cross_attention), ("query", policy.query_projection),
                         ("w_out", getattr(policy, "w_out", None))) if module is not None}
        before_frozen = tensor_fingerprint({name: p for name, p in policy.named_parameters() if not p.requires_grad})
        torch.cuda.synchronize()
        started, data_seconds = time.perf_counter(), 0.0
        for index in range(updates):
            requested = time.perf_counter()
            batch = next_batch()
            torch.cuda.synchronize()
            data_seconds += time.perf_counter() - requested
            if stage == "C" and index == updates - 1:
                trainer.save()
                saved_fingerprints = training_fingerprints(trainer)
                saved_rng = rng_state()
                resume_batch = batch
            losses = update(batch, stage)
            if index == 0 or (index + 1) % 20 == 0:
                print(json.dumps({"event": "smoke_update", "ablation": policy.ablation,
                                  "stage": stage, "updates": index + 1, **losses}), flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert before_frozen == tensor_fingerprint({name: p for name, p in policy.named_parameters() if not p.requires_grad})
        if stage in ("A", "B"):
            assert teacher_encoder == tensor_fingerprint(policy.obs_encoder.state_dict())
        else:
            assert teacher_encoder != tensor_fingerprint(policy.obs_encoder.state_dict())
        changed_roles = {name: before_roles[name] != tensor_fingerprint(module.state_dict())
                         for name, module in (("unet", policy.model), ("utility", policy.utility_predictor),
                         ("attention", policy.cross_attention), ("query", policy.query_projection),
                         ("w_out", getattr(policy, "w_out", None))) if module is not None}
        if exact:
            assert changed_roles["utility"]
            assert all(changed == (stage != "A") for name, changed in changed_roles.items() if name != "utility")
        report["phases"][stage] = dict(changed_roles=changed_roles, updates=updates, seconds=elapsed,
                                       samples_per_second=updates * 32 / elapsed,
                                       data_wait_seconds=data_seconds, frozen_exact=True,
                                       parameters=parameter_report(policy, gradients=True),
                                       clipping={key: value for key, value in trainer.stats.items()
                                                 if "clip" in key or "gradient" in key}, **losses)

    import copy
    expected_tensors = {"model": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
                        "ema": {k: v.detach().cpu().clone() for k, v in trainer.ema.state_dict().items()},
                        "optimizer": copy.deepcopy(trainer.optimizer.state_dict())}
    expected_rng = rng_state()
    expected = {"model": tensor_fingerprint(policy.state_dict()),
                "ema": tensor_fingerprint(trainer.ema.state_dict()),
                "optimizer": optimizer_fingerprint(trainer.optimizer),
                "scheduler": trainer.scheduler.state_dict()}
    trainer.restore(trainer.output / "checkpoints/latest.ckpt")
    assert saved_fingerprints == training_fingerprints(trainer)
    assert_rng_equal(saved_rng, rng_state())
    set_stage_trainability(policy, "C")
    update(resume_batch, "C")
    actual = {"model": tensor_fingerprint(policy.state_dict()),
                        "ema": tensor_fingerprint(trainer.ema.state_dict()),
                        "optimizer": optimizer_fingerprint(trainer.optimizer),
                        "scheduler": trainer.scheduler.state_dict()}
    comparisons = {}
    for role, state in (("model", policy.state_dict()), ("ema", trainer.ema.state_dict())):
        different = []
        for name, value in state.items():
            a, b = expected_tensors[role][name], value.detach().cpu()
            if not torch.equal(a, b):
                different.append({"name": name, "max_abs": float((a-b).abs().max()),
                                  "reference_max_abs": float(a.abs().max())})
        comparisons[role] = sorted(different, key=lambda x: -x["max_abs"])
    actual_rng = rng_state()
    comparisons["torch_rng_exact"] = torch.equal(expected_rng["torch"], actual_rng["torch"])
    comparisons["cuda_rng_exact"] = all(torch.equal(a, b) for a, b in zip(expected_rng["cuda"], actual_rng["cuda"]))
    comparisons["fingerprints"] = {key: expected[key] == actual[key] for key in expected}
    comparisons["checkpoint_state_exact"] = True
    comparisons["native_step_tolerance"] = tolerance
    (output / "resume_diagnostic.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    assert expected["scheduler"] == actual["scheduler"]
    assert_rng_equal(expected_rng, actual_rng)
    # Existing Full does not enable deterministic CUDA kernels. Preserve its
    # protocol; require exact restoration and tightly bounded native-step rounding.
    for role, state in (("model", policy.state_dict()), ("ema", trainer.ema.state_dict())):
        for name, value in state.items():
            torch.testing.assert_close(expected_tensors[role][name], value.detach().cpu(), **tolerance)
    optimizer_differences = []
    for index, state in trainer.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            reference = expected_tensors["optimizer"]["state"][index][name]
            torch.testing.assert_close(reference, value, **tolerance)
            if torch.is_tensor(value) and not torch.equal(reference, value):
                optimizer_differences.append(float((reference-value).abs().max()))
    comparisons["optimizer_max_abs_difference"] = max(optimizer_differences, default=0.0)
    (output / "resume_diagnostic.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    if exact:
        assert all(comparisons["fingerprints"].values()), "Deterministic resume differs"
    trainer.save()
    restored, _, _ = load_evaluation_policy(trainer.output / "checkpoints/latest.ckpt", device="cuda:0")
    observations = {key: value[:1] for key, value in resume_batch["obs"].items()}
    with torch.no_grad():
        before = trainer.ema.predict_action(observations, generator=torch.Generator(device="cuda").manual_seed(17))
        after = restored.predict_action(observations, generator=torch.Generator(device="cuda").manual_seed(17))
    torch.testing.assert_close(before["action"], after["action"], rtol=0, atol=0)
    assert torch.isfinite(after["action"]).all()
    if exact:
        assert torch.equal(before["action_pred"], after["action_pred"])
        del restored
        restored, _, _ = load_evaluation_policy(trainer.output / "checkpoints/latest.ckpt",
                                                device="cuda:0", weights="raw")
        policy.eval()
        before_raw = policy.predict_action(observations, generator=torch.Generator(device="cuda").manual_seed(17))
        after_raw = restored.predict_action(observations, generator=torch.Generator(device="cuda").manual_seed(17))
        for key in before_raw:
            assert torch.equal(before_raw[key], after_raw[key])
        report["portable_raw_inference"] = True
    encoded = policy.encode_stage2(observations)
    if policy.ablation in ("C", "E", "C_teacher_residual"):
        assert torch.equal(encoded["tokens"], encoded["modulated"])
    if policy.ablation in ("B", "D", "E"):
        assert torch.equal(encoded["query"], encoded["z_t"])
    if exact:
        import hashlib
        report["source_sha256"] = contract["source_sha256"]
        report["source_manifest_sha256"] = hashlib.sha256(Path("source.sha256").read_bytes()).hexdigest()
        report["teacher_parity"] = json.loads((output / "teacher_parity.json").read_text())
        report["deterministic_resume_bitwise"] = True
    report.update(exact_checkpoint_state_and_rng_restore=True, native_next_step_fp32_close=True,
                  native_next_step_bitwise_equal=comparisons["fingerprints"],
                  native_next_step_max_abs={role: max([r["max_abs"] for r in comparisons[role]], default=0.0)
                                            for role in ("model", "ema")},
                  portable_ema_inference=True,
                  finite_action_shape=list(after["action"].shape),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), status="PASS")
    (output / "SMOKE_PASS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "smoke_pass", "ablation": policy.ablation}), flush=True)
