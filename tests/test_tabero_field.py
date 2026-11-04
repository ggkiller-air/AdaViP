"""Check Field causality, mixed-control losses and native action restoration."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from adavip.tabero.data import TaberoDataset
from adavip.tabero.model import CausalTCNBlock, FieldEncoder, Normalizer, TaberoL1Policy
from adavip.tabero.schema import CACHE_VERSION, IMAGE_KEYS, fingerprint


def make_cache(root):
    root.mkdir()
    manifest = {"version": CACHE_VERSION, "total_steps": 6, "tasks": ["firm", "gentle"],
                "episodes": [{"start": 0, "end": 3, "task_index": 0},
                             {"start": 3, "end": 6, "task_index": 1}]}
    manifest["cache_id"] = fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "COMPLETE.json").write_text("{}")
    for key in IMAGE_KEYS:
        np.save(root / f"{key}.npy", np.zeros((6, 3, 224, 224), dtype=np.uint8))
    np.save(root / "state.npy", np.repeat(np.arange(6, dtype=np.float32)[:, None], 7, axis=1))
    np.save(root / "action.npy", np.repeat((np.arange(6, dtype=np.float32) + 10)[:, None], 13, axis=1))
    np.save(root / "marker.npy", np.repeat(np.arange(6, dtype=np.float32)[:, None], 9 * 198 * 2, axis=1).reshape(6, 9, 198, 2))
    np.save(root / "language.npy", np.zeros((2, 512), dtype=np.float32))
    return TaberoDataset(str(root), horizon=4, n_obs_steps=2, preload_ram=False)


def test_windows_delta_pose_and_force_targets(tmp_path):
    dataset = make_cache(tmp_path / "cache")
    example = dataset[3]
    torch.testing.assert_close(example["obs"]["state"][:, 0], torch.tensor([3., 3.]))
    torch.testing.assert_close(example["action"][:, 0], torch.tensor([10., 10., 11., 12.]))
    torch.testing.assert_close(example["action"][:, 7], torch.tensor([13., 13., 14., 15.]))
    assert example["action_loss_mask"][:, 0].tolist() == [False, True, True, True]
    assert torch.all(example["obs"]["marker"] == 3)
    stats = dataset.normalization_statistics()
    assert stats["action"]["min"][0] == 9
    assert stats["action"]["max"][0] == 12
    assert stats["action"]["min"][7] == 10
    assert sorted(dataset.epoch_indices(42, 0)) == list(range(6))


def tiny_policy():
    config = {"horizon": 16, "n_obs_steps": 2, "n_action_steps": 8,
              "token_dim": 32, "down_dims": [32, 64, 128], "diffusion_step_embed_dim": 16,
              "num_diffusion_steps": 10, "num_inference_steps": 2,
              "full_input_probability": 1., "force_loss_weight": 0.1}
    backbone = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 8))
    return TaberoL1Policy(config, FieldEncoder(token_dim=32, backbone=backbone, image_feature_dim=8))


def batch():
    return {"obs": {**{key: torch.randint(0, 256, (2, 2, 3, 16, 16), dtype=torch.uint8) for key in IMAGE_KEYS},
                    "state": torch.randn(2, 2, 7), "marker": torch.randn(2, 9, 198, 2), "language": torch.randn(2, 512)},
            "action": torch.randn(2, 16, 13), "action_loss_mask": torch.ones(2, 16, 13, dtype=torch.bool)}


def test_tcn_causality_and_language_exclusion():
    torch.set_num_threads(1)
    block = CausalTCNBlock(3, 8)
    values = torch.randn(2, 9, 3)
    changed = values.clone()
    changed[:, 5:] += 1000
    torch.testing.assert_close(block(values)[:, :5], block(changed)[:, :5], rtol=0, atol=0)
    policy, sample = tiny_policy(), batch()
    tokens = policy.obs_encoder.encode_tokens(sample["obs"], policy.normalizer)
    flags = torch.ones(2, 5, dtype=torch.bool)
    first = policy.obs_encoder.condition(tokens, sample["obs"]["language"], flags)
    second = policy.obs_encoder.condition(tokens, sample["obs"]["language"] + 10, flags)
    assert not torch.allclose(first, second)
    flags[:, 4] = False
    torch.testing.assert_close(policy.obs_encoder.condition(tokens, sample["obs"]["language"], flags),
                               policy.obs_encoder.condition(tokens, sample["obs"]["language"] + 10, flags), rtol=0, atol=0)


def test_masked_modality_invariance():
    policy, sample = tiny_policy(), batch()
    tokens = policy.obs_encoder.encode_tokens(sample["obs"], policy.normalizer)
    flags = torch.ones(2, 5, dtype=torch.bool)
    for group in range(4):
        changed = tokens.clone()
        changed[:, group] += 1000
        missing = flags.clone()
        missing[:, group] = False
        torch.testing.assert_close(policy.obs_encoder.condition(tokens, sample["obs"]["language"], missing),
                                   policy.obs_encoder.condition(changed, sample["obs"]["language"], missing), rtol=0, atol=0)


def test_native_tcn_last_frame_receptive_field():
    """Pin the existing checkpoint semantics: nine stored frames, five used."""
    torch.manual_seed(42)
    encoder = torch.nn.Sequential(CausalTCNBlock(3, 8), CausalTCNBlock(8, 8))
    values = torch.randn(2, 9, 3, requires_grad=True)
    encoder(values)[:, -1].sum().backward()
    by_frame = values.grad.abs().sum((0, 2))
    assert torch.equal(by_frame[:4], torch.zeros(4))
    assert (by_frame[4:] > 0).all()


def test_weighted_force_loss_and_full_input_gradients():
    policy, sample = tiny_policy(), batch()
    result = policy.loss_components(torch.ones(2, 16, 13), sample["action_loss_mask"])
    torch.testing.assert_close(result["loss"], torch.full((2,), 7.6 / 13))
    policy.compute_loss(sample).backward()
    for module in (policy.obs_encoder.images, policy.obs_encoder.tcn, policy.obs_encoder.state,
                   policy.obs_encoder.language_head, policy.model):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    with pytest.raises(ValueError, match="no modality dropout"):
        policy.compute_losses(sample, condition_ids=torch.ones(2, dtype=torch.long))


def test_ddpm_deployment_restores_absolute_pose():
    policy, sample = tiny_policy().eval(), batch()
    actions = policy.predict_action(sample["obs"])
    assert actions["action"].shape == (2, 8, 13)
    assert torch.isfinite(actions["action"]).all()


def test_exact_optimizer_ema_resume(tmp_path):
    from adavip.tabero.train import Trainer

    dataset = make_cache(tmp_path / "cache")
    dataset.horizon = 16
    config = {**tiny_policy().config, "seed": 42, "batch_size": 2, "num_epochs": 2,
              "num_workers": 0, "precision": "fp32", "learning_rate": 1e-4,
              "betas": [0.95, 0.999], "weight_decay": 1e-6, "warmup_steps": 2,
              "gradient_clip_norm": 1., "checkpoint_every_epochs": 1,
              "checkpoint_every_steps": 100, "checkpoint_first_step": 1,
              "preflight_steps": 4, "preflight_window": 2,
              "min_samples_per_second": 0., "max_data_wait_fraction": 1.}
    torch.manual_seed(17)
    first = tiny_policy()
    initial = {key: value.clone() for key, value in first.state_dict().items()}
    continuous = Trainer(config, tmp_path / "continuous", device="cpu", policy=first, dataset=dataset)
    continuous.run(stop_after_steps=4)
    second = tiny_policy()
    second.load_state_dict(initial)
    partial = Trainer(config, tmp_path / "resumed", device="cpu", policy=second, dataset=dataset)
    partial.run(stop_after_steps=2)
    resumed = Trainer(config, tmp_path / "resumed", device="cpu", policy=tiny_policy(), dataset=dataset)
    resumed.resume(tmp_path / "resumed/checkpoints/latest.ckpt")
    resumed.run(stop_after_steps=4)
    for reference, result in ((continuous.policy, resumed.policy), (continuous.ema.model, resumed.ema.model)):
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, result.state_dict()[name], rtol=0, atol=0)
    reference = continuous.optimizer.state_dict()["state"]
    for index, values in resumed.optimizer.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, reference[index][name], rtol=0, atol=0)
