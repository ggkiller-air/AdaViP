"""Architecture, all-data transport, diagnostic isolation and exact CPU resume."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")
from omegaconf import OmegaConf
from torch import nn

from adavip.manifeel.decoded_sensor_cache import build_decoded_cache
from adavip.manifeel.full_sensor_dataset import FullSensorManiFeelDataset
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, make_shape_meta
from adavip.manifeel.language_conditioned_obs_encoder import LanguageConditionedObsEncoder
from adavip.manifeel.language_conditioning_dataset import LanguageConditioningDataset
from adavip.manifeel.language_conditioning_diagnostics import gradient_matrix, measure_probes, select_probe_indices
from adavip.manifeel.language_conditioning_workspace import BalancedEpochSampler, LanguageConditioningWorkspace


@pytest.fixture
def data_args(tmp_path: Path) -> dict:
    specs = []
    for task, dimensions in enumerate((6, 7)):
        name = f"task{task}"
        specs.append(dict(task_id=name, text=name, dataset=name, isaacgym_cfg_name="unused", action_dim=dimensions))
        group = zarr.open_group(str(tmp_path / name), mode="w")
        data = group.create_group("data")
        group.create_group("meta").create_dataset("episode_ends", data=np.array([10, 21, 33]))
        rng = np.random.default_rng(task)
        for key in IMAGE_KEYS:
            data.create_dataset(key, data=rng.random((33, 8, 8, 3), dtype=np.float32))
        data.create_dataset("state", data=rng.random((33, 7), dtype=np.float32))
        data.create_dataset("action", data=rng.random((33, dimensions), dtype=np.float32))
    embeddings = tmp_path / "embeddings.npz"
    np.savez(embeddings, task_ids=[s["task_id"] for s in specs], embeddings=np.ones((2, 4), np.float32))
    return dict(shape_meta=make_shape_meta((3, 8, 8), (3, 8, 8), 4), dataset_root=str(tmp_path),
                task_embedding_path=str(embeddings), task_specs=specs, horizon=8, n_obs_steps=2,
                n_action_steps=2, seed=42, val_ratio=0.34)


def small_config(data_args: dict, output: Path, variant: str = "l1") -> object:
    """Exercise the real trainer and U-Net with inexpensive image encoders."""
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/manifeel/train_language_conditioning.yaml")
    cfg.variant, cfg.output_dir = variant, str(output)
    cfg.horizon, cfg.n_action_steps, cfg.shape_meta = 8, 2, data_args["shape_meta"]
    cfg.dataset = OmegaConf.create(dict(_target_="adavip.manifeel.language_conditioning_dataset.LanguageConditioningDataset",
                                      **data_args))
    cfg.policy.obs_encoder.token_dim = 8
    cfg.policy.obs_encoder.image_feature_dim = 8
    cfg.policy.obs_encoder.rgb_model = {
        "_target_": "torch.nn.Sequential", "_args_": [
            {"_target_": "torch.nn.Conv2d", "in_channels": 3, "out_channels": 8, "kernel_size": 1},
            {"_target_": "torch.nn.AdaptiveAvgPool2d", "output_size": 1}, {"_target_": "torch.nn.Flatten"}]}
    cfg.policy.down_dims, cfg.policy.diffusion_step_embed_dim = [8, 16], 8
    cfg.policy.kernel_size, cfg.policy.n_groups = 3, 4
    cfg.policy.noise_scheduler.num_train_timesteps = 10
    cfg.policy.num_inference_steps = 2
    cfg.dataloader.batch_size, cfg.dataloader.num_workers = 8, 0
    cfg.dataloader.pin_memory = False
    cfg.training.device, cfg.training.num_epochs = "cpu", 2
    cfg.training.lr_warmup_steps = 2
    cfg.training.checkpoint_epochs, cfg.training.checkpoint_every_steps = [1, 2], 1000
    cfg.diagnostics.samples_per_task, cfg.diagnostics.noise_repeats = 3, 2
    cfg.diagnostics.gradient_samples, cfg.diagnostics.batch_size = 2, 2
    cfg.runtime.expected_samples, cfg.runtime.expected_episodes = 66, 6
    return cfg


def test_common_initialization_and_head_shapes(data_args: dict, tmp_path: Path) -> None:
    policies = [LanguageConditioningWorkspace(small_config(data_args, tmp_path / v, v)).model for v in ("l1", "l2")]
    left, right = policies
    for name, value in left.state_dict().items():
        if not name.startswith("obs_encoder.language_head."):
            torch.testing.assert_close(value, right.state_dict()[name], rtol=0, atol=0)
    for policy in policies:
        assert not hasattr(policy.obs_encoder, "language_projection")
        assert policy.obs_encoder.fusion[0].in_features == 6 * 8 + 6
        assert sum(isinstance(m, nn.Linear) for m in policy.obs_encoder.language_head) == 2
        assert policy.model.down_modules[0][0].cond_encoder[1].in_features == 16
    assert not right.obs_encoder.language_head[-1].weight.any()
    assert not right.obs_encoder.language_head[-1].bias.any()


@pytest.mark.parametrize("variant", ["l1", "l2"])
def test_full_path_gradients_identity_and_action_interface(data_args: dict, tmp_path: Path, variant: str) -> None:
    workspace = LanguageConditioningWorkspace(small_config(data_args, tmp_path / variant, variant))
    policy, dataset = workspace.model, LanguageConditioningDataset(**data_args)
    policy.set_normalizer(dataset.get_normalizer())
    batch = torch.utils.data.default_collate([dataset[3], dataset[36]])
    normalized = policy.normalizer.normalize(batch["obs"])
    state = policy.obs_encoder.fuse_state_tokens(policy.obs_encoder.encode_sensor_tokens(normalized))
    condition = policy.obs_encoder(normalized)
    if variant == "l2":
        torch.testing.assert_close(state, condition, rtol=0, atol=0)
    losses = policy.compute_loss_per_sample(batch, torch.zeros(2, dtype=torch.long))
    losses.mean().backward()
    for encoder in policy.obs_encoder.image_encoders.values():
        assert next(encoder.parameters()).grad.abs().sum() > 0
    assert policy.obs_encoder.fusion[0].weight.grad.abs().sum() > 0
    assert policy.obs_encoder.state_projection[0].weight.grad.abs().sum() > 0
    assert policy.obs_encoder.language_head[-1].weight.grad.abs().sum() > 0
    if variant == "l2":
        assert not policy.obs_encoder.language_head[0].weight.grad.any()
    assert policy.predict_action(batch["obs"])["action"].shape == (2, 2, 7)
    with pytest.raises(ValueError, match="dropout"):
        policy.obs_encoder(normalized, torch.zeros(2, 7, dtype=torch.bool))


def test_all_episodes_balancing_and_lossless_cache(data_args: dict, tmp_path: Path) -> None:
    source = FullSensorManiFeelDataset(**data_args)
    cache = tmp_path / "cache"
    build_decoded_cache(source, str(cache), workers=2)
    dataset = LanguageConditioningDataset(**data_args, decoded_cache_dir=str(cache))
    assert len(dataset) == 66
    assert sum(map(len, dataset.val_episodes)) == 0
    before = [dataset[i] for i in (0, 9, 32, 33, 65)]
    dataset.load_ram_cache()
    for expected, i in zip(before, (0, 9, 32, 33, 65)):
        actual = dataset[i]
        for key in expected["obs"]:
            torch.testing.assert_close(expected["obs"][key], actual["obs"][key], rtol=0, atol=0)
        torch.testing.assert_close(expected["action_loss_mask"], actual["action_loss_mask"], rtol=0, atol=0)
    sampler = BalancedEpochSampler(dataset, 42, 8)
    complete = list(sampler)
    sampler.set_epoch(0, 3)
    assert list(sampler) == complete[24:]
    assert sum(sampler.weights[:33]) == sum(sampler.weights[33:])


def test_diagnostics_fixed_and_no_training_side_effects(data_args: dict, tmp_path: Path) -> None:
    workspace = LanguageConditioningWorkspace(small_config(data_args, tmp_path / "diag"))
    dataset, policy = LanguageConditioningDataset(**data_args), workspace.model
    policy.set_normalizer(dataset.get_normalizer())
    indices = select_probe_indices(dataset, 3, 2042)
    for rows in indices:
        assert all(dataset[i]["action_valid_mask"].all() and dataset[i]["observation_valid_mask"].all() for i in rows)
    for p in policy.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    rng = torch.get_rng_state().clone()
    first = measure_probes(policy, dataset, indices, True, noise_repeats=2, gradient_samples=2, batch_size=2)
    second = measure_probes(policy, dataset, indices, True, noise_repeats=2, gradient_samples=2, batch_size=2)
    assert first == second
    assert policy.training and torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in policy.parameters() if p.requires_grad)
    matrix = gradient_matrix([torch.tensor([1., 0.]), torch.tensor([-1., 0.]), torch.zeros(2)])
    assert matrix["negative_pair_fraction"] == 1
    assert matrix["valid_pairs"] == 1 and matrix["cosine"][2][0] is None


@pytest.mark.parametrize("variant", ["l1", "l2"])
def test_exact_optimizer_ema_sampler_noise_resume(data_args: dict, tmp_path: Path, variant: str) -> None:
    complete_cfg = small_config(data_args, tmp_path / f"complete_{variant}", variant)
    complete = LanguageConditioningWorkspace(complete_cfg, complete_cfg.output_dir)
    complete.run()
    stopped_cfg = small_config(data_args, tmp_path / f"stopped_{variant}", variant)
    stopped_cfg.runtime.stop_after_steps = 3
    stopped = LanguageConditioningWorkspace(stopped_cfg, stopped_cfg.output_dir)
    stopped.run()
    resumed_cfg = copy.deepcopy(stopped_cfg)
    resumed_cfg.runtime.stop_after_steps = 0
    resumed_cfg.resume = str(Path(stopped_cfg.output_dir) / "checkpoints/latest.ckpt")
    resumed = LanguageConditioningWorkspace(resumed_cfg, resumed_cfg.output_dir)
    resumed.run()
    assert complete.global_step == resumed.global_step == 18
    assert complete.ema_step == resumed.ema_step
    assert complete.cumulative_task_counts == resumed.cumulative_task_counts
    assert complete.probe_indices == resumed.probe_indices
    for name in ("model", "ema_model"):
        for key, value in getattr(complete, name).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, name).state_dict()[key], rtol=0, atol=0)
    for i, state in complete.optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][i][key], rtol=0, atol=0)
    assert torch.equal(complete._noise_generator.get_state(), resumed._noise_generator.get_state())
    assert sorted(p.name for p in (Path(resumed_cfg.output_dir) / "checkpoints").glob("*.ckpt")) == ["epoch1.ckpt", "epoch2.ckpt", "latest.ckpt"]
    bad = copy.deepcopy(resumed_cfg)
    bad.variant = "l2" if variant == "l1" else "l1"
    with pytest.raises((ValueError, RuntimeError)):
        LanguageConditioningWorkspace(bad, bad.output_dir).run()


def test_first_formal_update_precedes_ram_load(data_args: dict, tmp_path: Path, monkeypatch: object) -> None:
    cache = tmp_path / "cache"
    build_decoded_cache(FullSensorManiFeelDataset(**data_args), str(cache), workers=2)
    cfg = small_config(data_args, tmp_path / "early")
    cfg.dataset.decoded_cache_dir = str(cache)
    cfg.training.num_epochs = 1
    workspace = LanguageConditioningWorkspace(cfg, cfg.output_dir)
    original = LanguageConditioningDataset.load_ram_cache
    observed = []

    def load(dataset: object) -> None:
        observed.append(workspace.global_step)
        assert (Path(cfg.output_dir) / "checkpoints/latest.ckpt").is_file()
        original(dataset)

    monkeypatch.setattr(LanguageConditioningDataset, "load_ram_cache", load)
    workspace.run()
    assert observed == [1]
    assert workspace.global_step == 9
    assert sum(workspace.cumulative_task_counts) == 66
    events = [json.loads(line)["event"] for line in (Path(cfg.output_dir) / "progress.jsonl").read_text().splitlines()]
    assert events.index("train_step") < events.index("all_data_ram_ready")
