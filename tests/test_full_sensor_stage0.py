"""CPU regression coverage for the full-sensor Stage-0 contract."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")
pytest.importorskip("diffusion_policy")
from diffusers import DDPMScheduler
from torch import nn

from adavip.manifeel.full_sensor_dataset import FullSensorManiFeelDataset
from adavip.manifeel.full_sensor_obs_encoder import FullSensorObsEncoder
from adavip.manifeel.full_sensor_schema import (
    CONDITION_NAMES, IMAGE_KEYS, SENSOR_KEYS, condition_presence,
    condition_probabilities, make_shape_meta,
)
from adavip.manifeel.reference_diffusion_policy import FullSensorReferencePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


@pytest.fixture
def dataset_arguments(tmp_path: Path) -> dict:
    """Create two tiny, distinguishable 6D/7D task stores with three episodes each."""
    specs = []
    for task_index, action_dim in enumerate((6, 7)):
        name = f"task_{task_index}"
        specs.append({
            "task_id": name, "text": name, "dataset": name,
            "isaacgym_cfg_name": "unused.yaml", "action_dim": action_dim,
        })
        group = zarr.open_group(str(tmp_path / name), mode="w")
        data = group.create_group("data")
        ends = np.array([4, 9, 15], dtype=np.int64)
        group.create_group("meta").create_dataset("episode_ends", data=ends)
        for sensor_index, key in enumerate(IMAGE_KEYS):
            height, width = (8, 8) if sensor_index < 3 else (6, 8)
            images = np.zeros((15, height, width, 3), dtype=np.float32)
            for timestep in range(15):
                images[timestep] = 0.1 + timestep * 0.01 + sensor_index * 0.005
            data.create_dataset(key, data=images)
        state = np.zeros((15, 7), dtype=np.float32)
        state[:, 0] = np.arange(15)
        state[:, 3] = 1
        data.create_dataset("state", data=state)
        actions = np.repeat(np.arange(15, dtype=np.float32)[:, None], action_dim, axis=1) * 0.1
        if action_dim == 7:
            actions[:, 6] = 0.02 + np.arange(15) * 0.001
        data.create_dataset("action", data=actions)
    embedding_path = tmp_path / "language.npz"
    np.savez(embedding_path, task_ids=np.array([spec["task_id"] for spec in specs]),
             embeddings=np.arange(8, dtype=np.float32).reshape(2, 4))
    return {
        "shape_meta": make_shape_meta((3, 8, 8), (3, 6, 8), 4),
        "dataset_root": str(tmp_path), "task_embedding_path": str(embedding_path),
        "task_specs": specs, "horizon": 8, "n_obs_steps": 2,
        "n_action_steps": 2, "val_ratio": 0.34, "seed": 42,
    }


def make_policy(shape_meta: dict, prediction_type: str = "epsilon") -> FullSensorReferencePolicy:
    """Build a real, small U-Net with cheap trainable image backbones."""
    encoder = FullSensorObsEncoder(
        shape_meta, token_dim=8, image_feature_dim=8,
        rgb_model=nn.Sequential(nn.Conv2d(3, 8, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()),
    )
    policy = FullSensorReferencePolicy(
        encoder, DDPMScheduler(num_train_timesteps=10, prediction_type=prediction_type),
        horizon=8, n_action_steps=2, num_inference_steps=2,
        diffusion_step_embed_dim=8, down_dims=(8, 16), kernel_size=3, n_groups=4,
    )
    normalizer = LinearNormalizer()
    for key in (*SENSOR_KEYS, "action"):
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    return policy


def collate_samples(dataset: FullSensorManiFeelDataset, indices: list[int]) -> dict:
    """Use PyTorch's normal collation on selected real dataset windows."""
    return torch.utils.data.default_collate([dataset[index] for index in indices])


def test_condition_contract_and_configurable_probability() -> None:
    presence = condition_presence()
    assert presence.shape == (10, 7)
    assert presence[0].all()
    np.testing.assert_array_equal(presence[1:8], ~np.eye(7, dtype=bool))
    np.testing.assert_array_equal(presence[8], [0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(presence[9], [1, 1, 1, 0, 0, 1, 1])
    probabilities = condition_probabilities()
    np.testing.assert_allclose(probabilities, [0.8] + [0.2 / 9] * 9)
    np.testing.assert_allclose(condition_probabilities(1), [1] + [0] * 9)
    weighted = condition_probabilities(0.5, [2] + [1] * 8)
    assert weighted[1] == pytest.approx(0.1)
    for probability, weights in [(-0.1, None), (1.1, None), (0.8, [0] * 9), (0.8, [-1] * 9)]:
        with pytest.raises(ValueError):
            condition_probabilities(probability, weights)


def test_dataset_seven_inputs_and_episode_boundary_masks(dataset_arguments: dict) -> None:
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    for record_index, (_, _, start, end) in enumerate(dataset.records):
        first_index = 0 if record_index == 0 else dataset.cumulative_lengths[record_index - 1]
        first = dataset[first_index]
        last = dataset[dataset.cumulative_lengths[record_index] - 1]
        assert set(first["obs"]) == set(SENSOR_KEYS)
        assert first["obs"]["state"][0, 0] == start
        assert first["obs"]["state"][1, 0] == start
        assert first["observation_valid_mask"].tolist() == [False, True]
        assert not first["action_valid_mask"][0]
        assert last["obs"]["state"][-1, 0] == end - 1
        assert last["action_valid_mask"].tolist() == [True, True] + [False] * 6
        assert torch.equal(last["action"][2:], last["action"][1:2].expand(6, -1))
        assert not last["action_loss_mask"][2:].any()
        assert first["action"].shape == (8, 7)
        task_index = first["task_index"].item()
        if dataset.task_specs[task_index].action_dim == 6:
            assert not first["action_loss_mask"][:, 6].any()
        else:
            assert first["action_loss_mask"][:, 6].sum() == end - start


def test_split_is_trajectory_disjoint_and_unused_episodes_are_not_validation(dataset_arguments: dict) -> None:
    dataset = FullSensorManiFeelDataset(**dataset_arguments, max_train_episodes=1)
    validation = dataset.get_validation_dataset()
    for task_index in range(2):
        train_ids = set(dataset.train_episodes[task_index].tolist())
        val_ids = set(dataset.val_episodes[task_index].tolist())
        assert len(train_ids) == len(val_ids) == 1
        assert train_ids.isdisjoint(val_ids)
        actual_val_ids = {record[1] for record in validation.records if record[0] == task_index}
        assert actual_val_ids == val_ids
    assert dataset.manifest == validation.manifest


def test_normalizer_uses_only_training_and_valid_action_dimensions(dataset_arguments: dict) -> None:
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    expected = dataset.get_normalizer()
    assert expected["action"].get_input_stats()["min"][6] > 0
    for task_index, episode_ids in enumerate(dataset.val_episodes):
        group = zarr.open_group(str(Path(dataset_arguments["dataset_root"]) / f"task_{task_index}"), mode="a")
        for episode_index in episode_ids:
            ends = dataset.episode_ends[task_index]
            start = 0 if episode_index == 0 else int(ends[episode_index - 1])
            group["data"]["action"][start:int(ends[episode_index])] = 10000
            group["data"]["state"][start:int(ends[episode_index])] = -10000
    rebuilt = FullSensorManiFeelDataset(**dataset_arguments)
    actual = rebuilt.get_validation_dataset().get_normalizer()
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[name], rtol=0, atol=0)
    assert not any(parameter.requires_grad for parameter in actual.parameters())


def test_future_targets_are_anchored_at_current_time_and_never_cross_episode(dataset_arguments: dict) -> None:
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    task_index, _, start, end = dataset.records[0]
    first = dataset.get_future_tactile(0, [1, 3])
    assert first["valid_mask"].tolist() == [True, True]
    raw = dataset.groups[task_index]["data"][IMAGE_KEYS[3]][start + 1]
    torch.testing.assert_close(first["obs"][IMAGE_KEYS[3]][0], torch.from_numpy(np.moveaxis(raw, -1, 0)))
    last = dataset.get_future_tactile(end - start - 1, [1, 3])
    assert not last["valid_mask"].any()
    assert all(not value.any() for value in last["obs"].values())
    with pytest.raises(ValueError):
        dataset.get_future_tactile(0, [0])


def test_balanced_sampler_has_equal_task_probability(dataset_arguments: dict) -> None:
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    sampler = dataset.make_task_balanced_sampler(torch.Generator().manual_seed(1))
    split = dataset.task_lengths[0]
    assert sampler.weights[:split].sum().item() == pytest.approx(1)
    assert sampler.weights[split:].sum().item() == pytest.approx(1)


@pytest.mark.parametrize("condition_index", range(1, 10))
def test_removed_sensor_cannot_leak_into_fusion(dataset_arguments: dict, condition_index: int) -> None:
    torch.manual_seed(7)
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    batch = collate_samples(dataset, [0, len(dataset) - 1])
    encoder = make_policy(dataset.shape_meta).obs_encoder.eval()
    presence = torch.from_numpy(condition_presence()[condition_index]).expand(2, -1)
    changed = {key: value.clone() for key, value in batch["obs"].items()}
    for sensor_index, key in enumerate(SENSOR_KEYS):
        if not presence[0, sensor_index]:
            changed[key] = changed[key] + 0.37
    original = encoder(batch["obs"], presence)
    counterfactual = encoder(changed, presence)
    torch.testing.assert_close(original, counterfactual, rtol=0, atol=0)
    assert encoder.encode_tokens(batch["obs"]).shape == (2, 7, 8)


def test_all_seven_inputs_and_history_have_gradients(dataset_arguments: dict) -> None:
    torch.manual_seed(4)
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    batch = collate_samples(dataset, [1, len(dataset) - 1])
    policy = make_policy(dataset.shape_meta)
    policy.set_normalizer(dataset.get_normalizer())
    loss = policy.compute_loss(batch, condition_ids=torch.zeros(2, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss)
    for key in IMAGE_KEYS:
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in policy.obs_encoder.image_encoders[key].parameters())
    for projection in (policy.obs_encoder.state_projection, policy.obs_encoder.language_projection):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in projection.parameters())
    altered = copy.deepcopy(batch["obs"])
    altered["front"][:, 0] += 0.4
    assert not torch.allclose(policy.obs_encoder(batch["obs"]), policy.obs_encoder(altered))


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_paired_losses_share_every_random_variable_and_preserve_signed_scores(
    dataset_arguments: dict, monkeypatch: pytest.MonkeyPatch, prediction_type: str,
) -> None:
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    batch = collate_samples(dataset, [0, len(dataset) - 1])
    policy = make_policy(dataset.shape_meta, prediction_type).eval()
    seen = []

    class RecordingDenoiser(nn.Module):
        """Record exactly the data seen by each counterfactual denoiser."""

        def forward(self, trajectory, timesteps, global_cond):
            seen.append((trajectory.clone(), timesteps.clone()))
            return global_cond[:, :1, None].expand_as(trajectory)

    policy.model = RecordingDenoiser().eval()
    monkeypatch.setattr(policy.obs_encoder, "fuse_tokens", lambda tokens, presence: presence.float().sum(1, keepdim=True).expand(-1, 8))
    noise = torch.zeros_like(batch["action"])
    timesteps = torch.tensor([2, 5])
    result = policy.paired_denoising_losses(batch, noise=noise, timesteps=timesteps)
    assert result["losses"].shape == (2, 10)
    assert result["utility"].shape == (2, 9)
    torch.testing.assert_close(result["utility"], result["losses"][:, 1:] - result["losses"][:, :1])
    if prediction_type == "epsilon":
        assert torch.all(result["utility"] < 0)
    assert len(seen) == 10
    for trajectory, steps in seen:
        torch.testing.assert_close(trajectory, seen[0][0], rtol=0, atol=0)
        torch.testing.assert_close(steps, timesteps, rtol=0, atol=0)
    policy.train()
    with pytest.raises(RuntimeError):
        policy.paired_denoising_losses(batch)


def test_invalid_action_and_padded_time_have_zero_gradient(dataset_arguments: dict) -> None:
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    batch = collate_samples(dataset, [dataset.cumulative_lengths[0] - 1])
    policy = make_policy(dataset.shape_meta)
    policy.set_normalizer(dataset.get_normalizer())

    class LearnablePrediction(nn.Module):
        """Expose direct denoising-output gradients for every action/time slot."""

        def __init__(self):
            super().__init__()
            self.output = nn.Parameter(torch.ones(1, 8, 7))

        def forward(self, trajectory, timesteps, global_cond):
            return self.output.expand_as(trajectory)

    policy.model = LearnablePrediction()
    noise = torch.zeros_like(batch["action"])
    sample = policy.prepare_denoising_sample(batch, noise, torch.tensor([2]))
    changed = copy.deepcopy(batch)
    changed["action"][batch["action_loss_mask"] == 0] = 100000
    other = policy.prepare_denoising_sample(changed, noise, torch.tensor([2]))
    torch.testing.assert_close(sample.noisy_action, other.noisy_action, rtol=0, atol=0)
    policy.denoising_loss_per_sample(sample, torch.zeros(1, 8)).mean().backward()
    gradient = policy.model.output.grad
    assert not gradient[batch["action_loss_mask"] == 0].any()
    assert torch.all(gradient[batch["action_loss_mask"] == 1] != 0)


def test_fixed_noise_pair_matches_full_compute_loss_and_checkpoint_roundtrip(dataset_arguments: dict) -> None:
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    batch = collate_samples(dataset, [0, len(dataset) - 1])
    policy = make_policy(dataset.shape_meta).eval()
    policy.set_normalizer(dataset.get_normalizer())
    sample = policy.prepare_denoising_sample(batch, generator=torch.Generator().manual_seed(7))
    paired = policy.paired_denoising_losses(batch, sample.noise, sample.timesteps)
    full = policy.compute_loss(batch, denoising_sample=sample)
    torch.testing.assert_close(full, paired["losses"][:, 0].mean())
    restored = make_policy(dataset.shape_meta).eval()
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(restored.compute_loss(batch, denoising_sample=sample), full, rtol=0, atol=0)
    actions = restored.predict_action(batch["obs"], generator=torch.Generator().manual_seed(3))
    assert actions["action"].shape == (2, 2, 7)
    assert actions["action_pred"].shape == (2, 8, 7)
    assert torch.isfinite(actions["action"]).all()


def test_condition_sampler_matches_eighty_twenty_default(dataset_arguments: dict) -> None:
    policy = make_policy(dataset_arguments["shape_meta"])
    sampled = policy.sample_condition_ids(100000, torch.Generator().manual_seed(4))
    frequencies = torch.bincount(sampled, minlength=len(CONDITION_NAMES)).float() / len(sampled)
    torch.testing.assert_close(frequencies, policy.condition_probabilities, atol=0.004, rtol=0)


def test_six_dimensional_only_normalizer_is_finite(dataset_arguments: dict) -> None:
    arguments = {**dataset_arguments, "task_specs": dataset_arguments["task_specs"][:1]}
    normalizer = FullSensorManiFeelDataset(**arguments).get_normalizer()
    assert all(torch.isfinite(value).all() for value in normalizer.state_dict().values())
    assert normalizer["action"].get_input_stats()["mean"][6] == 0


def test_single_training_frame_has_finite_statistics(dataset_arguments: dict) -> None:
    arguments = {**dataset_arguments, "task_specs": dataset_arguments["task_specs"][:1]}
    group = zarr.open_group(str(Path(arguments["dataset_root"]) / "task_0"), mode="a")
    for key in (*IMAGE_KEYS, "state", "action"):
        values = group["data"][key][:2]
        del group["data"][key]
        group["data"].create_dataset(key, data=values)
    del group["meta"]["episode_ends"]
    group["meta"].create_dataset("episode_ends", data=np.array([1, 2], dtype=np.int64))
    dataset = FullSensorManiFeelDataset(**arguments)
    assert len(dataset) == 1
    assert all(torch.isfinite(value).all() for value in dataset.get_normalizer().state_dict().values())


def test_all_nine_masks_support_training_gradients(dataset_arguments: dict) -> None:
    torch.manual_seed(8)
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    batch = collate_samples(dataset, [index % len(dataset) for index in range(9)])
    policy = make_policy(dataset.shape_meta).train()
    policy.set_normalizer(dataset.get_normalizer())
    losses = policy.compute_loss_per_sample(batch, torch.arange(1, 10))
    assert losses.shape == (9,)
    assert torch.isfinite(losses).all()
    losses.mean().backward()
    gradients = policy.obs_encoder.missing_tokens.grad
    assert gradients is not None
    assert torch.all(gradients.abs().sum(dim=1) > 0)
    with pytest.raises(ValueError, match="dtype"):
        policy.prepare_denoising_sample(batch, noise=torch.zeros_like(batch["action"], dtype=torch.float64))


@pytest.mark.parametrize("corruption", ["missing", "length", "state_shape", "dtype"])
def test_dataset_rejects_incomplete_or_misaligned_stores(dataset_arguments: dict, corruption: str) -> None:
    group = zarr.open_group(str(Path(dataset_arguments["dataset_root"]) / "task_0"), mode="a")
    key = "state" if corruption == "state_shape" else "side"
    original = group["data"][key][:]
    del group["data"][key]
    if corruption == "length":
        group["data"].create_dataset(key, data=original[:-1])
    elif corruption == "state_shape":
        group["data"].create_dataset(key, data=original[:, :6])
    elif corruption == "dtype":
        group["data"].create_dataset(key, data=original.astype(np.float64))
    with pytest.raises((KeyError, ValueError)):
        FullSensorManiFeelDataset(**dataset_arguments)


def workspace_config(dataset_arguments: dict, output_dir: Path):
    """Shrink only model/data dimensions while keeping the actual workspace path."""
    from omegaconf import OmegaConf

    config_path = Path(__file__).resolve().parents[1] / "configs/manifeel/train_full_sensor_reference.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.output_dir = str(output_dir)
    cfg.shape_meta = dataset_arguments["shape_meta"]
    cfg.dataset = {"_target_": "adavip.manifeel.full_sensor_dataset.FullSensorManiFeelDataset", **dataset_arguments}
    cfg.horizon = 8
    cfg.n_action_steps = 2
    cfg.policy.obs_encoder.token_dim = 8
    cfg.policy.obs_encoder.image_feature_dim = 8
    cfg.policy.obs_encoder.rgb_model = {
        "_target_": "torch.nn.Sequential", "_args_": [
            {"_target_": "torch.nn.Conv2d", "in_channels": 3, "out_channels": 8, "kernel_size": 1},
            {"_target_": "torch.nn.AdaptiveAvgPool2d", "output_size": 1},
            {"_target_": "torch.nn.Flatten"},
        ],
    }
    cfg.policy.noise_scheduler.num_train_timesteps = 10
    cfg.policy.num_inference_steps = 2
    cfg.policy.diffusion_step_embed_dim = 8
    cfg.policy.down_dims = [8, 16]
    cfg.policy.kernel_size = 3
    cfg.policy.n_groups = 4
    cfg.training.device = "cpu"
    cfg.training.num_epochs = 2
    cfg.training.max_train_steps = 2
    cfg.training.max_val_steps = 2
    cfg.training.checkpoint_every = 1
    cfg.training.lr_warmup_steps = 1
    for options in (cfg.dataloader, cfg.val_dataloader):
        options.batch_size = 3
        options.num_workers = 0
    return cfg


@pytest.mark.parametrize("mode", ["ram", "mmap"])
@pytest.mark.parametrize("resize", [False, True])
def test_decoded_cache_is_lossless_for_all_windows_and_normalizers(
    dataset_arguments: dict, tmp_path: Path, mode: str, resize: bool,
) -> None:
    from adavip.manifeel.decoded_sensor_cache import build_decoded_cache

    if resize:
        dataset_arguments = {**dataset_arguments, "shape_meta": make_shape_meta((3, 4, 6), (3, 3, 5), 4)}
    direct = FullSensorManiFeelDataset(**dataset_arguments, max_train_episodes=1)
    cache_dir = str(tmp_path / "decoded")
    manifest = build_decoded_cache(direct, cache_dir, workers=2, block_frames=2)
    assert build_decoded_cache(direct, cache_dir, workers=2, block_frames=3) == manifest
    cached = FullSensorManiFeelDataset(
        **dataset_arguments, max_train_episodes=1, decoded_cache_dir=cache_dir, decoded_cache_mode=mode,
    )
    assert cached.manifest == direct.manifest
    for source, prepared in ((direct, cached), (direct.get_validation_dataset(), cached.get_validation_dataset())):
        for index in range(len(source)):
            expected, actual = source[index], prepared[index]
            for key in expected:
                if key == "obs":
                    for sensor in expected[key]:
                        torch.testing.assert_close(expected[key][sensor], actual[key][sensor], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
            future = source.get_future_tactile(index, [1, 2])
            cached_future = prepared.get_future_tactile(index, [1, 2])
            for key in future["obs"]:
                torch.testing.assert_close(future["obs"][key], cached_future["obs"][key], rtol=0, atol=0)
    for key, value in direct.get_normalizer().state_dict().items():
        torch.testing.assert_close(value, cached.get_normalizer().state_dict()[key], rtol=0, atol=0)
    assert not cached.groups[0]["data"]["front"].values.flags.writeable
    values = cached.groups[0]["data"]["front"].values
    if mode == "ram":
        assert not isinstance(values, np.memmap)
        assert values.flags.owndata
    else:
        assert isinstance(values, np.memmap)
    missing = np.flatnonzero(cached.groups[0]["data"]["front"].row_map < 0)
    with pytest.raises(IndexError, match="unused"):
        cached.groups[0]["data"]["front"][missing[0]]


def test_decoded_cache_rejects_corruption_and_changed_contract(dataset_arguments: dict, tmp_path: Path) -> None:
    from adavip.manifeel.decoded_sensor_cache import build_decoded_cache

    direct = FullSensorManiFeelDataset(**dataset_arguments)
    cache_dir = tmp_path / "decoded"
    build_decoded_cache(direct, str(cache_dir), workers=2, block_frames=3)
    with pytest.raises(ValueError, match="contract"):
        FullSensorManiFeelDataset(**dataset_arguments, max_train_episodes=1, decoded_cache_dir=str(cache_dir))
    values = np.load(cache_dir / "task0_front.npy", mmap_mode="r+")
    values[0, 0, 0, 0] += 0.001
    values.flush()
    del values
    with pytest.raises(ValueError, match="Corrupt"):
        FullSensorManiFeelDataset(**dataset_arguments, decoded_cache_dir=str(cache_dir))


def test_decoded_ram_cache_rejects_insufficient_job_memory(
    dataset_arguments: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adavip.manifeel.decoded_sensor_cache import build_decoded_cache

    cache_dir = str(tmp_path / "decoded")
    build_decoded_cache(FullSensorManiFeelDataset(**dataset_arguments), cache_dir, workers=2)
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "1024")
    with pytest.raises(MemoryError, match="job memory"):
        FullSensorManiFeelDataset(**dataset_arguments, decoded_cache_dir=cache_dir)


@pytest.mark.parametrize("throughput,waiting,passed", [(220, 0.05, True), (90, 0.05, False), (220, 0.6, False)])
def test_throughput_gate_ignores_warmup_and_stops_bad_long_runs(
    dataset_arguments: dict, tmp_path: Path, throughput: float, waiting: float, passed: bool,
) -> None:
    from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace

    cfg = workspace_config(dataset_arguments, tmp_path)
    cfg.runtime.preflight_windows = 3
    cfg.runtime.min_samples_per_second = 150
    cfg.runtime.max_data_wait_fraction = 0.35
    workspace = FullSensorWorkspace(cfg, cfg.output_dir)
    workspace._check_throughput_preflight({"samples_per_second": 1, "data_wait_fraction": 0.99})
    record = {"samples_per_second": throughput, "data_wait_fraction": waiting}
    workspace._check_throughput_preflight(record)
    assert workspace._stop_requested is None
    workspace._check_throughput_preflight(record)
    assert workspace.throughput_preflight_failed == (not passed)
    assert (workspace._stop_requested is None) == passed


def test_legacy_checkpoint_resumes_exactly_with_cache_and_new_worker_count(
    dataset_arguments: dict, tmp_path: Path,
) -> None:
    import dill

    from adavip.manifeel.decoded_sensor_cache import build_decoded_cache
    from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace, manifest_fingerprint, training_protocol

    cfg = workspace_config(dataset_arguments, tmp_path / "direct_training")
    direct = FullSensorWorkspace(cfg, cfg.output_dir)
    direct.run()
    stopped_cfg = copy.deepcopy(cfg)
    stopped_cfg.output_dir = str(tmp_path / "stopped")
    stopped = FullSensorWorkspace(stopped_cfg, stopped_cfg.output_dir)
    original_step = stopped.optimizer.step

    def stop_after_one(*args, **kwargs):
        result = original_step(*args, **kwargs)
        stopped.request_stop("test_transport_change")
        return result

    stopped.optimizer.step = stop_after_one
    stopped.run()
    checkpoint = Path(stopped_cfg.output_dir) / "checkpoints/latest.ckpt"
    payload = torch.load(checkpoint, pickle_module=dill)
    provenance = dill.loads(payload["pickles"]["data_provenance"])
    provenance["protocol"] = manifest_fingerprint(training_protocol(payload["cfg"], runtime_independent=False))
    payload["pickles"]["data_provenance"] = dill.dumps(provenance)
    torch.save(payload, checkpoint, pickle_module=dill)
    cache_dir = str(tmp_path / "decoded")
    build_decoded_cache(FullSensorManiFeelDataset(**dataset_arguments), cache_dir, workers=2)
    cached_cfg = copy.deepcopy(stopped_cfg)
    cached_cfg.resume = str(checkpoint)
    cached_cfg.dataset.decoded_cache_dir = cache_dir
    cached_cfg.dataset.decoded_cache_mode = "ram"
    cached_cfg.dataloader.num_workers = 2
    cached = FullSensorWorkspace(cached_cfg, cached_cfg.output_dir)
    cached.run()
    assert cached.global_step == direct.global_step == 4
    for model_name in ("model", "ema_model"):
        for key, value in getattr(direct, model_name).state_dict().items():
            torch.testing.assert_close(value, getattr(cached, model_name).state_dict()[key], rtol=0, atol=0)
    changed_cfg = copy.deepcopy(cached_cfg)
    changed_cfg.dataloader.batch_size += 1
    with pytest.raises(ValueError, match="Resume rejected"):
        FullSensorWorkspace(changed_cfg, str(tmp_path / "wrong_batch")).run()


def test_workspace_resumes_exactly_and_rejects_changed_provenance(dataset_arguments: dict, tmp_path: Path) -> None:
    from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace

    cfg = workspace_config(dataset_arguments, tmp_path / "continuous")
    continuous = FullSensorWorkspace(cfg, cfg.output_dir)
    continuous.run()
    interrupted_cfg = copy.deepcopy(cfg)
    interrupted_cfg.output_dir = str(tmp_path / "interrupted")
    interrupted = FullSensorWorkspace(interrupted_cfg, interrupted_cfg.output_dir)
    original_save = interrupted._save

    def interrupt_at_boundary(tag: str) -> str:
        path = original_save(tag)
        if tag == "epoch1":
            raise RuntimeError("Simulated epoch-boundary interruption")
        return path

    interrupted._save = interrupt_at_boundary
    with pytest.raises(RuntimeError, match="Simulated"):
        interrupted.run()
    resumed_cfg = copy.deepcopy(interrupted_cfg)
    resumed_cfg.resume = str(Path(interrupted_cfg.output_dir) / "checkpoints/latest.ckpt")
    resumed = FullSensorWorkspace(resumed_cfg, resumed_cfg.output_dir)
    resumed.run()
    assert resumed.global_step == continuous.global_step == 4
    assert resumed.epoch == continuous.epoch == 2
    assert resumed.ema_step == continuous.ema_step == 4
    for model_name in ("model", "ema_model"):
        for name, value in getattr(continuous, model_name).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, model_name).state_dict()[name], rtol=0, atol=0)
    batch = collate_samples(FullSensorManiFeelDataset(**dataset_arguments), [0, 1])
    loaded = FullSensorWorkspace.create_from_checkpoint(resumed_cfg.resume, map_location="cpu")
    loaded.ema_model.eval()
    expected = resumed.ema_model.paired_denoising_losses(batch, generator=torch.Generator().manual_seed(7))
    actual = loaded.ema_model.paired_denoising_losses(batch, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(expected["losses"], actual["losses"], rtol=0, atol=0)
    changed_cfg = copy.deepcopy(resumed_cfg)
    changed_cfg.dataset.seed += 1
    with pytest.raises(ValueError, match="Resume rejected"):
        FullSensorWorkspace(changed_cfg, str(tmp_path / "rejected")).run()


@pytest.mark.parametrize("use_ema", [True, False])
def test_workspace_validation_uses_ema_and_sample_weighting(
    dataset_arguments: dict, tmp_path: Path, use_ema: bool,
) -> None:
    from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace

    cfg = workspace_config(dataset_arguments, tmp_path / "validation")
    cfg.training.max_val_steps = None
    cfg.training.use_ema = use_ema
    workspace = FullSensorWorkspace(cfg, cfg.output_dir)
    dataset = FullSensorManiFeelDataset(**dataset_arguments).get_validation_dataset()

    def forbidden_raw(*args, **kwargs):
        raise AssertionError("Validation must not mix raw weights with EMA")

    def deterministic_paired(batch, **kwargs):
        return {"losses": batch["source_index"].float().unsqueeze(1).expand(-1, 10)}

    selected = workspace.ema_model if use_ema else workspace.model
    if use_ema:
        workspace.model.paired_denoising_losses = forbidden_raw
    selected.paired_denoising_losses = deterministic_paired
    metrics = workspace.validate(dataset, torch.device("cpu"))
    expected = np.mean([int(dataset[index]["source_index"]) for index in range(len(dataset))])
    assert metrics["validation/micro/full"] == pytest.approx(expected)
    assert metrics["validation/task_coverage"] == 2


@pytest.mark.parametrize("interrupt_step", [1, 2, 3])
def test_workspace_mid_epoch_resume_preserves_weights_and_metrics(
    dataset_arguments: dict, tmp_path: Path, interrupt_step: int,
) -> None:
    import json

    from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace

    cfg = workspace_config(dataset_arguments, tmp_path / "continuous_mid")
    cfg.training.checkpoint_first_step = 1
    cfg.training.checkpoint_every_steps = 1
    cfg.training.log_every_steps = 1
    continuous = FullSensorWorkspace(cfg, cfg.output_dir)
    continuous.run()
    interrupted_cfg = copy.deepcopy(cfg)
    interrupted_cfg.output_dir = str(tmp_path / "interrupted_mid")
    interrupted = FullSensorWorkspace(interrupted_cfg, interrupted_cfg.output_dir)
    original_step = interrupted.optimizer.step

    def stop_after_update(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if interrupted.global_step + 1 == interrupt_step:
            interrupted.request_stop("test_signal")
        return result

    interrupted.optimizer.step = stop_after_update
    interrupted.run()
    assert interrupted.global_step == interrupt_step
    assert (Path(interrupted_cfg.output_dir) / "checkpoints/interrupt.ckpt").is_file()
    resumed_cfg = copy.deepcopy(interrupted_cfg)
    resumed_cfg.resume = str(Path(interrupted_cfg.output_dir) / "checkpoints/latest.ckpt")
    resumed = FullSensorWorkspace(resumed_cfg, resumed_cfg.output_dir)
    resumed.run()
    assert resumed.epoch == 2 and resumed.global_step == 4 and resumed.batch_in_epoch == 0
    for model_name in ("model", "ema_model"):
        for name, value in getattr(continuous, model_name).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, model_name).state_dict()[name], rtol=0, atol=0)
    expected = [json.loads(line) for line in (Path(cfg.output_dir) / "metrics.jsonl").read_text().splitlines()]
    actual = [json.loads(line) for line in (Path(resumed_cfg.output_dir) / "metrics.jsonl").read_text().splitlines()]
    assert len(expected) == len(actual) == 2
    for expected_epoch, actual_epoch in zip(expected, actual):
        for key, value in expected_epoch.items():
            if key not in ("train/seconds", "train/samples_per_second") and not key.startswith("runtime/"):
                assert actual_epoch[key] == value


def test_resume_sampler_skips_consumed_frames_without_changing_order(dataset_arguments: dict, tmp_path: Path) -> None:
    from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace

    cfg = workspace_config(dataset_arguments, tmp_path / "sampler")
    workspace = FullSensorWorkspace(cfg, cfg.output_dir)
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    for num_workers in (0, 2):
        cfg.dataloader.num_workers = num_workers
        complete = list(workspace._loader(dataset, True, epoch=3))
        remainder = list(workspace._loader(dataset, True, epoch=3, start_batch=2))
        assert len(remainder) == len(complete) - 2
        for expected, actual in zip(complete[2:], remainder):
            torch.testing.assert_close(expected["source_index"], actual["source_index"], rtol=0, atol=0)
            torch.testing.assert_close(expected["task_index"], actual["task_index"], rtol=0, atol=0)
