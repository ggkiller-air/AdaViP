"""Regression tests for the single-task teacher protocol."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")
pytest.importorskip("diffusion_policy")
from omegaconf import OmegaConf
from torch import nn
from diffusers import DDPMScheduler

from adavip.manifeel.single_task_teacher import (
    SingleTaskTeacherDataset, SingleTaskTeacherEncoder, SingleTaskTeacherPolicy, sensor_conditions,
)
from adavip.manifeel.single_task_teacher_workspace import SingleTaskTeacherWorkspace
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from manifeel.dataset.manifeel_dataset_equidp import ManifeelDataset
from scripts.manifeel.train_single_task_teacher import make_config


def tiny_policy(sensors: str = "five", action_dim: int = 6, probability: float = 0.6):
    cfg = make_config("power_plug_insertion", sensors, probability < 1, Path("unused"))
    shape = OmegaConf.to_container(cfg.shape_meta)
    shape["action"]["shape"] = [action_dim]
    for entry in shape["obs"].values():
        if entry["type"] == "rgb":
            entry["shape"] = [3, 8, 8]
    encoder = SingleTaskTeacherEncoder(shape, token_dim=8, image_feature_dim=8,
        rgb_model=nn.Sequential(nn.Conv2d(3, 8, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()))
    policy = SingleTaskTeacherPolicy(shape, encoder, DDPMScheduler(num_train_timesteps=10),
        full_input_probability=probability, horizon=8, n_action_steps=2, num_inference_steps=2,
        diffusion_step_embed_dim=8, down_dims=(8, 16), kernel_size=3, n_groups=4)
    normalizer = LinearNormalizer()
    for key in (*shape["obs"], "action"):
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    batch = {"obs": {key: torch.randn(2, 2, *entry["shape"]) for key, entry in shape["obs"].items()},
             "action": torch.randn(2, 8, action_dim)}
    return policy, batch


@pytest.mark.parametrize("sensors,dimensions", [("five", 7), ("two", 2)])
def test_mask_contract_and_always_present_state(sensors: str, dimensions: int) -> None:
    policy, batch = tiny_policy(sensors)
    names, masks = sensor_conditions(policy.obs_encoder.image_keys)
    assert len(names) == dimensions + 1
    assert len(torch.unique(masks, dim=0)) == dimensions + 1
    assert policy.condition_probabilities[0].item() == pytest.approx(0.6)
    assert policy.condition_probabilities[1:].sum().item() == pytest.approx(0.4)
    tokens = policy.obs_encoder.encode_tokens(batch["obs"])
    assert tokens.shape[1] == len(policy.obs_encoder.image_keys) + 1
    altered = copy.deepcopy(batch["obs"])
    first_key = policy.obs_encoder.image_keys[0]
    altered[first_key] += 100
    presence = masks[1].expand(2, -1)
    torch.testing.assert_close(policy.obs_encoder(batch["obs"], presence), policy.obs_encoder(altered, presence), rtol=0, atol=0)
    altered["state"] += torch.randn_like(altered["state"])
    assert not torch.equal(policy.obs_encoder(batch["obs"], presence), policy.obs_encoder(altered, presence))


@pytest.mark.parametrize("sensors,action_dim", [("five", 6), ("five", 7), ("two", 6), ("two", 7)])
def test_all_masks_backward_and_paired_inference_contract(sensors: str, action_dim: int) -> None:
    policy, batch = tiny_policy(sensors, action_dim)
    for index in range(len(policy.condition_names)):
        policy.zero_grad(set_to_none=True)
        loss = policy.losses(batch, torch.full((2,), index, dtype=torch.long)).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert policy.obs_encoder.state_projection[0].weight.grad.abs().sum() > 0
        removed = ~policy.condition_presence[index]
        for sensor_index, key in enumerate(policy.obs_encoder.image_keys):
            gradient = policy.obs_encoder.image_encoders[key][0].weight.grad
            assert (gradient.abs().sum() == 0) if removed[sensor_index] else (gradient.abs().sum() > 0)
    policy.eval()
    a = policy.paired_denoising_losses(batch, torch.Generator().manual_seed(10))
    b = policy.paired_denoising_losses(batch, torch.Generator().manual_seed(10))
    torch.testing.assert_close(a["losses"], b["losses"], rtol=0, atol=0)
    torch.testing.assert_close(a["utility"], a["losses"][:, 1:] - a["losses"][:, :1])
    action = policy.predict_action(batch["obs"])["action"]
    assert action.shape == (2, 2, action_dim) and torch.isfinite(action).all()


def test_no_dropout_initialization_and_noise_stream_match() -> None:
    torch.manual_seed(42)
    a, batch = tiny_policy(probability=0.6)
    torch.manual_seed(42)
    b, _ = tiny_policy(probability=1.)
    for key, value in a.state_dict().items():
        if key != "condition_probabilities":
            torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    assert (b.sample_condition_ids(100, torch.Generator().manual_seed(4)) == 0).all()
    a.sample_condition_ids(100, torch.Generator().manual_seed(4))
    torch.manual_seed(99)
    x = a.losses(batch, torch.zeros(2, dtype=torch.long))
    torch.manual_seed(99)
    y = b.losses(batch, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.fixture
def tiny_store(tmp_path: Path) -> tuple[dict, Path]:
    cfg = make_config("power_plug_insertion", "two", True, tmp_path / "run")
    shape = OmegaConf.to_container(cfg.shape_meta)
    for entry in shape["obs"].values():
        if entry["type"] == "rgb":
            entry["shape"] = [3, 8, 8]
    path = tmp_path / "store"
    group = zarr.open_group(str(path), "w")
    group.create_group("meta").create_dataset("episode_ends", data=np.array([12, 24, 36]))
    arrays = group.create_group("data")
    rng = np.random.default_rng(42)
    for key, entry in shape["obs"].items():
        tail = (8, 8, 3) if entry["type"] == "rgb" else (7,)
        arrays.create_dataset(key, data=rng.random((36, *tail), dtype=np.float32))
    arrays.create_dataset("action", data=rng.random((36, 6), dtype=np.float32))
    return shape, path


def test_dataset_matches_official_windows_and_ram_transport(tiny_store) -> None:
    shape, path = tiny_store
    dataset = SingleTaskTeacherDataset(shape, str(path), horizon=8, n_action_steps=2, val_ratio=0.34)
    official = ManifeelDataset(shape, str(path), horizon=8, pad_before=1, pad_after=1,
                               val_ratio=0.34, max_train_episodes=50)
    np.testing.assert_array_equal(dataset.sampler.indices, official.sampler.indices)
    original = [dataset[index] for index in range(len(dataset))]
    dataset.load_to_ram()
    for index in range(len(dataset)):
        current = dataset[index]
        for key in current["obs"]:
            torch.testing.assert_close(current["obs"][key], official[index]["obs"][key], rtol=0, atol=0)
            torch.testing.assert_close(current["obs"][key], original[index]["obs"][key], rtol=0, atol=0)
        torch.testing.assert_close(current["action"], official[index]["action"], rtol=0, atol=0)
    validation = dataset.get_validation_dataset()
    assert not np.any(dataset.train_mask & validation.train_mask)
    normalizer = dataset.get_normalizer()
    ends = dataset.replay_buffer.episode_ends
    frames = np.concatenate([dataset.replay_buffer["action"][0 if i == 0 else ends[i - 1]:ends[i]]
                             for i in np.flatnonzero(dataset.train_mask)])
    np.testing.assert_allclose(normalizer["action"].get_input_stats()["max"], frames.max(0))


def test_exact_mid_epoch_resume_model_ema_adam_scheduler_rng(tiny_store, tmp_path: Path) -> None:
    shape, path = tiny_store
    cfg = make_config("power_plug_insertion", "two", True, tmp_path / "continuous")
    cfg.shape_meta = shape
    cfg.policy.shape_meta = shape
    cfg.policy.obs_encoder.shape_meta = shape
    cfg.task.dataset.shape_meta = shape
    cfg.task.dataset.zarr_path = str(path)
    cfg.task.dataset.horizon = cfg.policy.horizon = cfg.horizon = 8
    cfg.task.dataset.n_action_steps = cfg.policy.n_action_steps = cfg.n_action_steps = 2
    cfg.policy.obs_encoder.token_dim = 8
    cfg.policy.obs_encoder.image_feature_dim = 8
    cfg.policy.obs_encoder.rgb_model = {"_target_": "torch.nn.Sequential", "_args_": [
        {"_target_": "torch.nn.Conv2d", "in_channels": 3, "out_channels": 8, "kernel_size": 1},
        {"_target_": "torch.nn.AdaptiveAvgPool2d", "output_size": 1}, {"_target_": "torch.nn.Flatten"}]}
    cfg.policy.down_dims = [8, 16]
    cfg.policy.diffusion_step_embed_dim = 8
    cfg.policy.n_groups = 4
    cfg.policy.kernel_size = 3
    cfg.policy.noise_scheduler.num_train_timesteps = 10
    cfg.policy.num_inference_steps = 2
    cfg.training.device = "cpu"
    cfg.training.num_epochs = 2
    cfg.runtime.validate_every = 1
    cfg.runtime.checkpoint_every_epochs = 1
    cfg.runtime.log_every_steps = 1
    cfg.dataloader.num_workers = cfg.val_dataloader.num_workers = 0
    first = SingleTaskTeacherWorkspace(cfg)
    first.run()
    split_cfg = copy.deepcopy(cfg)
    split_cfg.output_dir = str(tmp_path / "split")
    split = SingleTaskTeacherWorkspace(split_cfg)
    split.run(stop_after_steps=1)
    split_cfg.resume = str(tmp_path / "split/checkpoints/latest.ckpt")
    resumed = SingleTaskTeacherWorkspace(split_cfg)
    resumed.run()
    assert resumed.epoch == first.epoch == 2
    assert resumed.global_step == first.global_step
    for branch in ("model", "ema_model"):
        for key, value in getattr(first, branch).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, branch).state_dict()[key], rtol=0, atol=0)
    for key, value in first.optimizer.state_dict()["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(tensor, resumed.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
    assert first.scheduler.state_dict() == resumed.scheduler.state_dict()
    torch.testing.assert_close(first.rng_state["torch"], resumed.rng_state["torch"], rtol=0, atol=0)
    torch.testing.assert_close(first.rng_state["condition"], resumed.rng_state["condition"], rtol=0, atol=0)
    import dill
    payload = torch.load(split_cfg.resume, map_location="cpu", pickle_module=dill)
    inference = SingleTaskTeacherWorkspace(split_cfg)
    inference.load_payload(payload)
    assert inference.scheduler is None
    obs = torch.utils.data.default_collate([SingleTaskTeacherDataset(shape, str(path), horizon=8,
                                           n_action_steps=2)[0]])["obs"]
    torch.manual_seed(101)
    expected = first.ema_model.predict_action(obs)["action"]
    torch.manual_seed(101)
    actual = inference.ema_model.predict_action(obs)["action"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_all_twelve_configs_match_requested_official_optimization(tmp_path: Path) -> None:
    for task in ("power_plug_insertion", "usb_insertion", "ball_sorting"):
        for sensors in ("five", "two"):
            for dropout in (False, True):
                cfg = make_config(task, sensors, dropout, tmp_path / task)
                assert cfg.training.num_epochs == 400
                assert cfg.dataloader.batch_size == cfg.val_dataloader.batch_size == 8
                assert cfg.optimizer.lr == 8e-5
                assert cfg.policy.noise_scheduler.num_train_timesteps == cfg.policy.num_inference_steps == 100
                assert cfg.policy.full_input_probability == (0.6 if dropout else 1.)
                assert "task_embedding" not in cfg.shape_meta.obs
                assert cfg.shape_meta.action.shape[0] == (7 if task == "ball_sorting" else 6)
                if sensors == "two":
                    camera = "front" if task == "ball_sorting" else "wrist"
                    assert set(cfg.shape_meta.obs) == {camera, "right_tactile_camera_taxim", "state"}
                    shape = OmegaConf.to_container(cfg.shape_meta)
                    encoder = SingleTaskTeacherEncoder(shape, token_dim=8, image_feature_dim=8,
                        rgb_model=nn.Sequential(nn.Conv2d(3, 8, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()))
                    assert encoder.image_keys == (camera, "right_tactile_camera_taxim")
