"""Future alignment, independent heads, online gradients, EMA and exact continuation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zarr")
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from test_language_conditioning import data_args, small_config
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, TACTILE_KEYS
from adavip.manifeel.stage1_future import Stage1FutureHeads, gradient_diagnostics, stage1_losses
from adavip.manifeel.stage1_future_calibration import calibrate_future_weight
from adavip.manifeel.stage1_future_dataset import Stage1FutureDataset
from adavip.manifeel.stage1_future_workspace import Stage1FutureWorkspace


def future_config(data_args: dict, output: Path, experiment: str) -> object:
    """Use the real L1/U-Net trainer at small tensor sizes."""
    cfg = small_config(data_args, output)
    cfg._target_ = "adavip.manifeel.stage1_future_workspace.Stage1FutureWorkspace"
    cfg.experiment = experiment
    cfg.dataset._target_ = "adavip.manifeel.stage1_future_dataset.Stage1FutureDataset"
    cfg.dataset.experiment = experiment
    cfg.dataset.future_offset = 4
    cfg.dataset.raw_resolution = 8
    cfg.future = dict(offset=4, raw_resolution=8, momentum=0.996, head_seed=17042,
                      loss_weight=0.1 if experiment != "A" else 0.0, calibration_batches=2,
                      gradient_target_ratio=0.1, gradient_every_steps=3)
    return cfg


@pytest.mark.parametrize("experiment", ["A", "B", "C", "D", "E"])
def test_alignment_and_episode_end(data_args: dict, experiment: str) -> None:
    dataset = Stage1FutureDataset(**data_args, experiment=experiment, future_offset=4, raw_resolution=8)
    for index in (0, 5, 6, 9, 10, 16, 17, 20, 21, 28, 29, 32, 33, 65):
        sample = dataset[index]
        task, _, start, end, timestep = dataset._resolve_index(index)
        target = start + timestep + 4
        assert bool(sample["future_valid"]) == (target < end)
        assert int(sample["future_index"]) == (target if target < end else -1)
        for key, value in sample["future_obs"].items():
            if target < end:
                expected = torch.from_numpy(dataset._read_images(
                    dataset.groups[task]["data"][key], np.array([target]), key))[0]
                if experiment in ("B", "D"):
                    expected = F.interpolate(expected[None], (8, 8), mode="bilinear",
                                             align_corners=False, antialias=True)[0]
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            else:
                assert not value.any()


@pytest.mark.parametrize("experiment", ["A", "B", "C", "D", "E"])
def test_common_action_independent_heads_gradients_and_ema(data_args: dict, tmp_path: Path,
                                                         experiment: str) -> None:
    workspace = Stage1FutureWorkspace(future_config(data_args, tmp_path / experiment, experiment))
    policy, auxiliary = workspace.model, workspace.future
    dataset = Stage1FutureDataset(**data_args, experiment=experiment, raw_resolution=8)
    policy.set_normalizer(dataset.get_normalizer())
    batch = torch.utils.data.default_collate([dataset[2], dataset[37]])
    sample = policy.prepare_denoising_sample(batch)
    losses = stage1_losses(policy, auxiliary, batch, sample)
    original = policy.compute_loss_per_sample(batch, torch.zeros(2, dtype=torch.long), sample)
    torch.testing.assert_close(losses["action_per_sample"], original, rtol=0, atol=0)
    assert losses["z_t"].shape == (2, 8)
    assert len(auxiliary.heads) == (0 if experiment == "A" else 2 if experiment in ("B", "C") else 5)
    pointers = [set(p.data_ptr() for p in head.parameters()) for head in auxiliary.heads.values()]
    for i, left in enumerate(pointers):
        assert all(left.isdisjoint(right) for right in pointers[i + 1:])
    if experiment != "A":
        result = gradient_diagnostics(losses, policy, include_resnets=True)
        assert all(value > 0 for value in result["future_resnet_first_weight_grad_norms"].values())
        language = list(policy.obs_encoder.language_head.parameters())
        assert all(g is None for g in torch.autograd.grad(losses["future"], language,
                                                        retain_graph=True, allow_unused=True))
        expected_t = sum(losses["sensors"][key] for key in TACTILE_KEYS) / 2
        expected = expected_t if experiment in ("B", "C") else 0.5 * (
            expected_t + sum(losses["sensors"][key] for key in IMAGE_KEYS[:3]) / 3)
        torch.testing.assert_close(losses["future"], expected)
    old_online = {key: next(m.parameters()).detach().clone()
                  for key, m in policy.obs_encoder.image_encoders.items()}
    old_target = {key: next(m.parameters()).detach().clone()
                  for key, m in auxiliary.target_encoders.items()}
    (losses["action"] + 0.1 * losses["future"]).backward()
    assert all(p.grad is None for p in auxiliary.target_encoders.parameters())
    assert all(torch.isfinite(p.grad).all() for p in workspace.trainable_parameters() if p.grad is not None)
    workspace.optimizer.step()
    auxiliary.update_targets(policy.obs_encoder)
    for key, module in policy.obs_encoder.image_encoders.items():
        assert not torch.equal(old_online[key], next(module.parameters()))
    for key, module in auxiliary.target_encoders.items():
        expected = old_target[key] * 0.996 + next(policy.obs_encoder.image_encoders[key].parameters()) * 0.004
        torch.testing.assert_close(next(module.parameters()), expected)
        assert not module.training
    # Pure inference remains the original policy and requires no auxiliary module.
    assert not hasattr(policy, "future")
    assert policy.predict_action(batch["obs"])["action"].shape == (2, 2, 7)


def test_initialization_is_identical_across_five_groups(data_args: dict, tmp_path: Path) -> None:
    workspaces = [Stage1FutureWorkspace(future_config(data_args, tmp_path / e, e)) for e in "ABCDE"]
    assert len({w.policy_initialization for w in workspaces}) == 1
    for other in workspaces[1:]:
        for key, value in workspaces[0].model.state_dict().items():
            torch.testing.assert_close(value, other.model.state_dict()[key], rtol=0, atol=0)


def test_all_invalid_future_does_not_evaluate_targets(data_args: dict, tmp_path: Path) -> None:
    workspace = Stage1FutureWorkspace(future_config(data_args, tmp_path / "invalid", "E"))
    dataset = Stage1FutureDataset(**data_args, experiment="E", raw_resolution=8)
    batch = torch.utils.data.default_collate([dataset[9], dataset[20]])
    workspace.model.set_normalizer(dataset.get_normalizer())
    losses = stage1_losses(workspace.model, workspace.future, batch,
                          workspace.model.prepare_denoising_sample(batch))
    assert not losses["valid_count"] and losses["future"] == 0
    losses["future"].backward()
    assert all(p.grad is None for p in workspace.future.target_encoders.parameters())
    assert all(not p.grad.any() for p in workspace.trainable_parameters() if p.grad is not None)


def test_target_ema_normalization_buffers_and_eval_mode() -> None:
    encoder = nn.Module()
    encoder.token_dim, encoder.image_feature_dim = 8, 8
    encoder.image_encoders = nn.ModuleDict({key: nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8))
                                           for key in IMAGE_KEYS})
    auxiliary = Stage1FutureHeads(encoder, "E")
    auxiliary.train()
    assert all(not module.training for module in auxiliary.target_encoders.modules())
    for module in encoder.image_encoders.values():
        module[1].running_mean.fill_(2)
        module[1].running_var.fill_(3)
        module[1].num_batches_tracked.fill_(9)
    auxiliary.update_targets(encoder)
    for module in auxiliary.target_encoders.values():
        torch.testing.assert_close(module[1].running_mean, torch.full((8,), 0.008))
        torch.testing.assert_close(module[1].running_var, torch.full((8,), 1.008))
        assert module[1].num_batches_tracked == 9


def test_calibration_preserves_initial_state_and_rng(data_args: dict, tmp_path: Path) -> None:
    workspace = Stage1FutureWorkspace(future_config(data_args, tmp_path / "calibrate", "E"))
    dataset = Stage1FutureDataset(**data_args, experiment="E", raw_resolution=8)
    workspace.model.set_normalizer(dataset.get_normalizer())
    before = copy.deepcopy(workspace.model.state_dict())
    rng = torch.get_rng_state().clone()
    result = calibrate_future_weight(workspace.model, workspace.future, dataset,
                                     torch.device("cpu"), batch_size=8, batches=2)
    assert result["weight"] > 0 and result["optimizer_steps"] == result["target_ema_updates"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in workspace.trainable_parameters())
    for key, value in before.items():
        torch.testing.assert_close(value, workspace.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("experiment", ["A", "B", "C", "D", "E"])
def test_exact_future_checkpoint_resume(data_args: dict, tmp_path: Path, experiment: str) -> None:
    cfg = future_config(data_args, tmp_path / ("complete" + experiment), experiment)
    complete = Stage1FutureWorkspace(cfg, cfg.output_dir)
    complete.run()
    stopped_cfg = future_config(data_args, tmp_path / ("stopped" + experiment), experiment)
    stopped_cfg.runtime.stop_after_steps = 3
    stopped = Stage1FutureWorkspace(stopped_cfg, stopped_cfg.output_dir)
    stopped.run()
    resumed_cfg = copy.deepcopy(stopped_cfg)
    resumed_cfg.runtime.stop_after_steps = 0
    resumed_cfg.resume = str(Path(stopped_cfg.output_dir) / "checkpoints/latest.ckpt")
    resumed = Stage1FutureWorkspace(resumed_cfg, resumed_cfg.output_dir)
    resumed.run()
    for name in ("model", "ema_model", "future"):
        for key, value in getattr(complete, name).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, name).state_dict()[key], rtol=0, atol=0)
    assert complete.future_weight == resumed.future_weight
    assert complete.global_step == resumed.global_step == 18
    for i, state in complete.optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][i][key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        Stage1FutureWorkspace(cfg, cfg.output_dir).run()
