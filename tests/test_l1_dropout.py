"""Verify L1 mask interventions, baseline parity and exact mask-RNG continuation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zarr")
from omegaconf import OmegaConf

from adavip.manifeel.full_sensor_schema import SENSOR_KEYS, condition_presence
from adavip.manifeel.l1_dropout import L1DropoutWorkspace
from adavip.manifeel.language_conditioning_workspace import LanguageConditioningWorkspace
from test_language_conditioning import data_args, small_config


def dropout_config(data_args: dict, output: Path) -> object:
    """Use the real dropout implementation with a small policy and real Zarr loader."""
    cfg = small_config(data_args, output)
    cfg._target_ = "adavip.manifeel.l1_dropout.L1DropoutWorkspace"
    cfg.policy.obs_encoder._target_ = "adavip.manifeel.l1_dropout.L1DropoutObsEncoder"
    cfg.policy.full_input_probability = 0.9
    return cfg


def test_full_input_exactly_matches_original_l1(data_args: dict, tmp_path: Path) -> None:
    original = LanguageConditioningWorkspace(small_config(data_args, tmp_path / "original"))
    changed = L1DropoutWorkspace(dropout_config(data_args, tmp_path / "dropout"))
    assert dict(original.model.named_parameters()).keys() == dict(changed.model.named_parameters()).keys()
    for name, value in original.model.named_parameters():
        torch.testing.assert_close(value, dict(changed.model.named_parameters())[name], rtol=0, atol=0)
    observations = {key: torch.randn(2, 2, *entry["shape"]) for key, entry in
                    data_args["shape_meta"]["obs"].items()}
    observations["task_embedding"][:] = observations["task_embedding"][:, :1].clone()
    expected = original.model.obs_encoder(observations)
    torch.testing.assert_close(changed.model.obs_encoder(observations), expected, rtol=0, atol=0)
    torch.testing.assert_close(changed.model.obs_encoder(observations, torch.ones(2, 7, dtype=torch.bool)),
                               expected, rtol=0, atol=0)


def test_all_nine_masks_remove_information_and_gradients(data_args: dict, tmp_path: Path) -> None:
    workspace = L1DropoutWorkspace(dropout_config(data_args, tmp_path / "masks"))
    encoder = workspace.model.obs_encoder
    presence = torch.from_numpy(condition_presence())[1:]
    observations = {key: torch.randn(9, 2, *entry["shape"], requires_grad=True)
                    for key, entry in data_args["shape_meta"]["obs"].items()}
    # Language is a constant instruction within each window.
    observations["task_embedding"] = torch.randn(9, 1, 4).expand(-1, 2, -1).clone().requires_grad_()
    changed = {key: value.detach().clone() for key, value in observations.items()}
    for column, key in enumerate(SENSOR_KEYS):
        changed[key][~presence[:, column]] += 20
    expected = encoder(observations, presence)
    torch.testing.assert_close(encoder(changed, presence), expected, rtol=0, atol=0)
    assert not torch.equal(encoder(changed), encoder(observations))
    expected.square().sum().backward()
    for column, key in enumerate(SENSOR_KEYS):
        assert observations[key].grad[~presence[:, column]].abs().sum() == 0
        assert observations[key].grad[presence[:, column]].abs().sum() > 0


def test_drop10_distribution_and_rng_isolation(data_args: dict, tmp_path: Path) -> None:
    workspace = L1DropoutWorkspace(dropout_config(data_args, tmp_path / "rng"))
    rng = torch.get_rng_state().clone()
    ids = workspace._sample_training_conditions(100000, torch.device("cpu"))
    counts = torch.bincount(ids, minlength=10).double() / len(ids)
    torch.testing.assert_close(counts, workspace._dropout_probabilities, rtol=0, atol=0.003)
    assert torch.equal(rng, torch.get_rng_state())


def test_exact_dropout_optimizer_ema_resume(data_args: dict, tmp_path: Path) -> None:
    cfg = dropout_config(data_args, tmp_path / "complete")
    complete = L1DropoutWorkspace(cfg, cfg.output_dir)
    complete.run()
    stopped_cfg = dropout_config(data_args, tmp_path / "stopped")
    stopped_cfg.runtime.stop_after_steps = 3
    stopped = L1DropoutWorkspace(stopped_cfg, stopped_cfg.output_dir)
    stopped.run()
    resumed_cfg = copy.deepcopy(stopped_cfg)
    resumed_cfg.runtime.stop_after_steps = 0
    resumed_cfg.resume = str(Path(stopped_cfg.output_dir) / "checkpoints/latest.ckpt")
    resumed = L1DropoutWorkspace(resumed_cfg, resumed_cfg.output_dir)
    resumed.run()
    assert complete.global_step == resumed.global_step == 18
    assert complete.cumulative_task_counts == resumed.cumulative_task_counts
    for attr in ("model", "ema_model"):
        for name, value in getattr(complete, attr).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, attr).state_dict()[name], rtol=0, atol=0)
    for index, state in complete.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][index][name], rtol=0, atol=0)
    assert torch.equal(complete._noise_generator.get_state(), resumed._noise_generator.get_state())
    assert torch.equal(complete._dropout_generator.get_state(), resumed._dropout_generator.get_state())
    assert (Path(resumed_cfg.output_dir) / "checkpoints/epoch2.ckpt").exists()


def test_formal_config() -> None:
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/manifeel/train_l1_dropout.yaml")
    assert cfg.variant == "l1" and cfg.policy.full_input_probability == 0.9
    assert cfg.dataloader.batch_size == 32 and cfg.training.num_epochs == 100
    assert list(cfg.training.checkpoint_epochs) == [20, 40, 60, 80, 100]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Normalizer device regression requires CUDA")
def test_preflight_cuda_normalizer_and_no_updates(data_args: dict, tmp_path: Path) -> None:
    from scripts.manifeel.train_l1_dropout import preflight
    import json

    cfg = dropout_config(data_args, tmp_path / "gpu_preflight")
    Path(cfg.output_dir).mkdir()
    workspace = L1DropoutWorkspace(cfg, cfg.output_dir)
    initial = {name: value.detach().cpu().clone() for name, value in workspace.model.named_parameters()}
    mask_rng = workspace._dropout_generator.get_state().clone()
    torch.cuda.get_rng_state(0)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    preflight(workspace)
    assert all(parameter.device == workspace.model.device for parameter in workspace.model.normalizer.parameters())
    for name, value in workspace.model.named_parameters():
        if name in initial:
            torch.testing.assert_close(value.detach().cpu(), initial[name], rtol=0, atol=0)
    assert not workspace.optimizer.state
    assert workspace.global_step == workspace.ema_step == 0
    assert torch.equal(mask_rng, workspace._dropout_generator.get_state())
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state(0))
    report = json.loads((Path(cfg.output_dir) / "preflight.json").read_text())
    assert report["passed"] and report["normalizer_device"] == "cuda:0"
    assert report["normalized_observation_device"] == "cuda:0"
