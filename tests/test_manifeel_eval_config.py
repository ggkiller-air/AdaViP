"""CPU-only coverage for saved single-task evaluation configuration."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from omegaconf import OmegaConf

from adavip.manifeel.eval_config import (
    DEVICE_RUNNER_TARGET,
    FILM_RUNNER_TARGET,
    HYPER_RESNET_ENCODER_TARGET,
    configure_single_task_evaluation,
)


def _checkpoint_config(runner_target: str):
    return OmegaConf.create(
        {
            "task": {"env_runner": {"_target_": runner_target}},
            "policy": {
                "obs_encoder": {
                    "clip_encoder": {
                        "checkpoint_path": "/missing/clip/RN50.pt",
                        "source_root": "/missing/CLIP",
                    },
                    "sparsh_encoder": {
                        "checkpoint_path": "/missing/sparsh/dino_vitbase.ckpt",
                        "source_root": "/missing/sparsh",
                    },
                }
            },
        }
    )


def _configure(cfg):
    configure_single_task_evaluation(
        cfg,
        device="cuda:4",
        n_test=10,
        n_test_vis=0,
        max_steps=500,
        test_start_seed=100000,
    )


def test_film_checkpoint_uses_local_assets(tmp_path, monkeypatch) -> None:
    for variable in (
        "ADAVIP_FILM_FUSION_CLIP_CHECKPOINT",
        "ADAVIP_FILM_FUSION_CLIP_SOURCE_ROOT",
        "ADAVIP_FILM_FUSION_SPARSH_CHECKPOINT",
        "ADAVIP_FILM_FUSION_SPARSH_SOURCE_ROOT",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ADAVIP_FILM_FUSION_ASSET_ROOT", str(tmp_path))
    for relative_path in (
        "pretrained/film_fusion/clip/RN50.pt",
        "pretrained/film_fusion/sparsh/dino_vitbase.ckpt",
    ):
        asset = tmp_path / relative_path
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"checkpoint")

    cfg = _checkpoint_config(FILM_RUNNER_TARGET)
    _configure(cfg)

    assert cfg.task.env_runner._target_ == FILM_RUNNER_TARGET
    assert cfg.task.env_runner.device == "cuda:4"
    assert cfg.task.env_runner.n_test == 10
    assert cfg.task.env_runner.n_test_vis == 0
    assert cfg.task.env_runner.max_steps == 500
    assert cfg.task.env_runner.test_start_seed == 100000
    assert cfg.policy.obs_encoder.clip_encoder.checkpoint_path == str(
        tmp_path / "pretrained/film_fusion/clip/RN50.pt"
    )
    assert cfg.policy.obs_encoder.sparsh_encoder.checkpoint_path == str(
        tmp_path / "pretrained/film_fusion/sparsh/dino_vitbase.ckpt"
    )
    assert cfg.policy.obs_encoder.clip_encoder.source_root.endswith("third_party/CLIP")
    assert cfg.policy.obs_encoder.sparsh_encoder.source_root.endswith("third_party/sparsh")


def test_film_checkpoint_reports_missing_explicit_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADAVIP_FILM_FUSION_CLIP_CHECKPOINT", str(tmp_path / "missing.pt"))
    with pytest.raises(FileNotFoundError, match="ADAVIP_FILM_FUSION_CLIP_CHECKPOINT"):
        _configure(_checkpoint_config(FILM_RUNNER_TARGET))


def test_baseline_checkpoint_keeps_device_aware_runner() -> None:
    cfg = _checkpoint_config("manifeel.env_runner.vistac_pih_runner_unit.ManifeelRunner")
    _configure(cfg)
    assert cfg.task.env_runner._target_ == DEVICE_RUNNER_TARGET
    assert cfg.task.env_runner.device == "cuda:4"
    assert cfg.policy.obs_encoder.clip_encoder.checkpoint_path == "/missing/clip/RN50.pt"
    assert "vision_key" not in cfg.policy.obs_encoder


@pytest.mark.parametrize("vision_key", ["wrist", "front"])
def test_hyper_resnet_infers_saved_vision_key(vision_key: str) -> None:
    cfg = _checkpoint_config("manifeel.env_runner.vistac_pih_runner_unit.ManifeelRunner")
    cfg.policy.obs_encoder = {
        "_target_": HYPER_RESNET_ENCODER_TARGET,
        "shape_meta": {"obs": {
            vision_key: {"type": "rgb", "shape": [3, 256, 256]},
            "right_tactile_camera_taxim": {"type": "rgb", "shape": [3, 256, 256]},
            "state": {"type": "low_dim", "shape": [7]},
        }},
    }
    OmegaConf.set_struct(cfg, True)
    _configure(cfg)
    assert cfg.policy.obs_encoder.vision_key == vision_key
    assert set(cfg.policy.obs_encoder.shape_meta.obs) == {
        vision_key, "right_tactile_camera_taxim", "state",
    }


def test_hyper_resnet_preserves_explicit_vision_key() -> None:
    cfg = _checkpoint_config("manifeel.env_runner.vistac_pih_runner_unit.ManifeelRunner")
    cfg.policy.obs_encoder = {
        "_target_": HYPER_RESNET_ENCODER_TARGET,
        "vision_key": "front",
    }
    _configure(cfg)
    assert cfg.policy.obs_encoder.vision_key == "front"


def test_hyper_resnet_rejects_ambiguous_vision_keys() -> None:
    cfg = _checkpoint_config("manifeel.env_runner.vistac_pih_runner_unit.ManifeelRunner")
    cfg.policy.obs_encoder = {
        "_target_": HYPER_RESNET_ENCODER_TARGET,
        "shape_meta": {"obs": {
            "front": {"type": "rgb"},
            "wrist": {"type": "rgb"},
            "right_tactile_camera_taxim": {"type": "rgb"},
        }},
    }
    with pytest.raises(ValueError, match="unique HyperResNet vision key"):
        _configure(cfg)


def test_film_runner_selects_physical_device_adapter(tmp_path, monkeypatch) -> None:
    film_module = pytest.importorskip("adavip.manifeel.film_fusion_runner")

    class StubRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.env = object()

    class StubDeviceRunner(StubRunner):
        pass

    unit_module = ModuleType("manifeel.env_runner.vistac_pih_runner_unit")
    unit_module.ManifeelRunner = StubRunner
    device_module = ModuleType("adavip.manifeel.device_runner")
    device_module.DeviceAwareUnitManifeelRunner = StubDeviceRunner
    monkeypatch.setitem(sys.modules, unit_module.__name__, unit_module)
    monkeypatch.setitem(sys.modules, device_module.__name__, device_module)

    runner = film_module.FilmFusionRunner(
        output_dir=str(tmp_path),
        shape_meta={"obs": {"right_tactile_camera_taxim_pair": {"shape": [6, 320, 240]}}},
        isaacgym_cfg_name="isaacgym_config_power_plug.yaml",
        device="cuda:4",
    )
    assert isinstance(runner._runner, StubDeviceRunner)
    assert runner._runner.kwargs["device"] == "cuda:4"
    assert runner._runner.kwargs["n_obs_steps"] == 7
    assert "right_tactile_camera_taxim" in runner._runner.kwargs["shape_meta"]["obs"]
