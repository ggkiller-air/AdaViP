"""Prepare saved single-task ManiFeel configs for local evaluation."""

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict


FILM_RUNNER_TARGET = "adavip.manifeel.film_fusion_runner.FilmFusionRunner"
DEVICE_RUNNER_TARGET = "adavip.manifeel.device_runner.DeviceAwareManifeelRunner"
HYPER_RESNET_ENCODER_TARGET = "adavip.manifeel.hyper_resnet_obs_encoder.HyperResNetObsEncoder"


def configure_single_task_evaluation(
    cfg: DictConfig,
    *,
    device: str,
    n_test: int,
    n_test_vis: int,
    max_steps: int,
    test_start_seed: int,
) -> None:
    """Set rollout options and restore saved observation keys and asset paths."""
    with open_dict(cfg):
        _configure_hyper_resnet_vision_key(cfg)
        if cfg.task.env_runner._target_ == FILM_RUNNER_TARGET:
            _configure_film_assets(cfg)
        else:
            cfg.task.env_runner._target_ = DEVICE_RUNNER_TARGET
        cfg.task.env_runner.device = device
        cfg.task.env_runner.n_test = n_test
        cfg.task.env_runner.n_test_vis = n_test_vis
        cfg.task.env_runner.max_steps = max_steps
        cfg.task.env_runner.test_start_seed = test_start_seed


def _configure_hyper_resnet_vision_key(cfg: DictConfig) -> None:
    if OmegaConf.select(cfg, "policy.obs_encoder._target_") != HYPER_RESNET_ENCODER_TARGET:
        return
    encoder = cfg.policy.obs_encoder
    if "vision_key" in encoder:
        return
    tactile_key = encoder.get("tactile_key", "right_tactile_camera_taxim")
    observations = encoder.shape_meta.obs
    vision_keys = [
        key for key, metadata in observations.items()
        if metadata.get("type", "low_dim") == "rgb" and key != tactile_key
    ]
    if len(vision_keys) != 1:
        raise ValueError(f"Cannot infer a unique HyperResNet vision key: {vision_keys}")
    encoder.vision_key = vision_keys[0]


def _configure_film_assets(cfg: DictConfig) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    asset_root = Path(
        os.environ.get("ADAVIP_FILM_FUSION_ASSET_ROOT")
        or Path(os.environ.get("MANIFEEL_CHECKPOINT_ROOT", "/data/wangzihao/checkpoints/manifeel"))
        / "film_fusion"
    )
    asset_specs = (
        (
            "clip_encoder.checkpoint_path",
            "ADAVIP_FILM_FUSION_CLIP_CHECKPOINT",
            asset_root / "pretrained/film_fusion/clip/RN50.pt",
            "",
        ),
        (
            "clip_encoder.source_root",
            "ADAVIP_FILM_FUSION_CLIP_SOURCE_ROOT",
            repo_root / "third_party/CLIP",
            "clip/model.py",
        ),
        (
            "sparsh_encoder.checkpoint_path",
            "ADAVIP_FILM_FUSION_SPARSH_CHECKPOINT",
            asset_root / "pretrained/film_fusion/sparsh/dino_vitbase.ckpt",
            "",
        ),
        (
            "sparsh_encoder.source_root",
            "ADAVIP_FILM_FUSION_SPARSH_SOURCE_ROOT",
            repo_root / "third_party/sparsh",
            "tactile_ssl/model/__init__.py",
        ),
    )
    for attribute, env_name, fallback, required_file in asset_specs:
        config_key = f"policy.obs_encoder.{attribute}"
        saved_path = OmegaConf.select(cfg, config_key)
        override = os.environ.get(env_name)
        candidates = (
            [Path(override).expanduser()]
            if override
            else [Path(str(saved_path)).expanduser(), fallback]
        )
        selected = next(
            (
                candidate
                for candidate in candidates
                if (candidate / required_file).is_file()
            ),
            None,
        )
        if selected is None:
            raise FileNotFoundError(
                f"FiLM asset {config_key} not found; set {env_name} "
                f"or run scripts/manifeel/download_power_plug_adavip_film_fusion_assets.sh"
            )
        OmegaConf.update(cfg, config_key, str(selected), force_add=False)
