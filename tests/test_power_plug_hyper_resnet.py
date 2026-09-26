"""Regression checks for the trainable Power Plug HyperResNet variants."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

from adavip.manifeel.hyper_resnet_obs_encoder import HyperResNetObsEncoder


REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_META = {
    "obs": {
        "wrist": {"shape": [3, 8, 8], "type": "rgb"},
        "right_tactile_camera_taxim": {"shape": [3, 8, 8], "type": "rgb"},
        "state": {"shape": [7], "type": "low_dim"},
    },
    "action": {"shape": [6]},
}


def _fake_resnet() -> nn.Module:
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512))


@pytest.mark.parametrize("use_fusion", [False, True])
def test_hyper_resnet_starts_at_baseline_and_trains_both_streams(use_fusion: bool) -> None:
    baseline = MultiImageObsEncoder(shape_meta=SHAPE_META, rgb_model=_fake_resnet())
    encoder = HyperResNetObsEncoder(
        shape_meta=SHAPE_META,
        rgb_model=_fake_resnet(),
        use_fusion=use_fusion,
    )
    encoder.load_state_dict(baseline.state_dict(), strict=False)
    observations = {
        "wrist": torch.rand(2, 3, 8, 8),
        "right_tactile_camera_taxim": torch.rand(2, 3, 8, 8),
        "state": torch.rand(2, 7),
    }
    torch.testing.assert_close(encoder(observations), baseline(observations))
    assert encoder.output_shape() == (1031,)
    assert encoder.alpha.item() == pytest.approx(0.01)

    encoder(observations).square().mean().backward()
    for key in ("wrist", "right_tactile_camera_taxim"):
        assert any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in encoder.key_model_map[key].parameters()
        )
    assert encoder.hypernet[-1].weight.grad is not None
    assert encoder.hypernet[-1].weight.grad.abs().sum() > 0
    if use_fusion:
        assert encoder.fusion_gate.grad is not None


@pytest.mark.parametrize("method", ["fusion", "no_fusion"])
def test_power_plug_hyper_resnet_profile_and_launcher(method: str) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(
        REPO_ROOT / f"configs/manifeel/power_plug_hyper_resnet_{method}.yaml"
    )
    assert cfg.task == "vistac_wrist"
    assert cfg.dataset_path.endswith("plug_quan_Aug02")
    assert cfg.isaacgym_cfg_name == "isaacgym_config_power_plug.yaml"
    assert cfg.use_fusion == (method == "fusion")
    assert cfg.num_epochs == 400
    assert cfg.checkpoint_every == 50
    assert cfg.batch_size == cfg.val_batch_size == 8
    assert "batch8_ep400" in cfg.run_name

    script = (
        REPO_ROOT / f"slurm/manifeel/train_power_plug_hyper_resnet_{method}.sbatch"
    ).read_text()
    assert "--gres=gpu:1" in script
    for option in ("--nodelist", "--exclude", "--constraint"):
        assert option not in script
    result = subprocess.run(
        [
            "/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python",
            str(REPO_ROOT / "scripts/manifeel/train_power_plug_hyper_resnet.py"),
            method,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "HyperResNetObsEncoder" in result.stdout
    assert f"+policy.obs_encoder.use_fusion={str(method == 'fusion').lower()}" in result.stdout
    assert "MANIFEEL_CHECKPOINT_EVERY=50" in result.stdout
    assert "MANIFEEL_NUM_EPOCHS=400" in result.stdout
