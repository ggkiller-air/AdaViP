"""Check the baseline ResNet attention fusion ablation."""

from __future__ import annotations

from pathlib import Path
import subprocess

import torch
from torch import nn

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

from adavip.manifeel.hyper_resnet_obs_encoder import HyperResNetObsEncoder
from adavip.manifeel.resnet_fusion_obs_encoder import ResNetFusionObsEncoder


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


def test_resnet_fusion_matches_baseline_at_init_and_hyper_fusion_without_modulation() -> None:
    baseline = MultiImageObsEncoder(shape_meta=SHAPE_META, rgb_model=_fake_resnet())
    fusion = ResNetFusionObsEncoder(shape_meta=SHAPE_META, rgb_model=_fake_resnet())
    hyper_fusion = HyperResNetObsEncoder(
        shape_meta=SHAPE_META,
        rgb_model=_fake_resnet(),
        use_fusion=True,
    )
    fusion.load_state_dict(baseline.state_dict(), strict=False)
    hyper_fusion.load_state_dict(fusion.state_dict(), strict=False)
    assert not hasattr(fusion, "hypernet")
    assert not hasattr(fusion, "film")
    assert fusion.output_shape() == (1031,)

    observations = {
        "wrist": torch.rand(2, 3, 8, 8),
        "right_tactile_camera_taxim": torch.rand(2, 3, 8, 8),
        "state": torch.rand(2, 7),
    }
    torch.testing.assert_close(fusion(observations), baseline(observations))
    with torch.no_grad():
        fusion.fusion_gate.fill_(0.2)
        hyper_fusion.fusion_gate.fill_(0.2)
    torch.testing.assert_close(fusion(observations), hyper_fusion(observations))

    fusion(observations).square().mean().backward()
    for key in ("wrist", "right_tactile_camera_taxim"):
        assert any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in fusion.key_model_map[key].parameters()
        )
    assert fusion.fusion_gate.grad is not None
    assert fusion.cross_attention.in_proj_weight.grad is not None
    assert fusion.cross_attention.in_proj_weight.grad.abs().sum() > 0


def test_resnet_fusion_profile_and_launcher() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(REPO_ROOT / "configs/manifeel/power_plug_resnet_fusion.yaml")
    assert cfg.task == "vistac_wrist"
    assert cfg.dataset_path.endswith("plug_quan_Aug02")
    assert cfg.isaacgym_cfg_name == "isaacgym_config_power_plug.yaml"
    assert cfg.num_epochs == 400
    assert cfg.checkpoint_every == 50
    assert cfg.batch_size == cfg.val_batch_size == 8
    assert "batch8_ep400" in cfg.run_name

    script = (
        REPO_ROOT / "slurm/manifeel/train_power_plug_resnet_fusion.sbatch"
    ).read_text()
    assert "--gres=gpu:1" in script
    for option in ("--nodelist", "--exclude", "--constraint"):
        assert option not in script
    result = subprocess.run(
        [
            "/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python",
            str(REPO_ROOT / "scripts/manifeel/train_power_plug_resnet_fusion.py"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ResNetFusionObsEncoder" in result.stdout
    assert "MANIFEEL_CHECKPOINT_EVERY=50" in result.stdout
    assert "MANIFEEL_NUM_EPOCHS=400" in result.stdout
