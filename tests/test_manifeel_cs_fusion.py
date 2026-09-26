"""Check the CS fusion ablation and its two single-task training profiles."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeClip(nn.Module):
    """Produce a deterministic CLIP-width feature vector."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, 1024)


class FakeSparsh(nn.Module):
    """Produce a deterministic Sparsh-width feature vector."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, 768)


def test_cs_fusion_matches_zero_film_path_and_keeps_backbones_frozen() -> None:
    from adavip.manifeel.cs_fusion_obs_encoder import CSFusionObsEncoder
    from adavip.manifeel.film_fusion_obs_encoder import FilmFusionObsEncoder

    cs = CSFusionObsEncoder(FakeClip(), FakeSparsh())
    film = FilmFusionObsEncoder(FakeClip(), FakeSparsh())
    film.load_state_dict(cs.state_dict(), strict=False)
    with torch.no_grad():
        film.hypernet.network[-1].weight.zero_()
        film.hypernet.network[-1].bias.zero_()
    cs.train()
    assert not cs.clip_encoder.training
    assert not cs.sparsh_encoder.training
    assert not any(param.requires_grad for param in cs.clip_encoder.parameters())
    assert not any(param.requires_grad for param in cs.sparsh_encoder.parameters())
    assert not hasattr(cs, "hypernet")
    assert not hasattr(cs, "film")

    obs = {
        "wrist": torch.rand(2, 3, 16, 16),
        "right_tactile_camera_taxim_pair": torch.rand(2, 6, 12, 10),
        "state": torch.rand(2, 7),
    }
    torch.testing.assert_close(cs(obs), film(obs))
    assert cs.output_shape() == (519,)


@pytest.mark.parametrize(
    ("task", "vision_key", "action_dim", "dataset_name", "isaacgym_cfg"),
    [
        ("power_plug", "wrist", 6, "plug_quan_Aug02", "isaacgym_config_power_plug.yaml"),
        ("ball_sorting", "front", 7, "sorting_quan_Aug8", "isaacgym_config_ball_sorting.yaml"),
    ],
)
def test_cs_fusion_config_and_launcher(
    task: str,
    vision_key: str,
    action_dim: int,
    dataset_name: str,
    isaacgym_cfg: str,
) -> None:
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.load(REPO_ROOT / f"configs/manifeel/{task}_cs_fusion.yaml")
    OmegaConf.resolve(cfg)
    assert cfg.policy.obs_encoder._target_.endswith("CSFusionObsEncoder")
    assert set(cfg.shape_meta.obs) == {vision_key, "right_tactile_camera_taxim_pair", "state"}
    assert cfg.shape_meta.action.shape == [action_dim]
    if task == "ball_sorting":
        assert cfg.policy.obs_encoder.vision_key == vision_key
    else:
        assert "vision_key" not in cfg.policy.obs_encoder
    assert cfg.dataset_path.endswith(dataset_name)
    assert cfg.isaacgym_cfg_name == isaacgym_cfg
    assert cfg.training.num_epochs == 400
    assert cfg.training.checkpoint_every == 50
    assert cfg.dataloader.batch_size == cfg.val_dataloader.batch_size == 8
    assert cfg._target_.endswith("RetainedDiffusionUnetImageWorkspace")

    script = (REPO_ROOT / f"slurm/manifeel/train_cs_fusion_{task}.sbatch").read_text()
    assert "--gres=gpu:1" in script
    for option in ("--nodelist", "--exclude", "--constraint"):
        assert option not in script
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/manifeel/train_cs_fusion.sh"), task, "--dry-run"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"--config-name={task}_cs_fusion" in result.stdout
    assert "batch8_ep400" in result.stdout
