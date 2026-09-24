"""Regression tests for the Piper-compatible ManiFeel film_fusion path."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeClip(nn.Module):
    """Cheap frozen-backbone stand-in producing CLIP-width features."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        base = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return base.expand(-1, 1024)


class FakeSparsh(nn.Module):
    """Cheap frozen-backbone stand-in producing Sparsh-width features."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        base = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return base.expand(-1, 768)


def test_film_fusion_encoder_matches_piper_shapes_and_freezing() -> None:
    module = pytest.importorskip("adavip.manifeel.film_fusion_obs_encoder")
    encoder = module.FilmFusionObsEncoder(FakeClip(), FakeSparsh())
    encoder.train()
    assert not encoder.clip_encoder.training
    assert not encoder.sparsh_encoder.training
    assert encoder.output_shape() == (519,)
    assert encoder.hypernet.output_dim == 3584
    assert encoder.film.alpha.item() == pytest.approx(0.01)

    observations = {
        "wrist": torch.rand(3, 3, 16, 16),
        "right_tactile_camera_taxim_pair": torch.rand(3, 6, 12, 10),
        "state": torch.rand(3, 7),
    }
    features = encoder.fusion_features(observations)
    assert features["hypernet_output"].shape == (3, 3584)
    assert features["vision_tokens"].shape == (3, 1, 256)
    assert features["tactile_tokens"].shape == (3, 1, 256)
    assert features["attention_output"].shape == (3, 1, 256)
    assert features["fusion_output"].shape == (3, 512)
    assert features["condition"].shape == (3, 519)


def test_residual_film_uses_positive_piper_formula() -> None:
    module = pytest.importorskip("adavip.manifeel.film_fusion_obs_encoder")
    film = module.LearnableResidualFiLM(alpha_init=0.01)
    values = torch.full((2, 1, 4), 2.0)
    gamma = torch.full((2, 4), 3.0)
    beta = torch.full((2, 4), 5.0)
    expected = values + 0.01 * (gamma.unsqueeze(1) * values + beta.unsqueeze(1))
    torch.testing.assert_close(film(values, gamma, beta), expected)
    assert film.alpha.item() > 0


def test_runner_adapter_pairs_current_then_delta_frame() -> None:
    module = pytest.importorskip("adavip.manifeel.film_fusion_runner")

    class DummyEnv:
        pass

    adapter = module.FilmFusionObservationAdapter(
        DummyEnv(),
        n_obs_steps=2,
        tactile_frame_delta=5,
    )
    tactile = np.arange(7, dtype=np.float32).reshape(1, 7, 1, 1, 1)
    tactile = np.repeat(tactile, 3, axis=2)
    observations = {
        "wrist": np.zeros((1, 7, 3, 2, 2), dtype=np.float32),
        "right_tactile_camera_taxim": tactile,
        "state": np.zeros((1, 7, 7), dtype=np.float32),
    }
    result = adapter._adapt(observations)
    pair = result["right_tactile_camera_taxim_pair"]
    assert pair.shape == (1, 2, 6, 1, 1)
    np.testing.assert_array_equal(pair[0, :, 0, 0, 0], [5, 6])
    np.testing.assert_array_equal(pair[0, :, 3, 0, 0], [0, 1])
    assert result["wrist"].shape[1] == 2
    assert "right_tactile_camera_taxim" not in result


def test_dataset_delta_pair_respects_episode_boundaries(tmp_path: Path) -> None:
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("cv2")
    pytest.importorskip("manifeel.dataset.manifeel_dataset_equidp")
    module = pytest.importorskip("adavip.manifeel.film_fusion_dataset")

    root = zarr.open(str(tmp_path / "tiny.zarr"), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    frames = np.arange(8, dtype=np.float32).reshape(8, 1, 1, 1)
    tactile = np.broadcast_to(frames, (8, 2, 3, 3)).copy()
    data.array("wrist", np.zeros((8, 2, 2, 3), dtype=np.float32))
    data.array("right_tactile_camera_taxim", tactile)
    data.array("state", np.zeros((8, 7), dtype=np.float32))
    data.array("action", np.zeros((8, 6), dtype=np.float32))
    meta.array("episode_ends", np.array([4, 8], dtype=np.int64))

    shape_meta = {
        "obs": {
            "wrist": {"shape": [3, 2, 2], "type": "rgb"},
            "right_tactile_camera_taxim_pair": {
                "shape": [6, 2, 3],
                "type": "rgb",
            },
            "state": {"shape": [7], "type": "low_dim"},
        },
        "action": {"shape": [6]},
    }
    dataset = module.FilmFusionDataset(
        shape_meta=shape_meta,
        zarr_path=str(tmp_path / "tiny.zarr"),
        horizon=3,
        pad_before=1,
        pad_after=0,
        n_obs_steps=2,
        tactile_frame_delta=5,
        val_ratio=0.0,
    )

    second_episode_start = next(
        index
        for index in range(len(dataset))
        if dataset._observation_indices(index).tolist() == [4, 4]
    )
    pair = dataset[second_episode_start]["obs"]["right_tactile_camera_taxim_pair"]
    assert pair.shape == (2, 6, 2, 3)
    torch.testing.assert_close(pair[:, 0], torch.full((2, 2, 3), 4.0))
    torch.testing.assert_close(pair[:, 3], torch.full((2, 2, 3), 4.0))

    normalizer = dataset.get_normalizer()
    image = torch.tensor([0.0, 0.5, 1.0])
    torch.testing.assert_close(normalizer["wrist"].normalize(image), image)
    torch.testing.assert_close(
        normalizer["right_tactile_camera_taxim_pair"].normalize(image),
        image,
    )


def test_film_fusion_config_has_only_active_piper_path() -> None:
    omega = pytest.importorskip("omegaconf")
    omega.OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = omega.OmegaConf.load(
        REPO_ROOT / "configs/manifeel/power_plug_adavip_film_fusion.yaml"
    )
    omega.OmegaConf.resolve(cfg)
    assert cfg._target_.endswith("RetainedDiffusionUnetImageWorkspace")
    assert cfg.policy.obs_encoder._target_.endswith("FilmFusionObsEncoder")
    assert cfg.task.dataset._target_.endswith("FilmFusionDataset")
    assert cfg.task.env_runner._target_.endswith("FilmFusionRunner")
    assert set(cfg.shape_meta.obs) == {
        "wrist",
        "right_tactile_camera_taxim_pair",
        "state",
    }
    assert cfg.shape_meta.obs.right_tactile_camera_taxim_pair.shape == [6, 320, 240]
    assert cfg.tactile_frame_delta == 5
    assert cfg.dataloader.batch_size == 8
    assert cfg.val_dataloader.batch_size == 8
    assert cfg.training.num_epochs == 400
    assert cfg.training.checkpoint_every == 50
    assert "batch8" in cfg.exp_name
    assert "ep400" in cfg.exp_name
    assert "RN50.pt" in cfg.policy.obs_encoder.clip_encoder.checkpoint_path
    assert "dino_vitbase.ckpt" in cfg.policy.obs_encoder.sparsh_encoder.checkpoint_path
    rendered = omega.OmegaConf.to_yaml(cfg)
    assert "MultiImageObsEncoder" not in rendered
    assert "resnet" not in rendered.lower()
    assert "task_embedding" not in rendered


def test_film_fusion_config_allows_external_asset_overrides(monkeypatch) -> None:
    omega = pytest.importorskip("omegaconf")
    omega.OmegaConf.register_new_resolver("eval", eval, replace=True)
    monkeypatch.setenv("ADAVIP_FILM_FUSION_CLIP_CHECKPOINT", "/models/RN50.pt")
    monkeypatch.setenv("ADAVIP_FILM_FUSION_CLIP_SOURCE_ROOT", "/src/CLIP")
    monkeypatch.setenv("ADAVIP_FILM_FUSION_SPARSH_CHECKPOINT", "/models/sparsh.ckpt")
    monkeypatch.setenv("ADAVIP_FILM_FUSION_SPARSH_SOURCE_ROOT", "/src/sparsh")
    cfg = omega.OmegaConf.load(
        REPO_ROOT / "configs/manifeel/power_plug_adavip_film_fusion.yaml"
    )
    omega.OmegaConf.resolve(cfg)
    assert cfg.policy.obs_encoder.clip_encoder.checkpoint_path == "/models/RN50.pt"
    assert cfg.policy.obs_encoder.clip_encoder.source_root == "/src/CLIP"
    assert cfg.policy.obs_encoder.sparsh_encoder.checkpoint_path == "/models/sparsh.ckpt"
    assert cfg.policy.obs_encoder.sparsh_encoder.source_root == "/src/sparsh"


def test_film_fusion_slurm_script_avoids_node_selection() -> None:
    script = (
        REPO_ROOT / "slurm/manifeel/train_power_plug_adavip_film_fusion.sbatch"
    ).read_text()
    assert "--gres=gpu:1" in script
    for option in ("--nodelist", "--exclude", "--constraint"):
        assert option not in script


def test_film_fusion_launcher_dry_run() -> None:
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/manifeel/train_power_plug_adavip_film_fusion.sh"),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "power_plug_adavip_film_fusion" in result.stdout
    assert "batch8" in result.stdout
    assert "batch_size=512" not in result.stdout


def test_film_fusion_asset_downloader_is_direct_and_unpacks_sources() -> None:
    script = (
        REPO_ROOT
        / "scripts/manifeel/download_power_plug_adavip_film_fusion_assets.sh"
    ).read_text()
    assert "-u http_proxy" in script
    assert "pretrained/film_fusion/source/CLIP_source.tar.gz" in script
    assert "pretrained/film_fusion/source/sparsh_source.tar.gz" in script
    assert "tar -xzf" in script
    assert "clip/model.py" in script
    assert "tactile_ssl/model/__init__.py" in script
