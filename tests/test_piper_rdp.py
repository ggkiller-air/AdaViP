from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RDP_ROOT = REPO_ROOT / "third_party" / "reactive_diffusion_policy"
if str(RDP_ROOT) not in sys.path:
    sys.path.insert(0, str(RDP_ROOT))

pytest.importorskip("torch")
pytest.importorskip("zarr")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

import torch
import zarr
from omegaconf import OmegaConf

from adavip.real_world.gelsight_marker_processor import (
    MarkerDetectorConfig,
    assign_to_reference,
    fit_task_pca,
    order_reference_grid,
    process_episodes,
)
from adavip.real_world.piper_rdp_dataset import PiperRdpDataset, PiperRdpLatentDataset
from adavip.real_world.piper_rdp_runtime import OnlineGelSightPcaProcessor
from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from reactive_diffusion_policy.model.vae.model import VAE


def synthetic_marker_image(displacement_xy: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Render a 7x9 dark marker grid matching the real sensor geometry."""
    image = np.full((240, 320, 3), 120, dtype=np.uint8)
    yy, xx = np.ogrid[:240, :320]
    dx, dy = displacement_xy
    for y in np.linspace(55, 190, 7).astype(int):
        for x in np.linspace(60, 245, 9).astype(int):
            disk = (xx - x - dx) ** 2 + (yy - y - dy) ** 2 <= 5**2
            image[disk] = 10
    return image


def shape_meta(image_size: int = 16) -> dict:
    return {
        "obs": {
            "external_img": {"shape": [3, image_size, image_size], "type": "rgb"},
            "right_wrist_img": {
                "shape": [3, image_size, image_size],
                "type": "rgb",
            },
            "right_robot_qpos": {"shape": [7], "type": "low_dim"},
            "right_gelsight_marker_offset_emb": {
                "shape": [15],
                "type": "low_dim",
            },
        },
        "extended_obs": {
            "right_gelsight_marker_offset_emb": {
                "shape": [15],
                "type": "low_dim",
            }
        },
        "action": {"shape": [7]},
    }


def make_dataset(path: Path, frame_count: int = 36, image_size: int = 16) -> None:
    root = zarr.open(str(path / "replay_buffer.zarr"), mode="w")
    data = root.create_group("data")
    root.create_group("meta").array(
        "episode_ends", np.asarray([frame_count], dtype=np.int64)
    )
    rng = np.random.default_rng(7)
    data.array(
        "external_img",
        rng.integers(
            0, 256, (frame_count, image_size, image_size, 3), dtype=np.uint8
        ),
    )
    data.array(
        "right_wrist_img",
        rng.integers(
            0, 256, (frame_count, image_size, image_size, 3), dtype=np.uint8
        ),
    )
    data.array("right_robot_qpos", rng.normal(size=(frame_count, 7)).astype(np.float32))
    data.array(
        "right_gelsight_marker_offset_emb",
        rng.normal(size=(frame_count, 15)).astype(np.float32),
    )
    data.array("action", rng.normal(size=(frame_count, 7)).astype(np.float32))


def test_marker_tracking_uses_episode_reference_and_normalized_offsets() -> None:
    images = np.stack(
        [
            synthetic_marker_image(),
            synthetic_marker_image((2, 3)),
            synthetic_marker_image(),
            synthetic_marker_image((-4, 1)),
        ]
    )
    initial, offsets, audit = process_episodes(images, [2, 4])
    assert initial.shape == (4, 63, 2)
    assert offsets.shape == (4, 63, 2)
    np.testing.assert_allclose(offsets[[0, 2]], 0.0, atol=1e-7)
    np.testing.assert_allclose(offsets[1, :, 0], 2 / 320, atol=1e-6)
    np.testing.assert_allclose(offsets[1, :, 1], 3 / 240, atol=1e-6)
    np.testing.assert_allclose(offsets[3, :, 0], -4 / 320, atol=1e-6)
    assert all(row["detected_markers"] == 63 for row in audit)
    assert all(row["detection_success"] for row in audit)


def test_online_gelsight_processor_uses_first_frame_and_frozen_pca() -> None:
    transform = np.zeros((126, 15), dtype=np.float32)
    transform[0, 0] = 2.0
    transform[1, 1] = 3.0
    processor = OnlineGelSightPcaProcessor(
        transform_matrix=transform,
        mean=np.zeros(126, dtype=np.float32),
    )
    reference = processor.initialize(synthetic_marker_image())
    np.testing.assert_allclose(reference["marker_offset"], 0.0, atol=1e-7)
    assert reference["marker_offset_emb"].shape == (15,)
    assert reference["reference_initialized"]

    result = processor.process(synthetic_marker_image((2, 3)))
    np.testing.assert_allclose(result["marker_offset"][:, 0], 2 / 320, atol=1e-6)
    np.testing.assert_allclose(result["marker_offset"][:, 1], 3 / 240, atol=1e-6)
    np.testing.assert_allclose(result["marker_offset_emb"][0], 4 / 320, atol=1e-6)
    np.testing.assert_allclose(result["marker_offset_emb"][1], 9 / 240, atol=1e-6)
    assert result["detected_markers"] == 63

    processor.reset()
    with pytest.raises(RuntimeError, match="initialize"):
        processor.process(synthetic_marker_image())


def test_reference_order_and_assignment_preserve_marker_identity() -> None:
    config = MarkerDetectorConfig()
    centers = np.asarray(
        [(x, y) for y in range(7) for x in range(9)], dtype=np.float32
    )
    rng = np.random.default_rng(4)
    ordered = order_reference_grid(centers[rng.permutation(63)], config)
    np.testing.assert_array_equal(ordered, centers)
    moved = centers + np.asarray([0.25, -0.5], dtype=np.float32)
    tracked, distances = assign_to_reference(
        centers, moved[rng.permutation(63)], max_distance_px=2.0
    )
    np.testing.assert_allclose(tracked, moved)
    np.testing.assert_allclose(distances, np.sqrt(0.25**2 + 0.5**2))


def test_task_pca_exports_126_by_15_transform() -> None:
    rng = np.random.default_rng(11)
    latent = rng.normal(size=(80, 5))
    mixing = rng.normal(size=(5, 126))
    offsets = (latent @ mixing).reshape(80, 63, 2).astype(np.float32)
    embedding, pca = fit_task_pca(offsets, n_components=15)
    assert embedding.shape == (80, 15)
    assert pca.components_.T.shape == (126, 15)
    reconstructed = pca.inverse_transform(embedding)
    np.testing.assert_allclose(reconstructed, offsets.reshape(80, 126), atol=2e-5)


def test_dataset_sampling_and_normalizer(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    dataset = PiperRdpDataset(
        shape_meta=shape_meta(),
        dataset_path=str(tmp_path),
        horizon=29,
        pad_before=3,
        pad_after=25,
        n_obs_steps=4,
        obs_temporal_downsample_ratio=2,
    )
    sample = dataset[0]
    assert sample["obs"]["external_img"].shape == (2, 3, 16, 16)
    assert sample["obs"]["right_wrist_img"].shape == (2, 3, 16, 16)
    assert sample["obs"]["right_robot_qpos"].shape == (2, 7)
    assert sample["obs"]["right_gelsight_marker_offset_emb"].shape == (2, 15)
    assert sample["extended_obs"]["right_gelsight_marker_offset_emb"].shape == (
        29,
        15,
    )
    assert sample["action"].shape == (29, 7)
    normalizer = dataset.get_normalizer()
    assert set(normalizer.params_dict.keys()) == {
        "action",
        "external_img",
        "right_wrist_img",
        "right_robot_qpos",
        "right_gelsight_marker_offset_emb",
    }
    normalized = normalizer["action"].normalize(sample["action"])
    assert torch.isfinite(normalized).all()


def make_at(meta: dict, load_dir: Path | None = None) -> VAE:
    return VAE(
        horizon=29,
        shape_meta=meta,
        n_latent_dims=8,
        conv_latent_dims=16,
        conv_layer_num=1,
        use_conv_encoder=True,
        use_rnn_decoder=True,
        rnn_latent_dims=16,
        use_vq=False,
        n_embed=8,
        device="cpu",
        load_dir=str(load_dir) if load_dir is not None else None,
    )


def test_cpu_at_forward_and_checkpoint_restore(tmp_path: Path) -> None:
    meta = shape_meta()
    at = make_at(meta)
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
        np.random.default_rng(2).normal(size=(50, 7)).astype(np.float32)
    )
    normalizer["right_gelsight_marker_offset_emb"] = (
        SingleFieldLinearNormalizer.create_fit(
            np.random.default_rng(3).normal(size=(50, 15)).astype(np.float32)
        )
    )
    at.set_normalizer(normalizer)
    batch = {
        "action": torch.randn(2, 29, 7),
        "extended_obs": {
            "right_gelsight_marker_offset_emb": torch.randn(2, 29, 15)
        },
    }
    result = at.compute_loss_and_metric(batch)
    assert result["loss"].ndim == 0
    assert torch.isfinite(result["loss"])
    assert at.downsampled_input_h == 8

    checkpoint = tmp_path / "at_workspace.ckpt"
    torch.save({"state_dicts": {"model": at.state_dict()}}, checkpoint)
    restored = make_at(copy.deepcopy(meta), checkpoint)
    for key, value in at.state_dict()["encoder"].items():
        torch.testing.assert_close(value, restored.state_dict()["encoder"][key])


def test_latent_dataset_normalizer_contains_eight_dimensional_latent(
    tmp_path: Path,
) -> None:
    make_dataset(tmp_path)
    meta = shape_meta()
    dataset = PiperRdpLatentDataset(
        at=make_at(copy.deepcopy(meta)),
        shape_meta=meta,
        dataset_path=str(tmp_path),
        horizon=29,
        pad_before=3,
        pad_after=25,
        n_obs_steps=4,
        obs_temporal_downsample_ratio=2,
    )
    normalizer = dataset.get_normalizer()
    assert "latent_action" in normalizer.params_dict
    assert normalizer["latent_action"].params_dict["scale"].shape == (8,)


def test_cpu_ldp_forward_and_state_restore() -> None:
    pytest.importorskip("diffusers")
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from reactive_diffusion_policy.model.vision.multi_image_obs_encoder import (
        MultiImageObsEncoder,
    )
    from reactive_diffusion_policy.policy.latent_diffusion_unet_image_policy import (
        LatentDiffusionUnetImagePolicy,
    )

    meta = {
        "obs": {
            "right_robot_qpos": {"shape": [7], "type": "low_dim"},
            "right_gelsight_marker_offset_emb": {
                "shape": [15],
                "type": "low_dim",
            },
        },
        "extended_obs": {
            "right_gelsight_marker_offset_emb": {
                "shape": [15],
                "type": "low_dim",
            }
        },
        "action": {"shape": [7]},
    }
    at = make_at(copy.deepcopy(meta))
    encoder = MultiImageObsEncoder(
        shape_meta=copy.deepcopy(meta),
        rgb_model=torch.nn.Identity(),
        random_transforms=None,
    )
    scheduler = DDIMScheduler(num_train_timesteps=10, prediction_type="epsilon")
    policy = LatentDiffusionUnetImagePolicy(
        at=at,
        use_latent_action_before_vq=False,
        shape_meta=copy.deepcopy(meta),
        noise_scheduler=scheduler,
        obs_encoder=encoder,
        horizon=29,
        n_action_steps=26,
        n_obs_steps=2,
        num_inference_steps=2,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        n_groups=4,
    )
    normalizer = LinearNormalizer()
    rng = np.random.default_rng(9)
    for key, dimension in (
        ("action", 7),
        ("right_robot_qpos", 7),
        ("right_gelsight_marker_offset_emb", 15),
        ("latent_action", 8),
    ):
        normalizer[key] = SingleFieldLinearNormalizer.create_fit(
            rng.normal(size=(80, dimension)).astype(np.float32)
        )
    policy.set_normalizer(normalizer)
    batch = {
        "obs": {
            "right_robot_qpos": torch.randn(2, 2, 7),
            "right_gelsight_marker_offset_emb": torch.randn(2, 2, 15),
        },
        "extended_obs": {
            "right_gelsight_marker_offset_emb": torch.randn(2, 29, 15)
        },
        "action": torch.randn(2, 29, 7),
    }
    loss = policy.compute_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    result = policy.predict_action(
        batch["obs"],
        extended_obs_dict=batch["extended_obs"],
        dataset_obs_temporal_downsample_ratio=2,
    )
    assert result["action_pred"].shape == (2, 29, 7)
    restored = copy.deepcopy(policy)
    restored.load_state_dict(policy.state_dict())
    assert set(restored.state_dict()) == set(policy.state_dict())


def test_launcher_builds_local_at_and_ldp_commands(tmp_path: Path) -> None:
    launcher_path = REPO_ROOT / "scripts" / "real_world" / "train_piper_rdp.py"
    spec = importlib.util.spec_from_file_location("train_piper_rdp", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    accelerate = tmp_path / "accelerate"
    accelerate.touch()
    python = tmp_path / "python"
    at_command = launcher.build_stage_command(
        "at", python, RDP_ROOT, tmp_path, tmp_path / "at", 42, "disabled", None, []
    )
    ldp_command = launcher.build_stage_command(
        "ldp",
        python,
        RDP_ROOT,
        tmp_path,
        tmp_path / "ldp",
        42,
        "disabled",
        tmp_path / "latest.ckpt",
        [],
    )
    assert "task=piper_pick_and_place_at_18_75hz" in at_command
    assert "at=piper_rdp_18_75hz" in at_command
    checkpoint_override = next(
        value for value in ldp_command if value.startswith("at_load_dir=")
    )
    assert checkpoint_override == (
        f"at_load_dir='{(tmp_path / 'latest.ckpt').resolve()}'"
    )
    assert all("env_runner" not in value for value in ldp_command)


def test_launcher_quotes_hydra_checkpoint_paths_containing_equals(
    tmp_path: Path,
) -> None:
    launcher_path = REPO_ROOT / "scripts" / "real_world" / "train_piper_rdp.py"
    spec = importlib.util.spec_from_file_location("train_piper_rdp_equals", launcher_path)
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    checkpoint = tmp_path / "epoch=0530-train_loss=0.023877.ckpt"
    override = launcher.hydra_path_override("at_load_dir", checkpoint)
    assert override == f"at_load_dir='{checkpoint.resolve()}'"


def test_deployment_config_is_self_contained_and_cpu_restorable() -> None:
    export_path = REPO_ROOT / "scripts/real_world/export_piper_rdp_deployment.py"
    spec = importlib.util.spec_from_file_location("export_piper_rdp", export_path)
    assert spec is not None and spec.loader is not None
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    cfg = OmegaConf.create(
        {
            "policy": {
                "at": {"load_dir": "/external/at.ckpt", "device": "cuda:0"},
                "num_inference_steps": 100,
            },
            "task": {
                "dataset": {
                    "_target_": "example.LatentDataset",
                    "at": {"load_dir": "/external/at.ckpt"},
                    "use_latent_action_before_vq": False,
                }
            },
        }
    )
    policy_config = exporter.sanitize_policy_config(cfg, 8)
    assert policy_config["at"]["load_dir"] is None
    assert policy_config["at"]["device"] == "cpu"
    assert policy_config["num_inference_steps"] == 8
    dataset_config = exporter.sanitize_dataset_config(cfg)
    assert dataset_config["_target_"].endswith("PiperRdpDataset")
    assert "at" not in dataset_config
    assert "use_latent_action_before_vq" not in dataset_config
