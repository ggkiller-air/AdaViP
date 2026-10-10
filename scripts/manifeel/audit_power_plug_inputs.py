#!/usr/bin/env python3
"""Audit training Zarr images against recorded Power Plug simulator inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

from adavip.manifeel.film_fusion_dataset import FilmFusionDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIG = REPO_ROOT / "configs/manifeel/power_plug_adavip_film_fusion.yaml"


def image_stats(image: np.ndarray) -> dict[str, float | list[int]]:
    """Measure the actual channel-first input passed to the observation encoder."""
    return {
        "shape": list(image.shape),
        "minimum": float(image.min()),
        "maximum": float(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
    }


def save_rgb(image: np.ndarray, path: Path) -> None:
    """Write a channel-first [0, 1] RGB observation as a viewable PNG."""
    pixels = np.moveaxis(np.clip(image, 0.0, 1.0), 0, -1)
    pixels = (pixels * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {path}")


def audit_dataset(dataset_path: Path, output_dir: Path, input_stats_path: Path,
                  sample_count: int = 32) -> dict[str, object]:
    """Collect deterministic paired training observations and simulator ranges."""
    config = OmegaConf.load(TRAIN_CONFIG)
    dataset = FilmFusionDataset(
        shape_meta=OmegaConf.to_container(config.shape_meta, resolve=True),
        zarr_path=str(dataset_path),
        horizon=config.horizon,
        pad_before=config.n_obs_steps - 1,
        pad_after=config.n_action_steps - 1,
        n_obs_steps=config.n_obs_steps,
        tactile_frame_delta=config.tactile_frame_delta,
        seed=config.training.seed,
        val_ratio=config.task.dataset.val_ratio,
        max_train_episodes=config.task.dataset.max_train_episodes,
    )
    sample_indices = np.unique(np.linspace(0, len(dataset) - 1,
        num=min(sample_count, len(dataset)), dtype=np.int64))
    summaries: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=False)
    for sample_index in sample_indices:
        observation = dataset[int(sample_index)]["obs"]
        wrist = observation["wrist"].numpy()
        pair = observation["right_tactile_camera_taxim_pair"].numpy()
        data: dict[str, object] = {
            "sample_index": int(sample_index),
            "wrist": image_stats(wrist),
            "tactile_pair": image_stats(pair),
            "mean_absolute_frame_difference": float(np.abs(pair[:, :3] - pair[:, 3:]).mean()),
        }
        summaries.append(data)
        if sample_index == sample_indices[0]:
            save_rgb(wrist[-1], output_dir / "training_wrist.png")
            save_rgb(pair[-1, :3], output_dir / "training_tactile_current.png")
            save_rgb(pair[-1, 3:], output_dir / "training_tactile_past.png")

    simulator = json.loads(input_stats_path.read_text())
    simulator_wrist = [entry["wrist"] for entry in simulator]
    simulator_pairs = [entry["right_tactile_camera_taxim_pair"] for entry in simulator]
    normalizer = dataset.get_normalizer()
    result: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "dataset_episodes": dataset.replay_buffer.n_episodes,
        "training_sequence_count": len(dataset),
        "sample_count": len(summaries),
        "tactile_frame_delta": config.tactile_frame_delta,
        "training_wrist_mean": float(np.mean([entry["wrist"]["mean"] for entry in summaries])),
        "simulator_wrist_mean": float(np.mean([entry["mean"] for entry in simulator_wrist])),
        "training_tactile_mean": float(np.mean([entry["tactile_pair"]["mean"] for entry in summaries])),
        "simulator_tactile_mean": float(np.mean([entry["mean"] for entry in simulator_pairs])),
        "training_pair_frame_difference_mean": float(np.mean([
            entry["mean_absolute_frame_difference"] for entry in summaries])),
        "simulator_pair_frame_difference_mean": float(np.mean([
            entry["mean_absolute_frame_difference"] for entry in simulator_pairs])),
        "training_wrist_range": [min(entry["wrist"]["minimum"] for entry in summaries),
                                 max(entry["wrist"]["maximum"] for entry in summaries)],
        "training_tactile_range": [min(entry["tactile_pair"]["minimum"] for entry in summaries),
                                   max(entry["tactile_pair"]["maximum"] for entry in summaries)],
        "training_image_normalizer": {
            key: {"scale": normalizer[key].params_dict["scale"].tolist(),
                  "offset": normalizer[key].params_dict["offset"].tolist()}
            for key in ("wrist", "right_tactile_camera_taxim_pair")
        },
        "samples": summaries,
    }
    (output_dir / "input_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Audit observed training samples with bounded, reproducible sampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sim-input-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=32)
    args = parser.parse_args()
    if args.sample_count < 1:
        parser.error("sample-count must be positive")
    result = audit_dataset(args.dataset, args.output_dir, args.sim_input_stats,
                           sample_count=args.sample_count)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
