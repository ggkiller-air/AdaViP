"""Inspect all nine sources and representative full-sensor windows without a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from adavip.manifeel.full_sensor_workspace import manifest_fingerprint, normalizer_fingerprint


def main() -> None:
    """Emit reproducible metadata and verify at least one real window per task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/manifeel/train_full_sensor_reference.yaml",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    dataset = hydra.utils.instantiate(cfg.dataset)
    validation = dataset.get_validation_dataset()
    result = {
        "manifest": dataset.manifest,
        "manifest_sha256": manifest_fingerprint(dataset.manifest),
        "normalizer_sha256": normalizer_fingerprint(dataset.get_normalizer()),
        "train_samples": len(dataset), "validation_samples": len(validation), "tasks": {},
    }
    offset = 0
    for task_index, spec in enumerate(dataset.task_specs):
        samples = [dataset[offset], dataset[offset + dataset.task_lengths[task_index] - 1]]
        result["tasks"][spec.task_id] = {
            "train_frames": dataset.task_lengths[task_index],
            "validation_frames": validation.task_lengths[task_index],
            "train_episodes": len(dataset.train_episodes[task_index]),
            "validation_episodes": len(dataset.val_episodes[task_index]),
            "action_dim": spec.action_dim,
            "obs_shapes": {key: list(value.shape) for key, value in samples[0]["obs"].items()},
            "boundary_valid_action_counts": [int(sample["action_loss_mask"].sum()) for sample in samples],
        }
        offset += dataset.task_lengths[task_index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
