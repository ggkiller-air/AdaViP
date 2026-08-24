#!/usr/bin/env python3
"""Compare action trajectories from task-embedding evaluation modes."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


EMBEDDING_MODES = ("correct", "other_task", "zero")


def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    """Return deterministic elementwise difference statistics."""

    if left.shape != right.shape:
        return {
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "shape_match": False,
        }
    difference = left.astype(np.float64) - right.astype(np.float64)
    absolute = np.abs(difference)
    return {
        "shape": list(left.shape),
        "shape_match": True,
        "exactly_equal": bool(np.array_equal(left, right)),
        "mean_absolute_difference": float(np.mean(absolute)),
        "root_mean_square_difference": float(np.sqrt(np.mean(difference**2))),
        "max_absolute_difference": float(np.max(absolute)),
    }


def parse_args() -> argparse.Namespace:
    """Parse trajectory root and output path."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=EMBEDDING_MODES,
        default=EMBEDDING_MODES,
    )
    return parser.parse_args()


def main() -> None:
    """Compare correct, other-task, and zero embedding trajectories."""

    args = parse_args()
    archives = {
        mode: np.load(
            args.run_root
            / mode
            / "power_plug_insertion"
            / "action_trajectory.npz",
            allow_pickle=False,
        )
        for mode in args.modes
    }
    result: dict[str, object] = {
        "modes": {
            mode: {
                "embedding_norm": float(np.linalg.norm(archive["task_embedding"])),
                "embedding_source_task_id": str(
                    archive["embedding_source_task_id"].item()
                ),
                "action_chunks_shape": list(archive["action_chunks"].shape),
                "executed_actions_shape": list(archive["executed_actions"].shape),
                "action_predictions_shape": list(archive["action_predictions"].shape),
                "first_action_chunk_env0": archive["action_chunks"][0, 0].tolist(),
                "first_action_prediction_env0": archive[
                    "action_predictions"
                ][0, 0].tolist(),
            }
            for mode, archive in archives.items()
        },
        "pairs": {},
    }
    for left_mode, right_mode in itertools.combinations(args.modes, 2):
        left = archives[left_mode]
        right = archives[right_mode]
        result["pairs"][f"{left_mode}_vs_{right_mode}"] = {
            "first_action_chunk": comparison(
                left["action_chunks"][0], right["action_chunks"][0]
            ),
            "executed_actions": comparison(
                left["executed_actions"], right["executed_actions"]
            ),
            "action_predictions": comparison(
                left["action_predictions"], right["action_predictions"]
            ),
        }

    output = args.output or args.run_root / "action_trajectory_comparison.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
