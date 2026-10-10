#!/usr/bin/env python3
"""Compare recorded Power Plug rollouts without loading GPU checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_run(directory: Path) -> dict[str, object]:
    """Compute action, geometry, input, and feature statistics for one run."""
    evaluation_file = directory / "eval_log.json"
    evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.is_file() else {}
    inputs = json.loads((directory / "input_stats.json").read_text())
    features = json.loads((directory / "feature_stats.json").read_text())
    with np.load(directory / "rollout_trace.npz") as trace:
        executed_actions = trace["executed_actions"]
        initial_state = trace["initial_state"].tolist()
        success = trace["success_by_step"].reshape(executed_actions.shape[0],
            executed_actions.shape[1], -1).any(axis=-1)
        per_env_success = success.any(axis=0)
        legacy_window = 3 if directory.name.startswith(("baseline_", "corrected_baseline_")) else 8
        result: dict[str, object] = {
            "episode_success_rate": float(per_env_success.mean()),
            "success_by_env": per_env_success.tolist(),
            "reported_success_rate": evaluation.get("test/success_rate"),
            "old_last_window_success_rate": float(success[-legacy_window:].any(axis=0).mean()),
            "quaternion_sign_corrections": evaluation.get("diagnostics/policy", {}).get(
                "quaternion_sign_corrections"),
            "first_success_step_by_env": [
                int(np.argmax(success[:, index])) if success[:, index].any() else None
                for index in range(success.shape[1])
            ],
            "initial_ee_pose_by_env": initial_state,
            "executed_steps": int(executed_actions.shape[0]),
            "mean_absolute_action_by_dim": np.abs(executed_actions).mean(axis=(0, 1)).tolist(),
            "max_absolute_action_by_dim": np.abs(executed_actions).max(axis=(0, 1)).tolist(),
            "mean_translation_action_norm": float(np.linalg.norm(executed_actions[..., :3], axis=-1).mean()),
            "mean_rotation_action_norm": float(np.linalg.norm(executed_actions[..., 3:], axis=-1).mean()),
            "action_nonfinite_count": int((~np.isfinite(executed_actions)).sum()),
            "feature_norm_mean": {
                key: float(np.mean([entry["norm"] for entry in values]))
                for key, values in features.items() if values
            },
            "feature_norm_time_std": {
                key: float(np.std([entry["norm"] for entry in values]))
                for key, values in features.items() if values
            },
            "feature_nonfinite_count": {
                key: int(sum(entry["nonfinite"] for entry in values))
                for key, values in features.items() if values
            },
        }
        if "socket_pos" in trace and "plug_pos" in trace:
            distances = np.linalg.norm(trace["socket_pos"] - trace["plug_pos"], axis=-1)
            result["initial_socket_pos_by_env"] = trace["socket_pos"][0].tolist()
            result["initial_center_distance_by_env"] = distances[0].tolist()
            result["minimum_center_distance_by_env"] = distances.min(axis=0).tolist()
            result["final_center_distance_by_env"] = distances[-1].tolist()
            result["ever_within_10mm_center_count"] = int((distances.min(axis=0) < 0.01).sum())
            orientation = trace["ee_pose"][..., 3:7]
            orientation_dot = np.sum(orientation[1:] * orientation[:-1], axis=-1)
            result["quaternion_equivalent_sign_flips"] = int((orientation_dot < -0.99).sum())

    tactile_pairs = [entry["right_tactile_camera_taxim_pair"]
        for entry in inputs if "right_tactile_camera_taxim_pair" in entry]
    if tactile_pairs:
        result["tactile_pair_difference_first"] = tactile_pairs[0]["mean_absolute_frame_difference"]
        result["tactile_pair_difference_mean"] = float(np.mean([
            entry["mean_absolute_frame_difference"] for entry in tactile_pairs]))
        result["tactile_pair_value_min"] = min(entry["min"] for entry in tactile_pairs)
        result["tactile_pair_value_max"] = max(entry["max"] for entry in tactile_pairs)
    return result


def main() -> None:
    """Write a compact JSON comparison of all diagnostic runs under a root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        directory.name: summarize_run(directory)
        for directory in sorted(args.root.glob("*_epoch*_n*_s*"))
        if (directory / "rollout_trace.npz").is_file()
    }
    if not runs:
        raise FileNotFoundError(f"no completed diagnostic evaluations under {args.root}")
    baseline = next((run for name, run in runs.items() if name.startswith("baseline_none_")), None)
    if baseline is not None:
        for run in runs.values():
            run["max_initial_ee_pose_difference_from_baseline"] = float(np.max(np.abs(
                np.asarray(run["initial_ee_pose_by_env"])
                - np.asarray(baseline["initial_ee_pose_by_env"]))))
            if "initial_socket_pos_by_env" in run:
                run["max_initial_socket_pos_difference_from_baseline"] = float(np.max(np.abs(
                    np.asarray(run["initial_socket_pos_by_env"])
                    - np.asarray(baseline["initial_socket_pos_by_env"]))))
    output = args.root / "diagnostic_summary.json"
    output.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")
    for name, run in runs.items():
        distances = run.get("minimum_center_distance_by_env", [float("nan")])
        print(name, "episode SR", run["episode_success_rate"], "old/reported",
              run["old_last_window_success_rate"], run["reported_success_rate"], "initial-delta",
              run.get("max_initial_ee_pose_difference_from_baseline"),
              "min-distance", min(distances))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
