#!/usr/bin/env python3
"""Summarize executed action traces separately for successful and failed seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_run(directory: Path) -> dict[str, object]:
    """Compute per-environment action diagnostics without assuming stage labels."""
    with np.load(directory / "rollout_trace.npz") as trace:
        actions = trace["executed_actions"]
        success = trace["success_by_step"]
        success = success.reshape(success.shape[0], success.shape[1], -1).any(axis=-1)
        geometry = {key: trace[key] for key in ("plug_pos", "socket_pos") if key in trace}
    if actions.ndim != 3 or success.shape != actions.shape[:2]:
        raise ValueError("action and success traces must have matching [step, environment] axes")
    if actions.shape[0] < 2 or not np.isfinite(actions).all():
        raise ValueError("expected at least two finite executed action steps")
    labels = success.any(axis=0)
    translation = np.linalg.norm(actions[..., :3], axis=-1)
    rotation = np.linalg.norm(actions[..., 3:6], axis=-1)
    delta = np.linalg.norm(np.diff(actions, axis=0), axis=-1)
    output: dict[str, object] = {"steps": actions.shape[0], "environments": actions.shape[1]}
    for name, mask in (("success", labels), ("failure", ~labels)):
        group: dict[str, object] = {"count": int(mask.sum()),
                                    "environment_indices": np.flatnonzero(mask).tolist()}
        if mask.any():
            group.update({
                "translation_norm_mean": float(translation[:, mask].mean()),
                "rotation_norm_mean": float(rotation[:, mask].mean()),
                "action_step_change_mean": float(delta[:, mask].mean()),
                "first_success_step_by_env": [int(np.argmax(success[:, index]))
                                               for index in np.flatnonzero(mask)] if name == "success" else [],
            })
            if actions.shape[-1] == 7:
                group["gripper_action_mean"] = float(actions[:, mask, 6].mean())
            if len(geometry) == 2:
                distance = np.linalg.norm(geometry["plug_pos"] - geometry["socket_pos"], axis=-1)
                group["minimum_plug_socket_distance_mean"] = float(distance[:, mask].min(axis=0).mean())
        output[name] = group
    chunk_file = directory / "policy_chunks.npz"
    if chunk_file.is_file():
        with np.load(chunk_file) as chunks:
            predicted = chunks["action_chunks"]
        if predicted.ndim != 4 or predicted.shape[1] != actions.shape[1]:
            raise ValueError("predicted chunks must be [chunk, environment, horizon, action]")
        output["prediction"] = {"chunks": predicted.shape[0], "horizon": predicted.shape[2],
                                "nonfinite": int((~np.isfinite(predicted)).sum())}
    return output


def main() -> None:
    """Print and persist per-seed failure summaries next to evaluation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize_run(args.run)
    output = args.output or (args.run / "action_failure_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
