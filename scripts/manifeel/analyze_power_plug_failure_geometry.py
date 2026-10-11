#!/usr/bin/env python3
"""Compare matched Power Plug failures with geometric, not attention-based, diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def geometry_summary(rollout: dict[str, np.ndarray], near_threshold: float) -> dict:
    """Classify failures descriptively and stop geometric measurements at first success."""
    rewards = rollout["reward_by_step"]
    if not np.isin(rewards, [0, 1]).all() or near_threshold <= 0:
        raise ValueError("binary benchmark rewards and a positive geometric threshold are required")
    errors = np.linalg.norm(rollout["keypoints_socket"] - rollout["keypoints_plug"], axis=-1).mean(axis=-1)
    if errors.shape != rewards.shape or not np.isfinite(errors).all():
        raise ValueError("keypoint geometry must match reward axes and be finite")
    succeeded = (rewards > 0.5).any(axis=0)
    contact = (np.linalg.norm(rollout["plug_socket_force"], axis=-1) > 0.001).any(axis=0)
    contact |= np.linalg.norm(rollout["initial_plug_socket_force"], axis=-1) > 0.001
    first_success = np.where(succeeded, (rewards > 0.5).argmax(axis=0), -1)
    minima, final = [], []
    for environment, success in enumerate(first_success):
        stop = int(success) + 1 if success >= 0 else len(rewards)
        minima.append(float(errors[:stop, environment].min()))
        final.append(float(errors[stop - 1, environment]))
    near = np.asarray(minima) < near_threshold
    categories = np.full(rewards.shape[1], "contacted_far_failure", dtype="U32")
    categories[contact & near] = "contacted_near_goal_failure"
    categories[~contact] = "no_contact_failure"
    categories[succeeded] = "success"
    return {"success": succeeded.tolist(), "first_success_reward_index": first_success.tolist(),
            "minimum_pre_success_keypoint_error_m": minima, "last_pre_success_keypoint_error_m": final,
            "categories": categories.tolist(), "counts": {label: int((categories == label).sum()) for label in (
                "success", "no_contact_failure", "contacted_far_failure", "contacted_near_goal_failure")}}


def main() -> None:
    """Reuse existing traces to localize geometric gains without launching simulation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-root", type=Path, default=Path("results/hypernet_batch8_20261010"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--main-run", type=Path)
    parser.add_argument("--success-distance-threshold", type=float, default=0.0069916)
    args = parser.parse_args()
    if (args.baseline_run is None) != (args.main_run is None):
        parser.error("provide both --baseline-run and --main-run")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    names = {"baseline": "gpu3_rollouts/baseline_epoch100_n50_steps500_seed100000",
             "hypernet_fusion": "gpu4_rollouts/hyperresnet_epoch100_n50_steps500_seed100000"}
    if args.baseline_run is not None:
        names = {"baseline": args.baseline_run.resolve(), "hypernet_fusion": args.main_run.resolve()}
    rollouts = {}
    for label, directory in names.items():
        with np.load(args.prior_root / directory / "rollout_trace.npz", allow_pickle=False) as archive:
            rollouts[label] = {key: archive[key] for key in archive.files}
    for key in ("initial_state", "initial_plug_pos", "initial_socket_pos", "initial_plug_quat", "initial_socket_quat"):
        if not np.array_equal(rollouts["baseline"][key], rollouts["hypernet_fusion"][key]):
            raise ValueError(f"unmatched initial condition: {key}")
    near_threshold = 2 * args.success_distance_threshold
    report = {"near_goal_error_threshold_m": near_threshold,
              "sources": {name: str((args.prior_root / directory).resolve()) for name, directory in names.items()},
              "scope": "Descriptive failure geometry on paired saved policies, not causal phase utility or architecture isolation.",
              "runs": {name: geometry_summary(data, near_threshold) for name, data in rollouts.items()}}
    original = np.asarray(report["runs"]["baseline"]["success"])
    changed = np.asarray(report["runs"]["hypernet_fusion"]["success"])
    groups = {"main_only_success": np.flatnonzero(~original & changed),
              "baseline_only_success": np.flatnonzero(original & ~changed)}
    report["paired_changes"] = {name: {"environments": indices.tolist(),
        "baseline_categories": [report["runs"]["baseline"]["categories"][index] for index in indices],
        "main_categories": [report["runs"]["hypernet_fusion"]["categories"][index] for index in indices]}
        for name, indices in groups.items()}
    report["near_threshold_sensitivity"] = {str(multiplier): {name: geometry_summary(data,
        multiplier * args.success_distance_threshold)["counts"] for name, data in rollouts.items()}
        for multiplier in (1.5, 2, 3, 4)}
    args.output_dir.mkdir(parents=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    categories = ["no_contact_failure", "contacted_far_failure", "contacted_near_goal_failure", "success"]
    bottom = np.zeros(2)
    for category in categories:
        heights = [report["runs"][name]["counts"][category] for name in names]
        bars = axes[0].bar(list(names), heights, bottom=bottom, label=category)
        axes[0].bar_label(bars, labels=[str(value) if value else "" for value in heights], label_type="center")
        bottom += heights
    axes[0].set_ylabel("Environments")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12))
    baseline_error = 1000 * np.asarray(report["runs"]["baseline"]["minimum_pre_success_keypoint_error_m"])
    main_error = 1000 * np.asarray(report["runs"]["hypernet_fusion"]["minimum_pre_success_keypoint_error_m"])
    for label, mask in (("both succeed", original & changed), ("main only", ~original & changed),
                        ("baseline only", original & ~changed), ("both fail", ~original & ~changed)):
        axes[1].scatter(baseline_error[mask], main_error[mask], label=label)
    maximum = max(baseline_error.max(), main_error.max())
    axes[1].plot([0, maximum], [0, maximum], color="gray", linestyle="--")
    axes[1].set(xlabel="Baseline minimum keypoint error (mm)", ylabel="Main minimum keypoint error (mm)")
    axes[1].legend(fontsize=8)
    figure.suptitle("Matched saved-policy rollouts: minimum error before first benchmark success")
    figure.savefig(args.output_dir / "failure_geometry.png", dpi=170)
    plt.close(figure)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: item["counts"] for name, item in report["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
