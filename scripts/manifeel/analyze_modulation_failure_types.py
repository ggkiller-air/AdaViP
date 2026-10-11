#!/usr/bin/env python3
"""Group matched normal/zero-modulation outcomes by descriptive failure geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from scripts.manifeel.analyze_power_plug_failure_geometry import geometry_summary


FAILURE_TYPES = ("no_contact_failure", "contacted_far_failure",
                 "near_then_departed_failure", "near_goal_unfinished_failure")
CATEGORIES = (*FAILURE_TYPES, "success")
DISPLAY_NAMES = ("No target contact", "Contacted, never near",
                 "Near, then departed", "Near, unfinished")


def refined_geometry(rollout: dict[str, np.ndarray], success_threshold: float,
                     near_multiplier: float = 2.0) -> dict[str, object]:
    """Split near failures by final geometry, never inferring grasp or drop events."""
    if not np.isfinite(success_threshold) or not np.isfinite(near_multiplier):
        raise ValueError("geometric thresholds must be finite")
    if success_threshold <= 0 or near_multiplier <= 1:
        raise ValueError("positive success threshold and near multiplier above one required")
    near_threshold = success_threshold * near_multiplier
    summary = geometry_summary(rollout, near_threshold)
    categories = np.asarray(summary["categories"], dtype="U40")
    final = np.asarray(summary["last_pre_success_keypoint_error_m"])
    near_failure = categories == "contacted_near_goal_failure"
    categories[near_failure & (final >= near_threshold)] = "near_then_departed_failure"
    categories[near_failure & (final < near_threshold)] = "near_goal_unfinished_failure"
    summary["categories"] = categories.tolist()
    summary["counts"] = {category: int((categories == category).sum()) for category in CATEGORIES}
    summary["success_threshold_m"] = success_threshold
    summary["near_threshold_m"] = near_threshold
    summary["minimum_error_in_success_threshold_units"] = (
        np.asarray(summary["minimum_pre_success_keypoint_error_m"]) / success_threshold).tolist()
    return summary


def paired_groups(normal: dict[str, object], zero: dict[str, object]) -> dict[str, object]:
    """Describe the failed counterfactual for helpful and harmful normal modulation."""
    normal_success = np.asarray(normal["success"], dtype=bool)
    zero_success = np.asarray(zero["success"], dtype=bool)
    normal_categories = np.asarray(normal["categories"])
    zero_categories = np.asarray(zero["categories"])
    if normal_success.shape != zero_success.shape or normal_success.ndim != 1:
        raise ValueError("paired one-dimensional outcomes required")
    if normal_categories.shape != normal_success.shape or zero_categories.shape != normal_success.shape:
        raise ValueError("failure categories must match environment counts")
    if not np.isin(normal_categories, CATEGORIES).all() or not np.isin(zero_categories, CATEGORIES).all():
        raise ValueError("unknown failure category")
    if not np.array_equal(normal_categories == "success", normal_success):
        raise ValueError("normal categories disagree with success")
    if not np.array_equal(zero_categories == "success", zero_success):
        raise ValueError("zero categories disagree with success")
    definitions = {
        "normal_only_success": (normal_success & ~zero_success, zero_categories, "zero"),
        "zero_only_success": (~normal_success & zero_success, normal_categories, "normal"),
    }
    groups = {}
    for label, (selected, failed_categories, failed_condition) in definitions.items():
        groups[label] = {"count": int(selected.sum()), "failed_condition": failed_condition,
                         "failure_type_counts": {category: int((selected & (failed_categories == category)).sum())
                                                 for category in FAILURE_TYPES},
                         "environment_indices": np.flatnonzero(selected).tolist()}
    matrix = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    for normal_category, zero_category in zip(normal_categories, zero_categories):
        matrix[CATEGORIES.index(normal_category), CATEGORIES.index(zero_category)] += 1
    return {"groups": groups, "transition_category_order": list(CATEGORIES),
            "normal_rows_zero_columns": matrix.tolist(),
            "both_success": int((normal_success & zero_success).sum()),
            "both_failure": int((~normal_success & ~zero_success).sum())}


def analyze_pair(normal_directory: Path, zero_directory: Path, threshold: float,
                 multiplier: float) -> dict[str, object]:
    """Verify recorded protocol and paired starts before categorizing outcomes."""
    required = ("reward_by_step", "keypoints_plug", "keypoints_socket", "plug_socket_force",
                "initial_plug_socket_force", "initial_state", "initial_plug_pos",
                "initial_socket_pos", "initial_plug_quat", "initial_socket_quat")
    rollouts = {}
    for label, directory in (("normal", normal_directory), ("zero", zero_directory)):
        with np.load(directory / "rollout_trace.npz", allow_pickle=False) as trace:
            rollouts[label] = {key: trace[key] for key in required}
        reward = rollouts[label]["reward_by_step"]
        if reward.shape != (500, 50):
            raise ValueError(f"unexpected protocol: {directory}")
        log = json.loads((directory / "eval_log.json").read_text())
        maxima = reward.max(axis=0)
        if not np.isclose(maxima.mean(), log["test/mean_score"]):
            raise ValueError("reward and mean_score disagree")
        if not np.array_equal(maxima, np.asarray([log[f"test/sim_max_reward_{index}"] for index in range(50)])):
            raise ValueError("per-environment rewards disagree with logs")
    for key in required[5:]:
        if not np.array_equal(rollouts["normal"][key], rollouts["zero"][key]):
            raise ValueError(f"unmatched initial condition: {key}")
    summaries = {label: refined_geometry(rollout, threshold, multiplier)
                 for label, rollout in rollouts.items()}
    return {"runs": summaries, "paired": paired_groups(summaries["normal"], summaries["zero"])}


def main() -> None:
    """Analyze pre-existing runs on CPU and retain all fixed threshold sensitivities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    specifications = {
        "power_plug_fusion": (Path("results/mechanism_followup_batch8_20261010/controlled_matrix_report/summary.json"),
                              "original", "TacSLTaskPowerInsertion.yaml"),
        "power_plug_no_fusion": (Path("results/no_fusion_zero_batch8_20261011/paired_report/summary.json"),
                                 "normal", "TacSLTaskPowerInsertion.yaml"),
        "peg_insertion_fusion": (Path("results/cross_task_zero_batch8_20261011/peg_insertion_paired_report/summary.json"),
                                 "normal", "TacSLTaskInsertion.yaml"),
        "nut_bolt_fusion": (Path("results/cross_task_zero_batch8_20261011/nut_bolt_assembly_paired_report/summary.json"),
                            "normal", "TacSLTaskBoltNut.yaml"),
    }
    records = {}
    for task, (source, normal_label, task_config) in specifications.items():
        matrix_report = json.loads(source.read_text())
        config_path = Path("third_party/manifeel/manifeel/config/task") / task_config
        config = OmegaConf.load(config_path)
        threshold = float(config.rl.close_error_thresh)
        sensitivities = {}
        sources = {}
        for multiplier in (1.5, 2.0, 3.0, 4.0):
            per_seed = {}
            pooled = {label: {category: 0 for category in FAILURE_TYPES}
                      for label in ("normal_only_success", "zero_only_success")}
            for seed in matrix_report["seeds"]:
                normal_directory = Path(matrix_report["runs"][f"seed{seed}_{normal_label}"]["directory"])
                zero_directory = Path(matrix_report["runs"][f"seed{seed}_zero"]["directory"])
                per_seed[str(seed)] = analyze_pair(normal_directory, zero_directory, threshold, multiplier)
                sources[str(seed)] = {"normal": str(normal_directory), "zero": str(zero_directory)}
                for group, counts in pooled.items():
                    for category in FAILURE_TYPES:
                        counts[category] += per_seed[str(seed)]["paired"]["groups"][group]["failure_type_counts"][category]
            sensitivities[str(multiplier)] = {"by_seed": per_seed, "pooled_failure_types": pooled}
        records[task] = {"source_matrix_report": str(source), "sources": sources,
                         "task_config": str(config_path), "success_threshold_m": threshold,
                         "threshold_sensitivity": sensitivities}
    args.output_dir.mkdir(parents=True)
    report = {"tasks": records, "primary_near_multiplier": 2.0,
              "failure_types": list(FAILURE_TYPES),
              "scope": "Descriptive counterfactual failure geometry after intervention, not pre-treatment causal subgroups. Same training checkpoint within each pair; no evidence yet of stable task-class rescue across training seeds. No grasp/joint/finger-contact data to infer missed grasps, drops or insertion depth."}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    titles = ("Power Plug + fusion", "Power Plug, no fusion", "Peg Insertion + fusion", "Nut-Bolt + fusion")
    for axis, title, item in zip(axes.flat, titles, records.values()):
        primary = item["threshold_sensitivity"]["2.0"]
        for group_index, group in enumerate(("normal_only_success", "zero_only_success")):
            counts = primary["pooled_failure_types"][group]
            positions = np.arange(len(FAILURE_TYPES)) + (group_index - 0.5) * 0.36
            bars = axis.bar(positions, [counts[category] for category in FAILURE_TYPES], width=0.36,
                            label="Normal succeeds / zero fails" if group_index == 0 else "Zero succeeds / normal fails")
            axis.bar_label(bars, padding=3)
        axis.set_xticks(np.arange(len(FAILURE_TYPES)), DISPLAY_NAMES, fontsize=9)
        axis.set(title=title, ylabel="Matched environments", ylim=(0, 16))
        axis.legend(fontsize=8, loc="upper right")
    figure.suptitle("Failure types among outcome swaps, not causes\n100 paired starts per saved model; near = 2x task success error; contact > 0.001")
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"failure_type_swaps.{extension}", dpi=170)
    plt.close(figure)
    print(json.dumps({task: item["threshold_sensitivity"]["2.0"]["pooled_failure_types"]
                      for task, item in records.items()}, indent=2))


if __name__ == "__main__":
    main()
