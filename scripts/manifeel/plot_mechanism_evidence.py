#!/usr/bin/env python3
"""Plot audited Power Plug mechanism evidence without launching inference or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_summary(root: Path, directory: str) -> dict:
    """Read a completed verifier's report, never partial evaluation logs."""
    return json.loads((root / directory / "summary.json").read_text())


def plot_failure_geometry(root: Path, output: Path, selected: dict) -> None:
    """Pool descriptive matched geometry across seeds without using post-success motion."""
    reports = [read_summary(root, f"selected_checkpoint_failure_geometry_seed{seed}") for seed in (300000, 400000)]
    categories = ["no_contact_failure", "contacted_far_failure", "contacted_near_goal_failure", "success"]
    names = ["baseline", "hypernet_fusion"]
    counts = {name: {category: sum(report["runs"][name]["counts"][category] for report in reports)
                     for category in categories} for name in names}
    success = {name: np.concatenate([report["runs"][name]["success"] for report in reports]) for name in names}
    errors = {name: 1000 * np.concatenate([report["runs"][name]["minimum_pre_success_keypoint_error_m"]
                                          for report in reports]) for name in names}
    for name, label in (("baseline", "baseline_best"), ("hypernet_fusion", "original")):
        if int(success[name].sum()) != selected["pooled_scores"][label]["successes"]:
            raise ValueError("geometric and reward-based success counts disagree")
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    bottom = np.zeros(2)
    for category, color in zip(categories, ["#999999", "#E69F00", "#CC79A7", "#009E73"]):
        heights = [counts[name][category] for name in names]
        bars = axes[0].bar(["Baseline ep200", "Main ep100"], heights, bottom=bottom, color=color,
                           label=category.replace("_", " "))
        axes[0].bar_label(bars, labels=[str(value) for value in heights], label_type="center")
        bottom += heights
    axes[0].set_ylabel("Environments across two evaluation seeds")
    axes[0].set_title("Net gain is associated with fewer contacted-far failures")
    axes[0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.08))
    original, changed = success["baseline"], success["hypernet_fusion"]
    for label, mask, color in (("Both succeed", original & changed, "#0072B2"),
                                ("Main only", ~original & changed, "#009E73"),
                                ("Baseline only", original & ~changed, "#D55E00"),
                                ("Both fail", ~original & ~changed, "#999999")):
        axes[1].scatter(errors["baseline"][mask], errors["hypernet_fusion"][mask],
                        label=f"{label}: {int(mask.sum())}", color=color, alpha=0.8)
    maximum = max(values.max() for values in errors.values())
    axes[1].plot([0, maximum], [0, maximum], color="gray", linestyle="--")
    near = 1000 * reports[0]["near_goal_error_threshold_m"]
    axes[1].axhline(near, color="gray", linestyle=":")
    axes[1].axvline(near, color="gray", linestyle=":")
    axes[1].set(xlabel="Baseline minimum pre-success keypoint error (mm)",
                ylabel="Main minimum pre-success keypoint error (mm)",
                title="Paired initial environments\nDotted lines: near-goal threshold")
    axes[1].legend(fontsize=9)
    figure.suptitle("Preselected policies, 100 matched environments: descriptive geometry, not causal phase attribution")
    for extension in ("png", "pdf"):
        figure.savefig(output / f"failure_geometry_pooled.{extension}", dpi=180)
    plt.close(figure)
    baseline_categories = np.concatenate([report["runs"]["baseline"]["categories"] for report in reports])
    main_categories = np.concatenate([report["runs"]["hypernet_fusion"]["categories"] for report in reports])
    gained, lost = ~original & changed, original & ~changed
    (output / "failure_geometry_pooled.json").write_text(json.dumps({
        "counts": counts, "gained": int(gained.sum()), "lost": int(lost.sum()),
        "gained_baseline_failure_categories": {category: int((baseline_categories[gained] == category).sum()) for category in categories[:-1]},
        "lost_main_failure_categories": {category: int((main_categories[lost] == category).sum()) for category in categories[:-1]},
        "near_goal_error_threshold_m": near / 1000,
        "scope": "Descriptive geometry for the two new evaluation seeds; no post-success motion or causal phase inference."
    }, indent=2) + "\n")


def main() -> None:
    """Combine selection, saturation and matched utility checks with explicit scopes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    audit = read_summary(args.results_root, "checkpoint_audit")
    attention = read_summary(args.results_root, "power_plug_attention_logits")
    peg_attention = read_summary(args.results_root, "peg_insertion_attention_logits")
    global_matrix = read_summary(args.results_root, "controlled_matrix_report")
    phase_matrix = read_summary(args.results_root, "phase_matrix_report")
    selected = read_summary(args.results_root, "selected_checkpoint_report")
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    figure, axes = plt.subplots(3, 2, figsize=(15, 15), layout="constrained")
    axis = axes[0, 0]
    for index, (label, values) in enumerate(audit["scores"].items()):
        axis.plot([100, 200, 300], 100 * np.asarray(values), marker="o", color=colors[index], label=label)
    axis.set(title="A. Checkpoint selection changes the apparent gain", xlabel="Saved epoch",
             ylabel="test/mean_score (%)", xticks=[100, 200, 300], ylim=(35, 85))
    axis.legend(fontsize=9)
    axis.text(0.02, 0.02, "Old seed 100000; not independent checkpoint validation",
              transform=axis.transAxes, fontsize=9)
    axis = axes[0, 1]
    for index, (label, data) in enumerate((("Power Plug rollout", attention), ("Peg demonstrations", peg_attention))):
        quantiles = np.asarray(data["signed_tactile_minus_vision_logit_quantiles"])
        axis.errorbar(np.arange(4) + (index - 0.5) * 0.15, quantiles[1],
            yerr=[quantiles[1] - quantiles[0], quantiles[2] - quantiles[1]],
            fmt="o", capsize=4, color=colors[index], label=label)
    axis.axhline(0, color="gray", linestyle="--")
    axis.set(title="B. Large QK margins explain near-fixed attention", xlabel="Attention head",
             ylabel="Tactile minus vision logit; median and 5–95%", xticks=np.arange(4))
    axis.legend(fontsize=9)
    axis = axes[1, 0]
    modulation = audit["main_rollout_modulation"]
    names = ["gamma_vision", "beta_vision", "gamma_tactile", "beta_tactile"]
    values = [100 * modulation[name]["saturated_fraction_abs_ge_095"] for name in names]
    bars = axis.bar(np.arange(4), values, color=[colors[0], colors[0], colors[2], colors[2]])
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.set(title="C. Most generated modulation is tanh-saturated",
             ylabel="Components with absolute value ≥0.95 (%)", ylim=(0, 110),
             xticks=np.arange(4), xticklabels=["Vision γ", "Vision β", "Tactile γ", "Tactile β"])
    axis.set_title("C. Most generated modulation is tanh-saturated\nPre-success inputs; saturation does not imply constancy", fontsize=11)
    configurations = [
        ("Dynamic", global_matrix["pooled_scores"]["original"]),
        ("Mean\nall stages", global_matrix["pooled_scores"]["mean"]),
        ("Zero\nall stages", global_matrix["pooled_scores"]["zero"]),
        ("Mean\npre-contact", phase_matrix["pooled_scores"]["mean_pre"]),
        ("Mean\npost-contact", phase_matrix["pooled_scores"]["mean_post"])]
    axis = axes[1, 1]
    bars = axis.bar(np.arange(5), [100 * data["test/mean_score"] for _, data in configurations], color=colors)
    axis.bar_label(bars, labels=[f"{data['successes']}/{data['environments']}" for _, data in configurations], padding=3)
    axis.set(title="D. Same-checkpoint closed-loop interventions", ylabel="test/mean_score (%)",
             ylim=(0, 110), xticks=np.arange(5), xticklabels=[name for name, _ in configurations])
    axis.set_title("D. Same-checkpoint closed-loop interventions\nTwo new seeds × 50 matched environments; not retraining", fontsize=11)
    comparisons = [
        ("Mean all", global_matrix["paired_differences"]["mean"]),
        ("Zero all", global_matrix["paired_differences"]["zero"]),
        ("Mean pre", phase_matrix["paired_differences"]["mean_pre"]),
        ("Mean post", phase_matrix["paired_differences"]["mean_post"])]
    axis = axes[2, 0]
    for index, (label, data) in enumerate(comparisons):
        lower, upper = data["paired_environment_bootstrap_95_interval_pp"]
        axis.errorbar(data["difference_pp"], index,
            xerr=[[data["difference_pp"] - lower], [upper - data["difference_pp"]]],
            fmt="o", capsize=5, color=colors[index + 1])
    axis.axvline(0, color="gray", linestyle="--")
    axis.set(title="E. Intervention minus dynamic: paired uncertainty", xlabel="Score difference (percentage points)",
             yticks=np.arange(4), yticklabels=[name for name, _ in comparisons])
    axis.set_title("E. Intervention minus dynamic: paired uncertainty\n95% environment bootstrap, conditional on fixed seeds", fontsize=11)
    axis = axes[2, 1]
    names = ["original", "baseline_best", "no_fusion_best", "static_fusion_best"]
    descriptions = ["Main\nep100", "Baseline\nep200", "HyperNet\nno fusion ep300", "Static fusion\nep300"]
    old_scores = [audit["scores"]["hypernet_fusion"][0], audit["scores"]["baseline"][1],
                  audit["scores"]["hypernet_no_fusion"][2], audit["scores"]["static_fusion"][2]]
    positions = np.arange(4)
    axis.bar(positions - 0.18, 100 * np.asarray(old_scores), width=0.36, color="#bbbbbb", label="Old selection seed")
    new_data = [selected["pooled_scores"][name] for name in names]
    bars = axis.bar(positions + 0.18, [100 * data["test/mean_score"] for data in new_data],
                    width=0.36, color=colors[0], label="New seeds 300000/400000")
    axis.bar_label(bars, labels=[f"{data['successes']}/{data['environments']}" for data in new_data], padding=3)
    axis.set(title="F. Preselected checkpoints on two new seeds", ylabel="test/mean_score (%)", ylim=(0, 110),
             xticks=positions, xticklabels=descriptions)
    axis.legend(fontsize=9)
    figure.suptitle("Power Plug mechanism audit: distinguish attention, conditioning and actual utility", fontsize=17)
    args.output_dir.mkdir(parents=True)
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"mechanism_evidence.{extension}", dpi=180)
    plt.close(figure)
    plot_failure_geometry(args.results_root, args.output_dir, selected)
    (args.output_dir / "sources.json").write_text(json.dumps({
        "results_root": str(args.results_root.resolve()),
        "reports": ["checkpoint_audit", "power_plug_attention_logits", "peg_insertion_attention_logits",
                    "controlled_matrix_report", "phase_matrix_report", "selected_checkpoint_report"],
        "scope": "One training seed; exploratory conditional environment uncertainty, not proof of equivalence."
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
