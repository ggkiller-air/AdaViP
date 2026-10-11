#!/usr/bin/env python3
"""Summarize paired Power Plug rollouts and contact-aligned HyperNet signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adavip.manifeel.contact_probe import contact_phase_masks, environment_level_summary


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read numeric diagnostic arrays without allowing pickled objects."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def plot_environment(rollout: dict, hypernet: dict, phases: dict, environment: int,
                      label: str, output: Path) -> None:
    """Show physical events and modulation for an explicitly identified environment."""
    steps = hypernet["chunk_start_sim_step"]
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for modality_index, modality in enumerate(("vision", "tactile")):
        axes[0].plot(steps, 100 * hypernet["residual_ratio"][:, environment, -1, modality_index], label=modality)
    axes[0].set_ylabel("Modulation / feature (%)")
    axes[0].legend()
    attention = hypernet["attention"][:, environment, -1, :, 1].T
    image = axes[1].pcolormesh(steps, np.arange(attention.shape[0]), attention,
                               shading="nearest", vmin=0, vmax=1, cmap="coolwarm")
    axes[1].set_yticks(np.arange(attention.shape[0]))
    axes[1].set_ylabel("Attention head")
    figure.colorbar(image, ax=axes[1], label="Attention to TacRGB")
    axes[2].plot(np.arange(len(phases["force_norm"])), phases["force_norm"][:, environment], color="black")
    axes[2].set_ylabel("Plug/socket force norm")
    distance = np.linalg.norm(rollout["keypoints_plug"][:, environment]
                               - rollout["keypoints_socket"][:, environment], axis=-1).mean(axis=-1)
    axes[3].plot(np.arange(1, len(distance) + 1), distance)
    axes[3].set_ylabel("Mean keypoint distance (m)")
    axes[3].set_xlabel("Simulator step; policy signals use the newest observation")
    for event, color, name in ((phases["first_contact_step"][environment], "tab:orange", "first target contact"),
                                (phases["first_success_step"][environment], "tab:green", "first benchmark success")):
        if event >= 0:
            for axis in axes:
                axis.axvline(event, color=color, linestyle="--", alpha=0.8)
            axes[0].axvline(event, color=color, linestyle="--", label=name)
    axes[0].legend()
    figure.suptitle(f"{label}: environment {environment}; descriptive internal signals")
    figure.savefig(output / f"{label}_environment{environment}.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Verify benchmark outcomes, compare initial states and summarize physical stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("LABEL", "DIRECTORY"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-threshold", type=float, default=1e-3)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Report already exists: {args.output_dir}")
    labels = [label for label, directory in args.run]
    if len(set(labels)) != len(labels) or any(not label.replace("_", "").isalnum() for label in labels):
        parser.error("run labels must be unique alphanumeric/underscore identifiers")
    args.output_dir.mkdir(parents=True)
    report = {"contact_force_threshold": args.contact_threshold,
              "scope": "Descriptive phase-aligned rollout analysis, not phase-specific intervention evidence.",
              "phase_definition": "Before first plug/socket force threshold crossing versus after it, excluding policy inputs on/after first benchmark success.",
              "runs": {}, "paired_initial_states": {}, "paired_outcomes": {}}
    rollouts, phase_data = {}, {}
    for label, directory in args.run:
        root = Path(directory)
        rollout = load_arrays(root / "rollout_trace.npz")
        log = json.loads((root / "eval_log.json").read_text())
        rewards = rollout["reward_by_step"]
        hypernet_path = root / "hypernet_rollout_trace.npz"
        hypernet = load_arrays(hypernet_path) if hypernet_path.is_file() else None
        chunk_steps = hypernet["chunk_start_sim_step"] if hypernet is not None else np.arange(0, len(rewards), 8)
        phases = contact_phase_masks(rewards, rollout["plug_socket_force"],
                                     rollout["initial_plug_socket_force"], chunk_steps, args.contact_threshold)
        recomputed = float(phases["ever_success"].mean())
        if not np.isclose(recomputed, log["test/mean_score"]):
            raise ValueError(f"{label}: saved reward trace disagrees with test/mean_score")
        threshold_counts = {}
        for threshold in (1e-4, 1e-3, 1e-2):
            threshold_phases = contact_phase_masks(rewards, rollout["plug_socket_force"],
                rollout["initial_plug_socket_force"], chunk_steps, threshold)
            threshold_counts[str(threshold)] = int((threshold_phases["first_contact_step"] >= 0).sum())
        summary = {"directory": str(root.resolve()), "test/mean_score": log["test/mean_score"],
                   "environments": rewards.shape[1], "simulator_steps": len(rewards),
                   "ever_success": phases["ever_success"].tolist(),
                   "first_contact_step": phases["first_contact_step"].tolist(),
                   "first_success_step": phases["first_success_step"].tolist(),
                   "contacted_environments": int((phases["first_contact_step"] >= 0).sum()),
                   "contact_count_threshold_sensitivity": threshold_counts}
        if hypernet is not None:
            summary["phases"] = {}
            for phase in ("pre_contact", "post_contact_pre_success"):
                mask = phases[phase]
                summary["phases"][phase] = {
                    "residual_ratio": environment_level_summary(hypernet["residual_ratio"][:, :, -1], mask),
                    "tactile_attention_by_head": environment_level_summary(hypernet["attention"][:, :, -1, :, 1], mask)}
            cases = sorted(set([0] + [int(indices[0]) for indices in (
                np.flatnonzero(phases["ever_success"]), np.flatnonzero(~phases["ever_success"])) if len(indices)]))
            summary["plotted_environment_indices"] = cases
            for environment in cases:
                plot_environment(rollout, hypernet, phases, environment, label, args.output_dir)
        report["runs"][label] = summary
        rollouts[label] = rollout
        phase_data[label] = phases
    reference = labels[0]
    for label in labels[1:]:
        errors = {key: float(np.max(np.abs(rollouts[label][key] - rollouts[reference][key])))
                  for key in ("initial_state", "initial_plug_pos", "initial_socket_pos", "initial_plug_quat", "initial_socket_quat")}
        matched = all(value <= 1e-7 for value in errors.values())
        report["paired_initial_states"][f"{reference}_vs_{label}"] = {"matched_atol_1e_minus_7": matched, "max_abs_errors": errors}
        if matched:
            original = phase_data[reference]["ever_success"]
            changed = phase_data[label]["ever_success"]
            report["paired_outcomes"][f"{reference}_vs_{label}"] = {
                "both_success": int((original & changed).sum()),
                "only_reference_success": int((original & ~changed).sum()),
                "only_other_success": int((~original & changed).sum()),
                "both_failure": int((~original & ~changed).sum())}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    bars = axes[0].bar(labels, [100 * report["runs"][label]["test/mean_score"] for label in labels])
    bar_labels = [f"{100 * report['runs'][label]['test/mean_score']:.1f}%\n"
                  f"({sum(report['runs'][label]['ever_success'])}/{report['runs'][label]['environments']})"
                  for label in labels]
    axes[0].bar_label(bars, labels=bar_labels)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("test/mean_score (%)")
    axes[0].set_title("Saved EMA policies; one shared evaluation seed")
    for label in labels:
        cumulative = np.maximum.accumulate(rollouts[label]["reward_by_step"] > 0.5, axis=0).mean(axis=1)
        axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, label=label)
    axes[1].set_xlabel("Simulator step")
    axes[1].set_ylabel("Fraction reaching benchmark success")
    axes[1].legend()
    figure.savefig(args.output_dir / "rollout_comparison.png", dpi=170)
    plt.close(figure)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({label: report["runs"][label]["test/mean_score"] for label in labels}, indent=2))


if __name__ == "__main__":
    main()
