#!/usr/bin/env python3
"""Verify matched fifty-environment intervention runs across new evaluation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adavip.manifeel.mechanism_probe import paired_success_difference
from adavip.manifeel.contact_probe import contact_phase_masks


def verify_modulation(gamma: np.ndarray, beta: np.ndarray, selected: np.ndarray,
                      intervention: str) -> dict[str, object]:
    """Verify fixed modulation on every selected history, allowing dynamic unselected frames."""
    if gamma.shape != beta.shape or gamma.shape[:2] != selected.shape or selected.dtype != np.bool_:
        raise ValueError("modulation and phase masks must share chunk/environment axes")
    if not np.isfinite(gamma).all() or not np.isfinite(beta).all():
        raise ValueError("nonfinite modulation")
    if intervention not in ("none", "mean", "zero"):
        raise ValueError(f"unsupported modulation intervention: {intervention}")
    if intervention != "none":
        if not selected.any():
            raise ValueError("intervention never applied")
        for values in (gamma, beta):
            applied = values[selected].reshape(-1, *values.shape[-2:])
            if intervention == "zero" and np.any(applied != 0):
                raise ValueError("zero intervention was not applied")
            if intervention == "mean" and np.any(applied != applied[0]):
                raise ValueError("mean intervention varied on selected histories")
    return {"selected_environment_chunks": int(selected.sum()),
            "total_environment_chunks": int(selected.size)}


def main() -> None:
    """Check real rewards and applied modulation before making paired comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[300000, 400000])
    parser.add_argument("--run-template", nargs=2, action="append", metavar=("LABEL", "DIRECTORY_TEMPLATE"))
    parser.add_argument("--reference", default="original")
    parser.add_argument("--scope", default="Same saved EMA checkpoint; inference interventions, not retrained architecture ablations.")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    specifications = args.run_template or [
        ("original", "gpu3_original_batch8_n50_steps500_seed{seed}"),
        ("mean", "gpu4_mean_batch8_n50_steps500_seed{seed}"),
        ("zero", "gpu6_zero_batch8_n50_steps500_seed{seed}")]
    labels = dict(specifications)
    if len(labels) != len(specifications) or args.reference not in labels or len(labels) < 2:
        parser.error("provide at least two unique run labels including the reference")
    labels = {args.reference: labels[args.reference], **labels}
    outcomes = {label: [] for label in labels}
    report = {"seeds": args.seeds, "runs": {}, "initial_states": {},
              "reference": args.reference, "scope": args.scope}
    for seed in args.seeds:
        reference = None
        for label, template in labels.items():
            root = args.results_root / template.format(seed=seed)
            log = json.loads((root / "eval_log.json").read_text())
            with np.load(root / "rollout_trace.npz", allow_pickle=False) as trace:
                reward = trace["reward_by_step"]
                initial = {key: trace[key] for key in ("initial_state", "initial_plug_pos", "initial_socket_pos", "initial_plug_quat", "initial_socket_quat")}
                force, initial_force = trace["plug_socket_force"], trace["initial_plug_socket_force"]
            if reward.shape != (500, 50) or not np.isin(reward, (0, 1)).all():
                raise ValueError(f"unexpected rollout protocol: {root}")
            success = (reward > 0.5).any(axis=0)
            if not np.isclose(success.mean(), log["test/mean_score"]):
                raise ValueError(f"reward or intervention disagreement: {root}")
            if reference is None:
                reference = initial
            else:
                errors = {key: float(np.max(np.abs(value - reference[key]))) for key, value in initial.items()}
                if any(value != 0 for value in errors.values()):
                    raise ValueError(f"unmatched initial conditions: {root}: {errors}")
                report["initial_states"][f"seed{seed}_{args.reference}_vs_{label}"] = errors
            intervention = log.get("diagnostics/hypernet_intervention", "none")
            phase = log.get("diagnostics/intervention_phase", "all")
            verification = None
            hypernet_path = root / "hypernet_rollout_trace.npz"
            if hypernet_path.exists():
                with np.load(hypernet_path, allow_pickle=False) as trace:
                    gamma, beta = trace["gamma"], trace["beta"]
                    selected = np.ones(gamma.shape[:2], dtype=bool)
                    if phase != "all":
                        with np.load(root / "intervention_phase_masks.npz", allow_pickle=False) as masks:
                            selected = masks["selected"]
                        expected = contact_phase_masks(reward, force, initial_force,
                            trace["chunk_start_sim_step"], 0.001)[phase]
                        if not np.array_equal(selected, expected):
                            raise ValueError(f"phase mask disagrees with causal contact/reward history: {root}")
                    verification = verify_modulation(gamma, beta, selected, intervention)
            elif intervention != "none":
                raise ValueError(f"intervention lacks verification trace: {root}")
            outcomes[label].append(success)
            report["runs"][f"seed{seed}_{label}"] = {"directory": str(root), "successes": int(success.sum()),
                "environments": len(success), "test/mean_score": float(success.mean()),
                "intervention": intervention, "phase": phase, "verification": verification}
    outcomes = {label: np.stack(values) for label, values in outcomes.items()}
    report["pooled_scores"] = {label: {"successes": int(values.sum()), "environments": int(values.size),
        "test/mean_score": float(values.mean())} for label, values in outcomes.items()}
    report["paired_differences"] = {label: paired_success_difference(outcomes[args.reference], outcomes[label])
                                    for label in labels if label != args.reference}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    positions = np.arange(len(args.seeds))
    for index, label in enumerate(labels):
        values = 100 * outcomes[label].mean(axis=1)
        width = 0.8 / len(labels)
        bars = axes[0].bar(positions + (index - (len(labels) - 1) / 2) * width, values, width=width, label=label)
        axes[0].bar_label(bars, labels=[f"{int(values_seed.sum())}/50" for values_seed in outcomes[label]])
    axes[0].set_xticks(positions, [str(seed) for seed in args.seeds])
    axes[0].set(xlabel="New evaluation seed", ylabel="test/mean_score (%)", ylim=(0, 105))
    axes[0].legend()
    for index, label in enumerate(report["paired_differences"]):
        item = report["paired_differences"][label]
        lower, upper = item["paired_environment_bootstrap_95_interval_pp"]
        axes[1].errorbar(item["difference_pp"], index,
            xerr=[[item["difference_pp"] - lower], [upper - item["difference_pp"]]], fmt="o", capsize=4)
    axes[1].axvline(0, color="gray", linestyle="--")
    axes[1].set_yticks(np.arange(len(report["paired_differences"])),
                      [f"{label} minus {args.reference}" for label in report["paired_differences"]])
    axes[1].set_xlabel("Paired score difference (pp); bootstrap environments within seed")
    figure.suptitle("Matched evaluation; bootstrap does not measure training-seed uncertainty")
    args.output_dir.mkdir(parents=True)
    figure.savefig(args.output_dir / "controlled_matrix.png", dpi=170)
    plt.close(figure)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pooled_scores": report["pooled_scores"], "paired_differences": report["paired_differences"]}, indent=2))


if __name__ == "__main__":
    main()
