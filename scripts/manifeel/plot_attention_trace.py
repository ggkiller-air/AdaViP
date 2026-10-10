#!/usr/bin/env python3
"""Plot time-varying modality routing from one recorded evaluation rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize_attention(weights: np.ndarray, boundaries: tuple[int, int] | None
                        ) -> dict[str, object]:
    """Summarize the last observation's tactile weight by explicitly labeled stage."""
    if weights.ndim != 5 or weights.shape[-1] != 2:
        raise ValueError("weights must be [chunk, environment, observation, head, modality]")
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=-1), 1, atol=1e-4):
        raise ValueError("invalid modality attention probabilities")
    tactile = weights[:, :, -1, :, 1]
    result: dict[str, object] = {
        "mean_tactile_weight": float(tactile.mean()),
        "mean_tactile_weight_by_head": tactile.mean(axis=(0, 1)).tolist(),
    }
    if boundaries is not None:
        approach_end, contact_end = boundaries
        if not 0 < approach_end < contact_end < len(weights):
            raise ValueError("phase boundaries must satisfy 0 < approach_end < contact_end < chunks")
        result["stage_tactile_weight"] = {
            name: float(tactile[start:end].mean())
            for name, start, end in (
                ("approach", 0, approach_end),
                ("contact", approach_end, contact_end),
                ("manipulation", contact_end, len(weights)),
            )
        }
    return result


def main() -> None:
    """Save a per-head heatmap and a compact JSON stage summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--env", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approach-end", type=int)
    parser.add_argument("--contact-end", type=int)
    args = parser.parse_args()
    if (args.approach_end is None) != (args.contact_end is None):
        parser.error("provide both stage boundaries or neither")
    boundaries = ((args.approach_end, args.contact_end)
                  if args.approach_end is not None else None)
    with np.load(args.run / "attention_trace.npz") as trace:
        weights = trace["weights"]
        ratios = trace["fusion_to_vision_norm"]
        gate = float(trace["fusion_gate"])
    summary = summarize_attention(weights, boundaries)
    if args.env < 0 or args.env >= weights.shape[1]:
        parser.error(f"env must be between 0 and {weights.shape[1] - 1}")
    tactile = weights[:, args.env, -1, :, 1].T
    figure, (heatmap, curve) = plt.subplots(
        2, 1, figsize=(11, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    image = heatmap.imshow(tactile, origin="lower", aspect="auto", vmin=0, vmax=1,
                           extent=(0, len(weights), -0.5, tactile.shape[0] - 0.5),
                           cmap="coolwarm")
    figure.colorbar(image, ax=heatmap, label="Attention to TacRGB")
    heatmap.set_ylabel("Attention head")
    heatmap.set_yticks(np.arange(tactile.shape[0]))
    curve.plot(np.arange(len(weights)) + 0.5, tactile.mean(axis=0), label="TacRGB attention")
    curve.plot(np.arange(len(weights)) + 0.5, ratios[:, args.env, -1],
               label="Fusion / vision feature norm")
    curve.set_ylim(bottom=0)
    curve.set_ylabel("Fraction / norm ratio")
    curve.set_xlabel("Action chunk (last observation)")
    curve.legend(loc="upper right")
    if boundaries is not None:
        for boundary in boundaries:
            heatmap.axvline(boundary, color="black", linestyle="--", linewidth=1)
            curve.axvline(boundary, color="black", linestyle="--", linewidth=1)
    rollout = args.run / "rollout_trace.npz"
    if rollout.is_file():
        with np.load(rollout) as trace:
            if "success_by_step" in trace:
                success = trace["success_by_step"][:, args.env]
                success = success.reshape(success.shape[0], -1)
                succeeded = np.flatnonzero(success.any(axis=-1))
                summary["first_success_sim_step_env"] = (
                    int(succeeded[0]) if len(succeeded) else None)
    output = args.output or (args.run / f"attention_env{args.env}.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.suptitle(f"Environment {args.env} | tanh(fusion gate)={gate:.3f}")
    figure.savefig(output, dpi=170)
    plt.close(figure)
    summary.update({"environment": args.env, "chunks": len(weights), "fusion_gate": gate})
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
