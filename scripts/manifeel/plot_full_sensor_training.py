#!/usr/bin/env python3
"""Plot completed Stage0 epochs without changing the live training run."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path


def load_epoch_records(path: Path) -> list[dict]:
    """Read complete JSONL lines, keeping the last occurrence of each epoch."""
    required = ("train/loss", "validation/macro/full", "validation/micro/full")
    epochs: dict[int, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.endswith("\n"):
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict) or "epoch" not in record:
                continue
            if not all(key in record for key in required):
                continue
            if not all(math.isfinite(record[key]) and record[key] > 0 for key in required):
                raise ValueError(f"Invalid loss at {path}:{line_number}")
            epochs[int(record["epoch"])] = record
    if not epochs:
        raise ValueError(f"No completed Stage0 epoch metrics in {path}")
    return [epochs[epoch] for epoch in sorted(epochs)]


def plot_training(metrics_path: Path, output: Path) -> dict:
    """Save raw epoch losses and a JSON summary; validation uses EMA weights."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_epoch_records(metrics_path)
    epochs = [record["epoch"] for record in records]
    best = min(records, key=lambda record: record["validation/macro/full"])
    latest = records[-1]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    overview, validation = axes
    overview.plot(epochs, [record["train/loss"] for record in records],
                  label="Train: 80% full + 20% dropout (raw model)", color="#1976d2")
    overview.plot(epochs, [record["validation/macro/full"] for record in records],
                  label="Val: full input, task macro (EMA)", color="#d35400")
    overview.set_yscale("log")
    overview.set_ylabel("Denoising loss (log scale)")
    overview.set_title("Training vs held-out validation | raw epoch values, no smoothing")
    validation.plot(epochs, [record["validation/macro/full"] for record in records],
                    label="Full input: task macro (checkpoint selection)", color="#d35400")
    validation.plot(epochs, [record["validation/micro/full"] for record in records],
                    label="Full input: sample micro", color="#7b1fa2", linestyle="--")
    validation.scatter([best["epoch"]], [best["validation/macro/full"]],
                       color="#1b7f3a", zorder=5)
    validation.annotate(
        f"Best: epoch {best['epoch']} / {best['validation/macro/full']:.5f}",
        (best["epoch"], best["validation/macro/full"]), xytext=(16, 24),
        textcoords="offset points", arrowprops={"arrowstyle": "->", "color": "#1b7f3a"},
    )
    validation.set_title("Full-input validation | equal task weights vs equal sample weights")
    validation.set_ylabel("Denoising loss")
    validation.set_xlabel("Completed epoch")
    for axis in axes:
        axis.axvline(best["epoch"], color="#1b7f3a", alpha=0.45, linestyle=":")
        axis.grid(alpha=0.22)
        axis.legend(loc="upper left")
    figure.suptitle(
        f"Stage0 full-sensor DP | completed epoch {latest['epoch']} | "
        f"step {latest['global_step']:,}", fontsize=14,
    )
    figure.text(0.02, 0.015,
                "Train and validation differ in input mix and raw/EMA weights. "
                "Validation loss is not simulator success rate.", fontsize=9)
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    summary_keys = ("epoch", "global_step", "train/loss", "validation/macro/full",
                    "validation/micro/full")
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "metrics_path": str(metrics_path.resolve()),
        "png_path": str(output.resolve()),
        "completed_epoch_records": len(records),
        "latest": {key: latest[key] for key in summary_keys},
        "best": {key: best[key] for key in summary_keys},
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary


def main() -> None:
    """Render the completed epochs from one live or finished training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_dir / "plots" / "stage0_loss_curves.png"
    print(json.dumps(plot_training(args.run_dir / "metrics.jsonl", output), indent=2))


if __name__ == "__main__":
    main()
