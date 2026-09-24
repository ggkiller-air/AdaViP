#!/usr/bin/env python3
"""Compare training and validation losses from multiple ManiFeel JSONL logs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LossSeries:
    """Training and validation loss points from one run."""

    label: str
    train_steps: list[int]
    train_losses: list[float]
    val_steps: list[int]
    val_losses: list[float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "LOG_PATH"),
        required=True,
        help="Run label and path to logs.json.txt; repeat for each run",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output PNG path")
    parser.add_argument(
        "--smooth-steps",
        type=int,
        default=500,
        help="Moving-average window for training loss",
    )
    parser.add_argument(
        "--title",
        default="Training and Validation Loss Comparison",
        help="Figure title",
    )
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    """Return a trailing moving average with a partial initial window."""
    if window <= 0:
        raise ValueError("smooth-steps must be positive")
    result: list[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        result.append(running_sum / min(index + 1, window))
    return result


def load_losses(label: str, path: Path) -> LossSeries:
    """Load a run, keeping the last record for duplicate global steps."""
    train_by_step: dict[int, float] = {}
    validation_by_step: dict[int, float] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "train_loss" in record and "global_step" in record:
                train_by_step[int(record["global_step"])] = float(record["train_loss"])
            if "val_loss" in record and "global_step" in record:
                validation_by_step[int(record["global_step"])] = float(record["val_loss"])

    train = sorted(train_by_step.items())
    validation = sorted(validation_by_step.items())
    if not train:
        raise ValueError(f"No train_loss records found in {path}")
    if not validation:
        raise ValueError(f"No val_loss records found in {path}")
    if any(loss <= 0 for _, loss in train + validation):
        raise ValueError(f"Log scale requires positive losses in {path}")

    return LossSeries(
        label=label,
        train_steps=[step for step, _ in train],
        train_losses=[loss for _, loss in train],
        val_steps=[step for step, _ in validation],
        val_losses=[loss for _, loss in validation],
    )


def main() -> None:
    """Render the comparison figure."""
    args = parse_args()
    series = [load_losses(label, Path(path)) for label, path in args.run]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (train_axis, val_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1]},
    )
    colors = plt.get_cmap("tab10").colors
    for index, run in enumerate(series):
        color = colors[index % len(colors)]
        train_axis.plot(
            run.train_steps,
            run.train_losses,
            color=color,
            alpha=0.08,
            linewidth=0.45,
        )
        train_axis.plot(
            run.train_steps,
            moving_average(run.train_losses, args.smooth_steps),
            color=color,
            linewidth=2.0,
            label=run.label,
        )
        val_axis.plot(
            run.val_steps,
            run.val_losses,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            markeredgewidth=0,
            label=run.label,
        )

    for axis in (train_axis, val_axis):
        axis.set_yscale("log")
        axis.grid(True, which="major", linewidth=0.7, alpha=0.7)
        axis.grid(True, which="minor", axis="y", linewidth=0.5, alpha=0.45)
        axis.legend(frameon=True, framealpha=0.95)

    figure.suptitle(args.title)
    train_axis.set_ylabel("Training loss (log scale)")
    train_axis.text(
        0.01,
        0.02,
        f"Solid: {args.smooth_steps}-step moving average; faint: raw loss",
        transform=train_axis.transAxes,
        color="#4d555c",
        fontsize=9,
    )
    val_axis.set_xlabel("Global training step")
    val_axis.set_ylabel("Validation loss (log scale)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    print(f"output={args.output}")
    for run in series:
        print(
            f"{run.label}: train_records={len(run.train_losses)} "
            f"val_records={len(run.val_losses)} max_step={run.train_steps[-1]}"
        )


if __name__ == "__main__":
    main()
