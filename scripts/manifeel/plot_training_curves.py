#!/usr/bin/env python3
"""Plot training and validation losses from a ManiFeel JSONL log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", type=Path, help="Path to logs.json.txt")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: loss_curves.png beside the log)",
    )
    parser.add_argument(
        "--smooth-steps",
        type=int,
        default=500,
        help="Moving-average window for the displayed training curve",
    )
    return parser.parse_args()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    window = min(window, len(values))
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        result.append(running / min(index + 1, window))
    return result


def load_records(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def main() -> None:
    args = parse_args()
    records = load_records(args.log_path)
    train = [
        (int(record["global_step"]), float(record["train_loss"]))
        for record in records
        if "global_step" in record and "train_loss" in record
    ]
    validation = [
        (int(record["epoch"]), float(record["val_loss"]))
        for record in records
        if "epoch" in record and "val_loss" in record
    ]
    if not train:
        raise ValueError(f"No train_loss records found in {args.log_path}")

    output = args.output or args.log_path.with_name("loss_curves.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    train_steps = [step for step, _ in train]
    train_values = [value for _, value in train]
    smoothed = moving_average(train_values, args.smooth_steps)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    fig.suptitle(args.log_path.parent.name + " loss curves")

    axes[0].plot(train_steps, train_values, color="#9aa7b2", alpha=0.18, linewidth=0.5)
    axes[0].plot(train_steps, smoothed, color="#1565c0", linewidth=1.5, label="train loss (moving average)")
    axes[0].set_xlabel("Global step")
    axes[0].set_ylabel("Train loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    if validation:
        val_epochs = [epoch for epoch, _ in validation]
        val_values = [value for _, value in validation]
        axes[1].plot(val_epochs, val_values, marker="o", color="#d84315", linewidth=1.8, label="val loss")
        axes[1].set_xticks(val_epochs)
        axes[1].legend(loc="best")
    else:
        axes[1].text(0.5, 0.5, "No val_loss records", ha="center", va="center")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation loss")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"train_records={len(train)} val_records={len(validation)} output={output}")


if __name__ == "__main__":
    main()
