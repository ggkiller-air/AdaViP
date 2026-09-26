#!/usr/bin/env python3
"""Compare per-epoch training and validation losses through a chosen epoch."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_losses(path: Path, max_epoch: int) -> tuple[list[int], list[float], list[int], list[float]]:
    """Load per-epoch mean training losses and the latest validation records."""
    train_by_epoch: dict[int, list[float]] = defaultdict(list)
    val_by_epoch: dict[int, float] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            epoch = int(record["epoch"])
            if epoch > max_epoch:
                continue
            if "train_loss" in record:
                train_by_epoch[epoch].append(float(record["train_loss"]))
            if "val_loss" in record:
                val_by_epoch[epoch] = float(record["val_loss"])

    if not train_by_epoch or not val_by_epoch:
        raise ValueError(f"Missing training or validation loss in {path}")
    train_epochs = sorted(train_by_epoch)
    val_epochs = sorted(val_by_epoch)
    return (
        train_epochs,
        [sum(train_by_epoch[epoch]) / len(train_by_epoch[epoch]) for epoch in train_epochs],
        val_epochs,
        [val_by_epoch[epoch] for epoch in val_epochs],
    )


def main() -> None:
    """Plot all requested runs on shared epoch axes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("LABEL", "LOG"), required=True)
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for (label, path), color in zip(args.run, plt.get_cmap("tab10").colors):
        train_epochs, train_losses, val_epochs, val_losses = load_losses(Path(path), args.max_epoch)
        axes[0].plot(train_epochs, train_losses, color=color, linewidth=1.8, label=label)
        axes[1].plot(val_epochs, val_losses, color=color, linewidth=1.8, marker="o", markersize=2.5)
        print(f"{label}: train through epoch {train_epochs[-1]}, val through epoch {val_epochs[-1]}, val={val_losses[-1]:.6f}")

    for axis, ylabel in zip(axes, ("Mean training loss", "Validation loss")):
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].set_xlabel("Epoch")
    axes[1].set_xlim(0, args.max_epoch)
    figure.suptitle(f"ManiFeel training jobs: loss through epoch {args.max_epoch}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
