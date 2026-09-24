#!/usr/bin/env python3
"""Plot training losses for the three power-plug modality runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "308521 · Vision": Path(
        "/data/wangzihao/outputs/manifeel/dp_power_plug_vision_seed42/logs.json.txt"
    ),
    "308522 · Vision + tactile RGB": Path(
        "/data/wangzihao/outputs/manifeel/dp_power_plug_tacrgb_seed42/logs.json.txt"
    ),
    "308523 · Vision + tactile force field": Path(
        "/data/wangzihao/outputs/manifeel/dp_power_plug_tacff_seed42/logs.json.txt"
    ),
}
SMOOTHING_WINDOW = 100
OUTPUT = Path(__file__).resolve().parents[1] / "loss_308521_308522_308523.png"


def load_losses(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = []
    losses = []
    val_steps = []
    val_losses = []
    with path.open(encoding="utf-8") as log_file:
        for line in log_file:
            record = json.loads(line)
            if "train_loss" in record:
                steps.append(record["global_step"])
                losses.append(record["train_loss"])
            if "val_loss" in record:
                val_steps.append(record["global_step"])
                val_losses.append(record["val_loss"])
    return (
        np.asarray(steps),
        np.asarray(losses),
        np.asarray(val_steps),
        np.asarray(val_losses),
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / window


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (train_ax, val_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1]},
    )
    colors = ["#2667a8", "#d56a1f", "#18855b"]

    for (label, path), color in zip(RUNS.items(), colors):
        steps, losses, val_steps, val_losses = load_losses(path)
        train_ax.plot(steps, losses, color=color, alpha=0.13, linewidth=0.65)
        smooth = moving_average(losses, SMOOTHING_WINDOW)
        smooth_steps = steps[SMOOTHING_WINDOW - 1 :]
        train_ax.plot(smooth_steps, smooth, color=color, linewidth=2.1, label=label)
        val_ax.plot(
            val_steps,
            val_losses,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.1,
            markeredgewidth=0,
            label=label,
        )

    for ax in (train_ax, val_ax):
        ax.set_yscale("log")
        ax.set_xlim(0, 9999)
        ax.grid(True, which="major", color="#c9ced3", linewidth=0.7, alpha=0.75)
        ax.grid(True, which="minor", axis="y", color="#e2e5e8", linewidth=0.5, alpha=0.55)

    train_ax.set_ylabel("Training loss (log scale)")
    train_ax.set_title("Power-plug Training and Validation Loss Comparison")
    train_ax.legend(frameon=True, framealpha=0.95)
    train_ax.text(
        0.01,
        0.02,
        f"Solid: {SMOOTHING_WINDOW}-step moving average  ·  Faint: raw loss",
        transform=train_ax.transAxes,
        ha="left",
        va="bottom",
        color="#4d555c",
        fontsize=9,
    )
    val_ax.set_xlabel("Training step")
    val_ax.set_ylabel("Validation loss (log scale)")
    val_ax.text(
        0.01,
        0.02,
        "Validation measured every 10 epochs",
        transform=val_ax.transAxes,
        ha="left",
        va="bottom",
        color="#4d555c",
        fontsize=9,
    )
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
