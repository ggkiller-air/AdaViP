"""Plot six Stage1 v2 training metrics from a read-only JSONL snapshot."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


METRICS = (
    ("loss", "Total loss", True),
    ("action_loss", "Action loss", True),
    ("future_vision_loss", "Future vision loss (unweighted)", True),
    ("future_tactile_loss", "Future tactile loss (unweighted)", True),
    ("vision_latent_variance", "Vision latent variance", False),
    ("tactile_latent_variance", "Tactile latent variance", False),
)


def read_snapshot(path: Path) -> tuple:
    """Ignore only an unfinished final line and deduplicate resumed steps."""
    payload = path.read_bytes()
    complete = payload[:payload.rfind(b"\n") + 1]
    steps, epochs = {}, {}
    weights = None
    for line_number, line in enumerate(complete.splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        event = record.get("event")
        if event == "workspace_ready":
            current = (float(record["lambda_vision"]), float(record["lambda_tactile"]))
            if weights is not None and weights != current:
                raise ValueError("Future loss weights change within this log")
            weights = current
        elif event in ("train", "epoch_end"):
            required = [name for name, _, _ in METRICS if name != "loss" or event == "train"]
            if not all(name in record and math.isfinite(record[name]) and record[name] >= 0
                       for name in required):
                raise ValueError(f"Missing/nonfinite/negative metrics at line {line_number}")
            if event == "train":
                steps[int(record["global_step"])] = record
            else:
                epochs[int(record["epoch"])] = record
    if not steps or weights is None:
        raise ValueError("No training metrics or loss weights in the snapshot")
    records = [steps[step] for step in sorted(steps)]
    completed = [epochs[epoch] for epoch in sorted(epochs)]
    for record in records + completed:
        expected = record["action_loss"] + weights[0] * record["future_vision_loss"] \
            + weights[1] * record["future_tactile_loss"]
        if "loss" in record and not math.isclose(record["loss"], expected, rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError("Logged total loss differs from the weighted objective")
        if "loss" not in record:
            record["loss"] = expected
    return records, completed, weights, complete


def trailing_mean(values: list, window: int) -> list:
    """Smooth logged observations, not unrecorded training steps."""
    if window < 1:
        raise ValueError("Positive smoothing window required")
    result, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def plot_training(metrics_path: Path, output: Path, window: int = 50) -> dict:
    """Save overview/recent six-panel plots, exact source data and a summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    records, epochs, weights, snapshot = read_snapshot(metrics_path)
    output.mkdir(parents=True, exist_ok=False)
    (output / "progress_snapshot.jsonl").write_bytes(snapshot)
    steps = [record["global_step"] for record in records]
    smoothed = {name: trailing_mean([record[name] for record in records], window)
                for name, _, _ in METRICS}
    latest = records[-1]
    last_window = records[-window:]
    summary = {"source": str(metrics_path.resolve()),
               "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
               "snapshot_last_timestamp": datetime.fromtimestamp(latest["timestamp"], timezone.utc).isoformat(),
               "latest_step": steps[-1], "current_epoch": latest["epoch"],
               "train_log_points": len(records), "completed_epochs": len(epochs),
               "smoothing_logged_observations": window,
               "trailing_window_step_start": last_window[0]["global_step"],
               "lambda_vision": weights[0], "lambda_tactile": weights[1],
               "latest_logged_batch": {name: latest[name] for name, _, _ in METRICS},
               "trailing_logged_mean": {name: smoothed[name][-1] for name, _, _ in METRICS},
               "latest_completed_epoch": epochs[-1] if epochs else None,
               "notes": ["Train metrics include the original mixed-input/dropout sampling.",
                         "Future losses are unweighted; total uses the recorded fixed lambdas.",
                         "Smoothing averages logged batches, not every unlogged step.",
                         "Epoch dots are full-epoch sample-weighted averages; epoch total is reconstructed."]}
    fields = ["global_step", "epoch", "timestamp", *[name for name, _, _ in METRICS]]
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    for recent, filename in ((False, "training_metrics.png"), (True, "training_metrics_recent.png")):
        figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
        cutoff = max(steps[0], steps[-1] - 20000) if recent else steps[0]
        for axis, (name, title, logarithmic) in zip(axes.flat, METRICS):
            selected = np.asarray(steps) >= cutoff
            raw = np.asarray([record[name] for record in records])
            axis.plot(np.asarray(steps)[selected], raw[selected], color="#8d99a6", alpha=0.3,
                      linewidth=0.6, label="Logged batch")
            axis.plot(np.asarray(steps)[selected], np.asarray(smoothed[name])[selected], color="#146aab",
                      linewidth=1.7, label=f"Trailing mean ({window} logs)")
            visible_epochs = [record for record in epochs if record["global_step"] >= cutoff]
            axis.scatter([record["global_step"] for record in visible_epochs],
                         [record[name] for record in visible_epochs], color="#cf5c23", s=16,
                         zorder=4, label="Completed epoch mean")
            if logarithmic:
                axis.set_yscale("log")
            axis.set_title(f"{title} | recent mean {smoothed[name][-1]:.5g}")
            axis.set_xlabel("Optimizer step")
            axis.set_ylabel("Loss (log scale)" if logarithmic else "Variance")
            axis.grid(alpha=0.2)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.legend(fontsize=8, loc="upper right")
        stamp = datetime.fromtimestamp(latest["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        figure.suptitle(f"Stage1 v2 single H200 | batch32 | epoch {latest['epoch']} | "
                       f"step {steps[-1]:,} | {stamp} local\n"
                       f"Total = action + {weights[0]:.6f} future vision + {weights[1]:.6f} future tactile",
                       fontsize=13)
        figure.savefig(output / filename, dpi=160)
        if not recent:
            figure.savefig(output / "training_metrics.pdf")
        plt.close(figure)
    indices = np.unique(np.linspace(0, len(records) - 1, min(len(records), 1000), dtype=int))
    inline = {"step": [steps[index] for index in indices],
              "metrics": [{"key": name, "title": title, "log": logarithmic,
                           "raw": [records[index][name] for index in indices],
                           "mean": [smoothed[name][index] for index in indices],
                           "epochs": [[record["global_step"], record[name]] for record in epochs]}
                          for name, title, logarithmic in METRICS]}
    (output / "inline_plot_data.json").write_text(json.dumps(inline, allow_nan=False))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main() -> None:
    """Render current training metrics without touching the trainer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    arguments = parser.parse_args()
    print(json.dumps(plot_training(arguments.metrics, arguments.output, arguments.window), indent=2))


if __name__ == "__main__":
    main()
