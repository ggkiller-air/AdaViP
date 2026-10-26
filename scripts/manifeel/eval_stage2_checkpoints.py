#!/usr/bin/env python3
"""Evaluate both Stage2 variants from one downloaded checkpoint package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Run each checkpoint sequentially with the same nine-task protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants", nargs="+", choices=("transformer", "mlp"),
                        default=["transformer", "mlp"])
    parser.add_argument("--epochs", nargs="+", type=int, default=[20, 50, 100, 200, 300])
    args, rollout_args = parser.parse_known_args()
    reserved = ("--checkpoint", "--output-dir", "--checkpoint-root")
    if any(option.split("=", 1)[0] in reserved for option in rollout_args):
        parser.error("Checkpoint/output paths are assigned by the batch evaluator")
    checkpoints = [(variant, epoch, args.checkpoint_root / variant / "checkpoints" / f"epoch{epoch}.ckpt")
                   for variant in args.variants for epoch in args.epochs]
    if len({(variant, epoch) for variant, epoch, _ in checkpoints}) != len(checkpoints):
        parser.error("Duplicate variants or epochs")
    for _, _, path in checkpoints:
        if not path.is_file():
            parser.error(f"Checkpoint not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    entrypoint = Path(__file__).with_name("eval_full_sensor_checkpoint.py")
    reports = {}
    for variant, epoch, checkpoint in checkpoints:
        name = f"{variant}_epoch{epoch}"
        output = args.output_dir / name
        command = [sys.executable, str(entrypoint), "--checkpoint", str(checkpoint.resolve()),
                   "--output-dir", str(output.resolve()), *rollout_args]
        print(f"Evaluating {name}; log: {args.output_dir / (name + '.log')}", flush=True)
        with (args.output_dir / f"{name}.log").open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        report_path = output / ("checkpoint_check.json" if "--check-only" in rollout_args else "evaluation.json")
        reports[name] = json.loads(report_path.read_text())
        (args.output_dir / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
