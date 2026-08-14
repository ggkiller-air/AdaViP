#!/usr/bin/env python3
"""Evaluate a multi-task checkpoint with one ManiFeel task per process."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adavip.manifeel.task_protocol import DEFAULT_TASK_SPECS


def parse_args() -> argparse.Namespace:
    """Parse multi-task checkpoint evaluation arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--n-test-vis", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--task-id",
        choices=[spec.task_id for spec in DEFAULT_TASK_SPECS],
        help="Evaluate one task instead of running the full sequential suite.",
    )
    return parser.parse_args()


def main() -> None:
    """Run isolated task evaluations and aggregate the nine SR values."""
    args = parse_args()
    manifeel_root = REPO_ROOT / "third_party" / "manifeel"
    eval_entrypoint = manifeel_root / "eval.py"
    args.output_dir.mkdir(parents=True, exist_ok=False)

    combined_log: dict[str, object] = {}
    success_rates: list[float] = []
    task_specs = [
        spec
        for spec in DEFAULT_TASK_SPECS
        if args.task_id is None or spec.task_id == args.task_id
    ]
    for spec in task_specs:
        task_output_dir = args.output_dir / spec.task_id
        command = [
            sys.executable,
            str(eval_entrypoint),
            "--checkpoint",
            str(args.checkpoint),
            "--output_dir",
            str(task_output_dir),
            "--device",
            args.device,
            "--task-id",
            spec.task_id,
            "--n-test",
            str(args.n_test),
            "--n-test-vis",
            str(args.n_test_vis),
            "--max-steps",
            str(args.max_steps),
        ]
        subprocess.run(command, cwd=manifeel_root, check=True)

        task_log_path = task_output_dir / "eval_log.json"
        task_log = json.loads(task_log_path.read_text())
        prefix = f"test/{spec.task_id}/"
        combined_log.update(
            (key, value)
            for key, value in task_log.items()
            if key.startswith(prefix)
        )
        success_key = f"{prefix}success_rate"
        if success_key not in combined_log:
            raise RuntimeError(f"Missing {success_key} in {task_log_path}")
        success_rates.append(float(combined_log[success_key]))

    combined_log["test/macro_success_rate"] = float(np.mean(success_rates))
    output_path = args.output_dir / "eval_log.json"
    output_path.write_text(json.dumps(combined_log, indent=2, sort_keys=True) + "\n")
    print(json.dumps(combined_log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
