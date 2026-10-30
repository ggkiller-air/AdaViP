"""Submit the twelve independent batch8 single-task teacher runs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    """Preserve partial submission records and refuse duplicate launch manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "submissions.json"
    if manifest.exists():
        raise FileExistsError(f"Submission manifest exists; inspect before resubmitting: {manifest}")
    records = []
    for task, short in (("power_plug_insertion", "plug"), ("usb_insertion", "usb"), ("ball_sorting", "sorting")):
        for sensors in ("five", "two"):
            for dropout in (0, 1):
                name = f"{short}_{sensors}_{'drop40' if dropout else 'nodrop'}_batch8_ep400_seed42"
                output = args.output / name
                command = ["python3", str(args.source / "scripts/manifeel/submit_partition_job.py"),
                           "--prefix", "ST", "--source", str(args.source), "--output", str(output),
                           "--script", "slurm/manifeel/train_single_task_teacher.sbatch", "--name", name,
                           "--set", f"TASK={task}", "--set", f"SENSORS={sensors}", "--set", f"DROPOUT={dropout}"]
                record = {"name": name, "task": task, "sensors": sensors, "dropout": bool(dropout),
                          "output": str(output), "command": command}
                if not args.dry_run:
                    # Persist intent before invoking a potentially successful external mutation.
                    records.append(record)
                    manifest.write_text(json.dumps(records, indent=2) + "\n")
                    result = subprocess.run(command, capture_output=True, text=True, check=True)
                    record["submission"] = result.stdout.strip()
                    record["job_id"] = result.stdout.strip().split()[-1]
                    manifest.write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
