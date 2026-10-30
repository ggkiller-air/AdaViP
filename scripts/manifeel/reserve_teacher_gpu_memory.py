"""Reserve administrator-requested VRAM inside only the listed teacher jobs."""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def hold(output: Path, fraction: float) -> None:
    """Keep a zero-filled, non-model tensor alive without consuming RNG or compute."""
    import torch

    output.mkdir(parents=True, exist_ok=True)
    with (output / "gpu_memory_reservation.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if not 0.15 <= fraction <= 0.25:
            raise ValueError("Reservation must be between15% and25% of this allocated GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly the teacher job's one allocated CUDA device")
        free, total = torch.cuda.mem_get_info()
        amount = int(total * fraction)
        if free < amount + 4 * 1024 ** 3:
            raise RuntimeError("Insufficient spare VRAM for the requested reservation")
        reservation = torch.empty(amount, dtype=torch.uint8, device="cuda:0")
        reservation.zero_()
        torch.cuda.synchronize()
        record = {"event": "gpu_memory_reserved", "timestamp": time.time(), "pid": os.getpid(),
                  "job_id": os.environ.get("SLURM_JOB_ID"), "fraction": fraction,
                  "reserved_bytes": reservation.numel(), "total_bytes": total,
                  "device": torch.cuda.get_device_name(), "part_of_model": False,
                  "part_of_optimizer": False, "saved_in_checkpoint": False}
        (output / "gpu_memory_reservation.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        while not (output / "COMPLETE.json").exists() and not (output / "SAFE_STOP.json").exists():
            time.sleep(30)


def supervise(root: Path, fraction: float) -> None:
    """Follow original/continuation queue IDs and attach one quiet holder per run."""
    processes = {}
    handles = []
    while True:
        runs = json.loads((root / "all_twelve_submissions.json").read_text())
        result = subprocess.run(["squeue", "--user", getpass.getuser(), "--noheader", "--format=%i|%T"],
                                check=True, capture_output=True, text=True)
        states = dict(line.strip().split("|", 1) for line in result.stdout.splitlines())
        active = 0
        for run in runs:
            output = Path(run["output"])
            if (output / "COMPLETE.json").exists():
                continue
            ids = {run["job_id"]}
            for path in (output / "queue").glob("*.jobs"):
                ids.update(path.read_text().split())
            for job_id in ids:
                if job_id not in states:
                    continue
                active += 1
                if states[job_id] != "RUNNING":
                    continue
                if job_id in processes and processes[job_id].poll() is None:
                    continue
                log = (output / f"gpu_memory_reservation_{job_id}.log").open("a")
                handles.append(log)
                command = ["srun", f"--jobid={job_id}", "--overlap", "--ntasks=1", "--cpus-per-task=1",
                           "--cpu-bind=cores", "--unbuffered", sys.executable, "-u", str(Path(__file__).resolve()),
                           "--hold", "--output", str(output), "--fraction", str(fraction)]
                processes[job_id] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                print(json.dumps({"event": "reservation_step_started", "job_id": job_id,
                                  "pid": processes[job_id].pid}), flush=True)
        if not active:
            return
        time.sleep(60)


def main() -> None:
    """Run a tiny CPU supervisor or a holder within an existing GPU allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fraction", type=float, default=0.16)
    parser.add_argument("--hold", action="store_true")
    args = parser.parse_args()
    if args.hold:
        hold(args.output, args.fraction)
    else:
        supervise(args.root, args.fraction)


if __name__ == "__main__":
    main()
