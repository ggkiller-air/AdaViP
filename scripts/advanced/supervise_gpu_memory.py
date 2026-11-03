"""Attach administrator-authorized memory reservations to our allocated GPUs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def latest_usage(output, job, dp=False):
    path = output / (f"gpu_monitor_{job}.log" if dp else f"gpu_{job}.log")
    if not path.exists():
        return None
    # Read a bounded tail even after several days of telemetry.
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 8192))
        lines = handle.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 6 and fields[1].startswith("GPU-"):
            total = float(fields[3].split()[0])
            used = float(fields[4].split()[0])
            return used / total
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    directory = args.registry.parent / "gpu_memory_guard"
    directory.mkdir(exist_ok=True)
    lock = (directory / "supervisor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    holders, handles = {}, []
    python = "/public/home/wangzihao/.venvs/uvip-ftp1/bin/python"
    script = "/public/home/wangzihao/projects/AdaViP-main/scripts/manifeel/reserve_teacher_gpu_memory.py"
    dp_output = Path("/data/wangzihao/experiments/univtac/l1_six_task_batch64_ep500_seed42_20261102_233553/train")
    while True:
        registry = json.loads(args.registry.read_text())
        states = {}
        result = subprocess.check_output(["squeue", "-u", "wangzihao", "-h", "-o", "%i|%T|%j"], text=True)
        for line in result.splitlines():
            job, state, name = line.split("|", 2)
            states[job] = (state, name)
        runs = [(str(run.get("job_id", "")), Path(run["output"]), False) for run in registry["runs"]]
        runs += [(job, dp_output, True) for job, (state, name) in states.items()
                 if name.startswith("univtac_L1_six")]
        for job, output, dp in runs:
            if states.get(job, (None,))[0] != "RUNNING" or (output / "COMPLETE.json").exists():
                continue
            if job in holders and holders[job].poll() is None:
                continue
            usage = latest_usage(output, job, dp)
            if usage is not None and usage >= .15:
                continue
            guard = output / "gpu_memory_guard" / job
            guard.mkdir(parents=True, exist_ok=True)
            # Reuse the existing quiet holder; release at training completion
            # or this allocation's soft-stop boundary. Allocation ends also
            # terminate its srun step through Slurm.
            targets = {"COMPLETE.json": output / "COMPLETE.json",
                       "SAFE_STOP.json": output / f"stop_{job}.txt"}
            for name, target in targets.items():
                link = guard / name
                if not link.is_symlink() and not link.exists():
                    link.symlink_to(target)
            log = (guard / "holder.log").open("a")
            handles.append(log)
            command = ["srun", f"--jobid={job}", "--overlap", "--nodes=1", "--ntasks=1",
                       "--cpus-per-task=1", "--cpu-bind=cores", "--unbuffered", python, "-u", script,
                       "--hold", "--output", str(guard), "--fraction", "0.16"]
            holders[job] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            print(json.dumps({"event": "memory_guard_started", "job": job,
                              "observed_fraction": usage, "reserved_fraction": .16,
                              "srun_pid": holders[job].pid, "output": str(guard)}), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
