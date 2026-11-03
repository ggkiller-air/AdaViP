"""Supervise native GPU runs from a separate CPU allocation after interruptions."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .common import atomic_json, fingerprint
from .scheduler import submit


def slurm(command: list[str]) -> str:
    """Bound scheduler calls; a scheduler outage must never trigger a new run."""
    return subprocess.check_output(command, text=True, timeout=30)


def retryable(record: dict, python_failure: bool = False) -> bool:
    """Retry infrastructure interruptions, preserving explicit cancellations/errors."""
    state = record["state"].split()[0].rstrip("+")
    if python_failure or state in {"CANCELLED", "OUT_OF_MEMORY"}:
        return False
    if state in {"TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "COMPLETED"}:
        return True
    if state != "FAILED":
        return False
    # Slurm may report SIGTERM as batch exit15:0 and step-derived exit0:15.
    codes = {record.get("exit_code", ""), record.get("derived_exit_code", "")}
    return bool(codes & {"0:9", "0:15", "9:0", "15:0", "137:0", "143:0"})


def progress_token(output: Path) -> str:
    """Track durable artifacts without loading a model or scanning large datasets."""
    paths = [output / "checkpoints/latest.pt", output / "checkpoints/latest.pkl",
             output / "labels/COMPLETE.json"]
    values = [(str(path.relative_to(output)), path.stat().st_mtime_ns, path.stat().st_size)
              for path in paths if path.exists()]
    labels = output / "labels"
    # Directory mtime changes when an atomic label shard is committed.
    if labels.exists():
        values.append(("labels", labels.stat().st_mtime_ns))
    return fingerprint(values)


def decide(output: Path, job: str, record: dict | None, active: bool,
           previous: dict, now: float, grace: int = 120) -> tuple[str, dict]:
    """Persist retry timing and defer to existing jobs before considering a restart."""
    state = dict(previous)
    if (output / "COMPLETE.json").exists():
        return "complete", state
    if (output / "RECOVERY_PAUSE").exists():
        return "paused", state
    if active:
        return "active", state
    if state.get("observed_job") != job:
        state.update(observed_job=job, terminal_since=now)
    if record is None:
        return "accounting_pending", state
    failure = output / "FAILED.json"
    python_failure = failure.exists() and str(json.loads(failure.read_text()).get("job_id")) == job
    if not retryable(record, python_failure):
        return "stopped", state
    token = progress_token(output)
    attempts = state.get("no_progress_retries", 0) if token == state.get("progress_token") else 0
    if attempts >= 8:
        return "no_progress_limit", state
    delay = max(grace, min(1800, 60 * 2 ** min(attempts, 5)))
    state.update(progress_token=token, no_progress_retries=attempts,
                 retry_after=state["terminal_since"] + delay)
    return ("retry" if now >= state["retry_after"] else "backoff"), state


def register_job(registry_path: Path, output: Path, job: str) -> None:
    """Adopt a continuation already submitted by the GPU worker itself."""
    with open(str(registry_path) + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads(registry_path.read_text())
        for run in registry["runs"]:
            if Path(run["output"]).resolve() == output.resolve():
                run["job_id"] = job
        atomic_json(registry_path, registry)


def queue_snapshot() -> list[dict]:
    """Include pending/requeued/completing jobs so an allocation is never duplicated."""
    rows = []
    for line in slurm(["squeue", "--me", "-h", "-o", "%i|%T|%j"]).splitlines():
        job, state, name = line.split("|", 2)
        rows.append({"job": job, "state": state, "name": name})
    return rows


def active_job(run: dict, queue: list[dict]) -> str | None:
    """Find this run, including continuations not yet recorded in the registry."""
    output = Path(run["output"])
    task = json.loads(Path(run["config"]).read_text()).get("task", "multi")
    name = f"uvip_{run['model']}_{task}"
    for row in queue:
        if row["job"] == str(run.get("job_id")):
            return row["job"]
        if row["name"] == name:
            details = slurm(["scontrol", "show", "job", row["job"], "-o"])
            match = re.search(r"(?:^| )StdOut=(\S+)", details)
            if match and Path(match[1]).parent.resolve() == output.resolve():
                return row["job"]
    return None


def accounting(jobs: list[str]) -> dict[str, dict]:
    """Query all parent allocations together, ignoring cancelled child steps."""
    if not jobs:
        return {}
    result = {}
    command = ["sacct", "-X", "-n", "-P", "-j", ",".join(jobs),
               "--format=JobIDRaw,State,ExitCode,DerivedExitCode"]
    for line in slurm(command).splitlines():
        job, state, exit_code, derived, *_ = line.split("|")
        result[job] = {"state": state, "exit_code": exit_code, "derived_exit_code": derived}
    return result


def poll(registry_path: Path, grace: int = 120) -> bool:
    """Recover interrupted native runs while leaving other allocations untouched."""
    registry = json.loads(registry_path.read_text())
    directory = registry_path.parent / "recovery"
    directory.mkdir(exist_ok=True)
    queue = queue_snapshot()
    records = accounting([str(run["job_id"]) for run in registry["runs"] if run.get("job_id")])
    statuses = []
    for run in registry["runs"]:
        output = Path(run["output"])
        state_path = directory / (fingerprint(str(output))[:16] + ".json")
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        existing = active_job(run, queue)
        job = str(run.get("job_id", ""))
        if existing and existing != job:
            register_job(registry_path, output, existing)
            job = existing
        action, state = decide(output, job, records.get(job), bool(existing), previous,
                               time.time(), grace)
        if action == "retry":
            # A fresh queue check closes the normal-continuation/accounting gap.
            existing = active_job(run, queue_snapshot())
            if existing:
                register_job(registry_path, output, existing)
                action = "active"
            else:
                os.environ["UVIP_SLURM_ACCOUNT"] = run.get("account", "yangyc")
                os.environ["UVIP_RUN_REGISTRY"] = str(registry_path)
                new_job = submit(run.get("source", registry["source"]), output,
                                 run["config"], run["model"])
                state.update(no_progress_retries=state.get("no_progress_retries", 0) + 1,
                             previous_job=job, submitted_job=new_job,
                             retries=state.get("retries", 0) + 1)
                with (directory / "events.jsonl").open("a") as handle:
                    handle.write(json.dumps({"time": time.time(), "output": str(output),
                                             "previous_job": job, "job_id": new_job,
                                             "previous_state": records[job]}) + "\n")
                job, action = new_job, "resubmitted"
        state.update(status=action, output=str(output), job_id=job, checked_at=time.time())
        atomic_json(state_path, state)
        statuses.append(state)
    atomic_json(directory / "status.json", {"checked_at": time.time(), "runs": statuses})
    return all(state["status"] == "complete" for state in statuses)


def main() -> None:
    """Keep a single durable CPU supervisor, independent of GPU job lifetimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    directory = args.registry.parent / "recovery"
    directory.mkdir(exist_ok=True)
    with (directory / "supervisor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A recovery supervisor already owns this registry")
        while True:
            try:
                complete = poll(args.registry)
            except Exception as error:
                atomic_json(directory / "ERROR.json", {"time": time.time(), "error": str(error)})
                print(f"Recovery poll deferred: {error}", flush=True)
                if args.once:
                    raise
            else:
                (directory / "ERROR.json").unlink(missing_ok=True)
                if complete or args.once:
                    break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
