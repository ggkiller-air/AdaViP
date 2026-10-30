"""Submit one restartable job eligible for both H200 partitions."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import uuid


def time_seconds(value: str) -> int:
    """Parse Slurm's finite days-hours:minutes:seconds limits."""
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    fields = [int(part) for part in clock.split(":")]
    if len(fields) == 1:
        return int(days) * 86400 + fields[0] * 60
    if len(fields) == 2:
        fields.insert(0, 0)
    if len(fields) != 3:
        raise ValueError(f"Unsupported time limit: {value}")
    return int(days) * 86400 + fields[0] * 3600 + fields[1] * 60 + fields[2]


def common_time_limit() -> tuple[str, dict[str, str]]:
    """Query current limits and use the largest time eligible in both partitions."""
    limits = {}
    for partition in ("gpu", "gpux"):
        result = subprocess.run(["scontrol", "show", "partition", partition, "-o"],
                                check=True, capture_output=True, text=True)
        fields = dict(part.split("=", 1) for part in result.stdout.split() if "=" in part)
        limits[partition] = fields["MaxTime"]
    finite = [v for v in limits.values() if v != "UNLIMITED"]
    if not finite:
        raise ValueError("Expected at least one finite H200 partition time limit")
    return min(finite, key=time_seconds), limits


def main() -> None:
    """Keep legacy queue-file contracts while creating exactly one Slurm job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dependency")
    args = parser.parse_args()
    queue = args.output / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    wave = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    limit, limits = common_time_limit()
    values = {"SOURCE": str(args.source), "OUTPUT": str(args.output), "WAVE": wave}
    values.update(dict(item.split("=", 1) for item in args.set))
    export = "ALL," + ",".join(f"{args.prefix}_{key}={value}" for key, value in values.items())
    command = ["sbatch", "--parsable", "--requeue", "--partition=gpu,gpux", f"--time={limit}",
               f"--job-name={args.name}", f"--export={export}"]
    if args.dependency:
        command.append(f"--dependency={args.dependency}")
    command.append(str(args.source / args.script))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch result: {result.stdout!r}")
    try:
        temporary = queue / f"{wave}.partial"
        temporary.write_text(job_id + "\n")
        os.replace(temporary, queue / f"{wave}.jobs")
        (queue / f"{wave}.json").write_text(json.dumps({"job_id": job_id, "partitions": "gpu,gpux",
            "partition_limits": limits, "time_limit": limit, "command": command}, indent=2) + "\n")
    except Exception:
        subprocess.run(["scancel", job_id], check=False)
        raise
    print(f"gpu,gpux {limit} {job_id}")


if __name__ == "__main__":
    main()
