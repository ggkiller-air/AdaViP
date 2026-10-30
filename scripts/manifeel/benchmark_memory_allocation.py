"""Compare NumPy hugepage allocation in an independent, bounded CPU allocation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def child(args: argparse.Namespace) -> None:
    """Measure owned array first-touch and hashing without reading dataset files."""
    allowed = sorted(os.sched_getaffinity(0))
    siblings = (Path(f"/sys/devices/system/cpu/cpu{allowed[0]}/topology/thread_siblings_list")
                .read_text().strip())
    physical = set()
    for group in siblings.split(","):
        ends = list(map(int, group.split("-")))
        physical.update(range(ends[0], ends[-1] + 1))
    os.sched_setaffinity(0, physical.intersection(allowed))
    import numpy as np

    started = time.perf_counter()
    usage = resource.getrusage(resource.RUSAGE_SELF)

    def allocate(_: int) -> tuple:
        begin = time.perf_counter()
        values = np.empty(args.mib * 1024 * 1024 // 4, dtype=np.float32)
        values.fill(0)
        ready = time.perf_counter()
        digest = hashlib.sha256(memoryview(values).cast("B")).hexdigest()
        return values, {"fill_seconds": ready - begin, "hash_seconds": time.perf_counter() - ready,
                        "sha256": digest}

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(allocate, range(2)))
    finished = resource.getrusage(resource.RUSAGE_SELF)
    huge = next((line.split()[1] for line in Path("/proc/self/smaps_rollup").read_text().splitlines()
                 if line.startswith("AnonHugePages:")), "0")
    print(json.dumps({"seconds": time.perf_counter() - started, "numpy": np.__version__,
                      "hugepage_requested": os.environ["NUMPY_MADVISE_HUGEPAGE"],
                      "cpus": sorted(os.sched_getaffinity(0)), "array_mib": args.mib,
                      "user_seconds": finished.ru_utime - usage.ru_utime,
                      "system_seconds": finished.ru_stime - usage.ru_stime,
                      "anon_hugepages_kib": int(huge), "arrays": [row for _, row in records]}), flush=True)


def main() -> None:
    """Run alternating fresh processes; change only each child's hugepage request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mib", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args)
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"host": os.uname().nodename, "job_id": os.environ.get("SLURM_JOB_ID"), "cases": []}
    for flag in ("0", "1", "0", "1"):
        begin = time.perf_counter()
        try:
            result = subprocess.run([sys.executable, __file__, "--child", "--mib", str(args.mib)],
                env={**os.environ, "NUMPY_MADVISE_HUGEPAGE": flag}, capture_output=True,
                text=True, check=True, timeout=args.timeout)
            record = json.loads(result.stdout)
        except subprocess.TimeoutExpired:
            record = {"hugepage_requested": flag, "timeout_seconds": args.timeout}
        except subprocess.CalledProcessError as error:
            record = {"hugepage_requested": flag, "error": error.stderr[-2000:]}
        record["total_wall_seconds"] = time.perf_counter() - begin
        report["cases"].append(record)
        temporary = args.output / "report.partial"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, args.output / "report.json")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
