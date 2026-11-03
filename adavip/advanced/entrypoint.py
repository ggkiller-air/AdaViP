"""Record run lifecycle/failures and dispatch a native model pipeline."""

import argparse
import importlib
import os
from pathlib import Path
import socket
import sys
import traceback

from .common import atomic_json, stop_requested
from .progress import update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",choices=["ftp1","tabero"],required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    failure = output / "FAILED.json"
    if failure.exists():
        failure.rename(output / f"FAILED_before_{os.environ.get('SLURM_JOB_ID','local')}.json")
    atomic_json(output / "runtime.json",{"job_id":os.environ.get("SLURM_JOB_ID"),"host":socket.gethostname(),
                                        "pid":os.getpid(),"affinity":sorted(os.sched_getaffinity(0)),"state":"running"})
    sys.argv = [sys.argv[0],"--config",args.config,"--output",args.output]
    try:
        importlib.import_module(f"adavip.advanced.{args.model}_run").main()
    except BaseException as error:
        atomic_json(failure,{"error":str(error),"traceback":traceback.format_exc(),
                             "job_id":os.environ.get("SLURM_JOB_ID")})
        update()
        raise
    atomic_json(output / "runtime.json",{"job_id":os.environ.get("SLURM_JOB_ID"),"host":socket.gethostname(),
                                        "state":"continuation_required" if stop_requested() else "completed"})
    update()


if __name__=="__main__":
    main()
