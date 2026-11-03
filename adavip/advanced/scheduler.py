"""Submit/resume one immutable native-model run on gpu,gpux."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from .common import atomic_json


def partition_limit(name):
    text = subprocess.check_output(["scontrol","show","partition",name],text=True)
    match = re.search(r"\bMaxTime=(\S+)",text)
    value = match.group(1)
    days,clock = value.split("-",1) if "-" in value else ("0",value)
    hours,minutes,seconds = map(int,clock.split(":"))
    return int(days)*86400+hours*3600+minutes*60+seconds


def submit(source,output,config,model,dependency=None):
    seconds = min(partition_limit("gpu"),partition_limit("gpux"))
    days,remain = divmod(seconds,86400)
    hours,remain = divmod(remain,3600)
    minutes,seconds = divmod(remain,60)
    walltime = f"{days}-{hours:02}:{minutes:02}:{seconds:02}"
    output = Path(output)
    args = ["sbatch","--parsable","--partition=gpu,gpux",f"--time={walltime}",
            f"--job-name=uvip_{model}_{json.loads(Path(config).read_text()).get('task','multi')}",
            f"--output={output}/slurm_%j.out",f"--error={output}/slurm_%j.err",
            "--export=ALL,"+",".join(f"{key}={value}" for key,value in {
                "UVIP_SOURCE":source,"UVIP_OUTPUT":output,"UVIP_CONFIG":config,"UVIP_MODEL":model}.items())]
    account = os.environ.get("UVIP_SLURM_ACCOUNT",os.environ.get("SLURM_JOB_ACCOUNT","yangyc"))
    args.append(f"--account={account}")
    if account=="overload":
        args.append("--qos=overload")
    if dependency:
        args.append(f"--dependency={dependency}")
    args.append(str(Path(source) / "slurm/advanced/native_uvip.sbatch"))
    job = subprocess.check_output(args,text=True).strip().split(";")[0]
    atomic_json(output / "submission.json",{"job_id":job,"partition":"gpu,gpux","walltime":walltime,
                                           "args":args,"dependency":dependency})
    registry_path = os.environ.get("UVIP_RUN_REGISTRY")
    if registry_path:
        import fcntl
        with open(str(registry_path)+".lock","w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            registry = json.loads(Path(registry_path).read_text())
            for run in registry["runs"]:
                if str(Path(run["output"]))==str(output):
                    run["job_id"] = job
            atomic_json(registry_path,registry)
        from .progress import update
        update()
    print(job,flush=True)
    return job


def main():
    parser = argparse.ArgumentParser()
    for name in ("source","output","config","model"):
        parser.add_argument(f"--{name}",required=True)
    parser.add_argument("--dependency")
    args = parser.parse_args()
    submit(**vars(args))


if __name__=="__main__":
    main()
