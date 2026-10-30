"""Ensure future submissions create one job eligible in both H200 partitions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.manifeel.submit_partition_job import time_seconds


@pytest.mark.parametrize("value,expected", [("4-04:00:00", 360000), ("2-00:00:00", 172800),
                                           ("01:00:00", 3600), ("30", 1800), ("05:30", 330)])
def test_slurm_time_limits(value: str, expected: int) -> None:
    assert time_seconds(value) == expected


@pytest.mark.parametrize("script,prefix,extra", [
    ("submit_language_conditioning.sh", "LC", {"LC_VARIANT": "l1"}),
    ("submit_stage2_encoder_ablation.sh", "S2A", {"S2A_STRATEGY": "s3"}),
    ("submit_stage2_t2.sh", "STAGE2", {"STAGE2_MODE": "full"}),
    ("submit_stage2_mlp_t2.sh", "STAGE2_MLP", {"STAGE2_MLP_MODE": "full"}),
])
def test_one_job_for_both_partitions(tmp_path: Path, script: str, prefix: str, extra: dict) -> None:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    control = binaries / "scontrol"
    control.write_text('#!/usr/bin/env python3\nimport sys\n'
                       'print("PartitionName=" + sys.argv[3] + " MaxTime=" + '
                       '("4-04:00:00" if sys.argv[3]=="gpu" else "2-00:00:00"))\n')
    batch = binaries / "sbatch"
    batch.write_text('#!/usr/bin/env python3\nimport sys,os,json\n'
                     'with open(os.environ["SUBMIT_CALLS"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                     'print("12345;cluster")\n')
    control.chmod(0o755)
    batch.chmod(0o755)
    output = tmp_path / "run"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "latest.ckpt").touch()
    calls = tmp_path / "calls.jsonl"
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
               SUBMIT_CALLS=str(calls), **extra)
    env.update({prefix + "_SOURCE": str(tmp_path / "source"), prefix + "_OUTPUT": str(output),
                prefix + "_DEPENDENCY": "afterok:123"})
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["bash", str(root / "scripts/manifeel" / script)], env=env, check=True,
                   capture_output=True, text=True)
    submitted = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(submitted) == 1
    args = submitted[0]
    assert "--partition=gpu,gpux" in args
    assert "--time=2-00:00:00" in args
    assert "--requeue" in args
    assert "--dependency=afterok:123" in args
    assert [p.read_text() for p in (output / "queue").glob("*.jobs")] == ["12345\n"]
    record = json.loads(next((output / "queue").glob("*.json")).read_text())
    assert record["partition_limits"] == {"gpu": "4-04:00:00", "gpux": "2-00:00:00"}
    if prefix in ("STAGE2", "STAGE2_MLP"):
        exported = next(arg for arg in args if arg.startswith("--export="))
        assert prefix + "_RESUME=" + str(checkpoints / "latest.ckpt") in exported
