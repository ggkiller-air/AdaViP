"""Prepare and queue five fresh ablations with the original L1 training budget."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
from omegaconf import OmegaConf

from scripts.manifeel.l1_training_protocol import validate_l1_training_protocol


def write_json(path: Path, value: dict) -> None:
    """Replace a manifest atomically without exposing incomplete JSON."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    """Audit all five configs before submitting any job; default to preparation only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--previous-future-root", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "submit.lock")):
        baseline = OmegaConf.load(args.baseline)
        OmegaConf.save(baseline, root / "original_l1_config.yaml", resolve=True)
        runs, audit = {}, {}
        for group in ("B", "C", "D", "E", "l1_dropout10"):
            future = group in "BCDE"
            filename = "train_stage1_future.yaml" if future else "train_l1_dropout.yaml"
            cfg = OmegaConf.load(source / "configs/manifeel" / filename)
            output = root / group
            cfg.output_dir, cfg.resume = str(output), None
            if future:
                cfg.experiment = group
                old_config = args.previous_future_root / group / "launch_config.yaml"
                old = OmegaConf.load(old_config)
                cfg.future.loss_weight = float(old.future.loss_weight)
                # The experimental branch is held fixed while shared L1 settings change.
                if OmegaConf.to_container(cfg.future, resolve=True) != OmegaConf.to_container(old.future, resolve=True):
                    raise ValueError(f"Future branch changed for {group}")
            audit[group] = validate_l1_training_protocol(cfg, args.baseline)
            validate_l1_training_protocol(cfg)
            output.mkdir(parents=True, exist_ok=True)
            path = output / "launch_config.yaml"
            if path.exists() and OmegaConf.to_container(OmegaConf.load(path), resolve=True) != OmegaConf.to_container(cfg, resolve=True):
                raise FileExistsError(f"Refusing to replace a different launch config: {path}")
            OmegaConf.save(cfg, path, resolve=True)
            runs[group] = {"output": str(output), "config": str(path),
                           "future_weight": float(cfg.future.loss_weight) if future else None}
        write_json(root / "protocol_audit.json", {"passed": True, "groups": audit,
                   "baseline": str(args.baseline.resolve()), "source": str(source),
                   "runtime_profile": "stage0", "fresh_initialization": True,
                   "steps_per_epoch": 1114, "total_updates_per_run": 557000,
                   "future_weights_source": str(args.previous_future_root.resolve())})
        if not args.submit:
            print(json.dumps({"prepared": runs, "audit": str(root / "protocol_audit.json")}))
            return
        manifest_path = root / "submissions.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        for group, run in runs.items():
            if group in manifest:
                print(json.dumps({"existing": group, **manifest[group]}), flush=True)
                continue
            output = Path(run["output"])
            if any((output / "checkpoints").glob("*.ckpt")) or any((output / "queue").glob("*.json")):
                raise FileExistsError(f"Fresh submission refuses existing checkpoint or queue: {output}")
            future = group in "BCDE"
            prefix = "S1F" if future else "L1X"
            script = "stage1_future.sbatch" if future else "l1_experiment.sbatch"
            name = f"s1f_{group}_k4_batch64_ep500" if future else "l1_drop10_batch64_ep500"
            settings = ["PHASE=train", f"EXPERIMENT={group}"] if future else ["MODE=train"]
            command = [sys.executable, str(source / "scripts/manifeel/submit_partition_job.py"),
                       "--prefix", prefix, "--source", str(source), "--output", str(output),
                       "--script", "slurm/manifeel/" + script, "--name", name]
            for setting in settings + [f"CONFIG={run['config']}"]:
                command.extend(["--set", setting])
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            job_id = result.stdout.strip().split()[-1]
            manifest[group] = {**run, "job_id": job_id, "name": name,
                               "submission": result.stdout.strip(), "source": str(source)}
            write_json(manifest_path, manifest)
            print(json.dumps({"submitted": group, **manifest[group]}), flush=True)


if __name__ == "__main__":
    main()
