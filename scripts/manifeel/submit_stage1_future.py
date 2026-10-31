"""Submit four fresh future runs only after all four GPU sanity checks pass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
from omegaconf import OmegaConf

from adavip.manifeel.stage1_future_workspace import future_source_fingerprint


def main() -> None:
    """Materialize independent configs and submit one combined-partition job per run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    preflight = root / "sanity"
    if not (preflight / "PASS.json").exists():
        raise RuntimeError("All four real GPU sanity checks must pass before formal submission")
    report = json.loads((preflight / "report.json").read_text())
    if (not report.get("passed") or not report.get("identical_initialization")
            or report.get("source_fingerprint") != future_source_fingerprint()):
        raise RuntimeError("Sanity evidence is incomplete or belongs to different source")
    manifest_path = root / "formal_submissions.json"
    with FileLock(str(root / "submit.lock")):
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        for experiment in "BCDE":
            if experiment in manifest:
                continue
            output = root / experiment
            output.mkdir(parents=True, exist_ok=True)
            if any((output / "checkpoints").glob("*.ckpt")):
                raise FileExistsError(f"Fresh training refuses existing checkpoints: {output}")
            cfg = OmegaConf.load(source / "configs/manifeel/train_stage1_future.yaml")
            cfg.experiment, cfg.output_dir, cfg.resume = experiment, str(output), None
            cfg.future.loss_weight = report["experiments"][experiment]["future_weight"]
            path = output / "launch_config.yaml"
            OmegaConf.save(cfg, path, resolve=True)
            calibration = report["experiments"][experiment]["calibration"]
            (output / "preflight_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
            command = [sys.executable, str(source / "scripts/manifeel/submit_partition_job.py"),
                       "--prefix", "S1F", "--source", str(source), "--output", str(output),
                       "--script", "slurm/manifeel/stage1_future.sbatch",
                       "--name", f"s1f_{experiment}_k4_batch32_ep100",
                       "--set", "PHASE=train", "--set", f"EXPERIMENT={experiment}",
                       "--set", f"CONFIG={path}"]
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            job_id = result.stdout.strip().split()[-1]
            manifest[experiment] = {"job_id": job_id, "output": str(output),
                                    "config": str(path), "weight": float(cfg.future.loss_weight),
                                    "submission": result.stdout.strip()}
            temporary = manifest_path.with_suffix(".partial")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            os.replace(temporary, manifest_path)
            print(json.dumps({"experiment": experiment, **manifest[experiment]}), flush=True)


if __name__ == "__main__":
    main()
