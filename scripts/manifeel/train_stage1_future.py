"""Train one independent Stage-1 future-supervision experiment."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

import torch
from omegaconf import OmegaConf

from adavip.manifeel.stage1_future_workspace import Stage1FutureWorkspace
from scripts.manifeel.language_runtime import configure_profile


def main() -> None:
    """Require the approved protocol and checkpoint safely on scheduler signals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2]
                        / "configs/manifeel/train_stage1_future.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.resolve(cfg)
    if (cfg.experiment not in ("B", "C", "D", "E") or cfg.variant != "l1"
            or cfg.dataloader.batch_size != 32 or cfg.training.num_epochs != 100
            or cfg.future.offset != 4 or cfg.optimizer.lr != 8e-5
            or list(cfg.training.checkpoint_epochs) != [20, 40, 60, 80, 100]):
        raise ValueError("Expected B--E, L1, k4, batch32, E100, LR8e-5, snapshots every20")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "H200" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Training requires exactly one allocated H200")
    profile = os.environ.get("LC_RUNTIME_PROFILE", "tf32_deterministic")
    runtime = {"profile": profile, **configure_profile(profile),
               "job_id": os.environ.get("SLURM_JOB_ID"),
               "trainer_affinity": sorted(os.sched_getaffinity(0)),
               "allocated_cpus": os.environ.get("SLURM_CPUS_PER_TASK")}
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"runtime_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(
        json.dumps(runtime, indent=2) + "\n")
    workspace = Stage1FutureWorkspace(cfg, cfg.output_dir)

    def request_stop(signum: int, frame: object) -> None:
        workspace.request_stop(signal.Signals(signum).name)

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    workspace.run()


if __name__ == "__main__":
    main()
