"""Start one formal L1/L2 run directly, with no GPU preflight phase."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

import torch
from omegaconf import OmegaConf

from adavip.manifeel.language_conditioning_workspace import LanguageConditioningWorkspace
from scripts.manifeel.language_runtime import configure_profile


def main() -> None:
    """Resolve the approved common protocol and checkpoint on scheduler signals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2]
                        / "configs/manifeel/train_language_conditioning.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, True)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    OmegaConf.resolve(cfg)
    if (cfg.variant not in ("l1", "l2") or cfg.dataloader.batch_size != 64
            or cfg.training.num_epochs != 500 or cfg.optimizer.lr != 8e-5
            or cfg.training.seed != 42 or cfg.training.lr_warmup_steps != 500
            or list(cfg.training.checkpoint_epochs) != [50, 100, 200, 300, 400, 500]):
        raise ValueError("Approved protocol is L1/L2, batch64/500 epochs, LR8e-5, seed42")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "H200" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Formal training requires exactly one allocated H200")
    profile = os.environ.get("LC_RUNTIME_PROFILE", "strict")
    runtime = {"profile": profile, **configure_profile(profile),
               "job_id": os.environ.get("SLURM_JOB_ID"),
               "cpus": sorted(os.sched_getaffinity(0))}
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"runtime_profile_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(
        json.dumps(runtime, indent=2) + "\n")
    print(json.dumps({"event": "language_runtime", **runtime}), flush=True)
    workspace = LanguageConditioningWorkspace(cfg, output_dir=cfg.output_dir)

    def request_stop(signum: int, frame: object) -> None:
        workspace.request_stop(signal.Signals(signum).name)

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    workspace.run()


if __name__ == "__main__":
    main()
