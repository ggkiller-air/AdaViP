#!/usr/bin/env python3
"""Validate and launch one upstream ManiFeel Power Plug DP baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs" / "manifeel"
METHOD_TASKS = {
    "vision": "vision_wrist",
    "tacrgb": "vistac_wrist",
    "tacff": "visff_wrist",
}


def load_config(method: str):
    """Load a method profile and reject mismatched task or dataset settings."""
    config = OmegaConf.load(CONFIG_ROOT / f"power_plug_{method}.yaml")
    if config.task != METHOD_TASKS[method]:
        raise ValueError(f"Unexpected task for {method}: {config.task}")
    if config.isaacgym_cfg_name != "isaacgym_config_power_plug.yaml":
        raise ValueError("Power Plug must use its own Isaac Gym task config")
    dataset = Path(config.dataset_path)
    if not dataset.is_dir():
        raise FileNotFoundError(f"ManiFeel dataset missing: {dataset}")
    for key in (
        "num_demos", "num_epochs", "seed", "checkpoint_every", "val_every",
        "batch_size", "num_workers", "val_batch_size", "val_num_workers",
    ):
        if int(config[key]) < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}: {config[key]}")
    if int(config.rollout_every) < 0:
        raise ValueError("rollout_every cannot be negative")
    return config


def main() -> int:
    """Launch the shared single-task entrypoint with one isolated run name."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHOD_TASKS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.method)
    env = os.environ.copy()
    for key, value in {
        "TASK_CONFIG": cfg.task,
        "DATASET_PATH": cfg.dataset_path,
        "ISAACGYM_CONFIG": cfg.isaacgym_cfg_name,
        "RUN_NAME": cfg.run_name,
        "NUM_DEMOS": cfg.num_demos,
        "NUM_EPOCHS": cfg.num_epochs,
        "SEED": cfg.seed,
        "ROLLOUT_EVERY": cfg.rollout_every,
        "CHECKPOINT_EVERY": cfg.checkpoint_every,
        "VAL_EVERY": cfg.val_every,
    }.items():
        env[f"MANIFEEL_{key}"] = str(value)
    env.setdefault("MANIFEEL_WANDB_MODE", "offline")
    env.setdefault("MANIFEEL_WANDB_PROJECT", "manifeel_power_plug_baselines")

    command = [
        "bash", str(REPO_ROOT / "scripts/manifeel/train_single_task_dp.sh"),
        "_target_=adavip.manifeel.retained_dp_workspace.RetainedDiffusionUnetImageWorkspace",
        f"dataloader.batch_size={cfg.batch_size}",
        f"dataloader.num_workers={cfg.num_workers}",
        "+dataloader.prefetch_factor=1",
        "dataloader.persistent_workers=true",
        f"val_dataloader.batch_size={cfg.val_batch_size}",
        f"val_dataloader.num_workers={cfg.val_num_workers}",
        "+val_dataloader.prefetch_factor=1",
        "+checkpoint.save_best_val_ckpt=true",
    ]
    if args.dry_run:
        print(f"method={args.method}")
        names = (
            "TASK_CONFIG", "DATASET_PATH", "ISAACGYM_CONFIG", "RUN_NAME",
            "NUM_DEMOS", "NUM_EPOCHS", "SEED", "ROLLOUT_EVERY",
            "CHECKPOINT_EVERY", "VAL_EVERY", "WANDB_MODE",
        )
        for key in sorted(f"MANIFEEL_{name}" for name in names):
            if key in env:
                print(f"{key}={env[key]}")
        print("command=" + " ".join(command))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
