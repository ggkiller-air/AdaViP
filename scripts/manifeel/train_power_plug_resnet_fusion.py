#!/usr/bin/env python3
"""Launch the Power Plug TacRGB baseline with attention fusion only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPO_ROOT / "configs/manifeel/power_plug_resnet_fusion.yaml"


def main() -> int:
    """Run the upstream baseline protocol with a fusion observation encoder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(PROFILE)
    if cfg.task != "vistac_wrist" or cfg.isaacgym_cfg_name != "isaacgym_config_power_plug.yaml":
        raise ValueError("ResNet fusion requires the Power Plug TacRGB task")
    if not Path(cfg.dataset_path).is_dir():
        raise FileNotFoundError(f"Power Plug dataset missing: {cfg.dataset_path}")
    if cfg.num_epochs != 400 or cfg.checkpoint_every != 50:
        raise ValueError("ResNet fusion requires 400 epochs and 50-epoch checkpoints")
    if cfg.batch_size != 8 or cfg.val_batch_size != 8:
        raise ValueError("Power Plug paper protocol requires batch size 8")

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
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "manifeel_power_plug_resnet_fusion",
    }.items():
        env[f"MANIFEEL_{key}"] = str(value)
    command = [
        "bash", str(REPO_ROOT / "scripts/manifeel/train_single_task_dp.sh"),
        "_target_=adavip.manifeel.retained_dp_workspace.RetainedDiffusionUnetImageWorkspace",
        "policy.obs_encoder._target_=adavip.manifeel.resnet_fusion_obs_encoder.ResNetFusionObsEncoder",
        "training.freeze_encoder=false",
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
        for key in (
            "TASK_CONFIG", "DATASET_PATH", "ISAACGYM_CONFIG", "RUN_NAME",
            "NUM_DEMOS", "NUM_EPOCHS", "SEED", "ROLLOUT_EVERY",
            "CHECKPOINT_EVERY", "VAL_EVERY",
        ):
            print(f"MANIFEEL_{key}={env[f'MANIFEEL_{key}']}")
        print("command=" + " ".join(command))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
