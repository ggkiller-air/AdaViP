#!/usr/bin/env python3
"""Launch a Power Plug TacRGB ResNet policy with HyperNet modulation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs" / "manifeel"


def load_config(method: str):
    """Validate the fixed Power Plug protocol for a fusion variant."""
    cfg = OmegaConf.load(CONFIG_ROOT / f"power_plug_hyper_resnet_{method}.yaml")
    if cfg.task != "vistac_wrist":
        raise ValueError("HyperResNet must use the upstream TacRGB task")
    if cfg.isaacgym_cfg_name != "isaacgym_config_power_plug.yaml":
        raise ValueError("HyperResNet must use the Power Plug simulator")
    if not Path(cfg.dataset_path).is_dir():
        raise FileNotFoundError(f"Power Plug dataset missing: {cfg.dataset_path}")
    if cfg.num_epochs != 400 or cfg.checkpoint_every != 50:
        raise ValueError("HyperResNet protocol requires 400 epochs and 50-epoch checkpoints")
    if cfg.batch_size != 8 or cfg.val_batch_size != 8:
        raise ValueError("Power Plug paper protocol requires batch size 8")
    if bool(cfg.use_fusion) != (method == "fusion"):
        raise ValueError("fusion variant and use_fusion setting disagree")
    return cfg


def main() -> int:
    """Pass the baseline training settings to its upstream ManiFeel launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("fusion", "no_fusion"))
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
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "manifeel_power_plug_hyper_resnet",
    }.items():
        env[f"MANIFEEL_{key}"] = str(value)

    command = [
        "bash", str(REPO_ROOT / "scripts/manifeel/train_single_task_dp.sh"),
        "_target_=adavip.manifeel.retained_dp_workspace.RetainedDiffusionUnetImageWorkspace",
        "policy.obs_encoder._target_=adavip.manifeel.hyper_resnet_obs_encoder.HyperResNetObsEncoder",
        f"+policy.obs_encoder.use_fusion={str(bool(cfg.use_fusion)).lower()}",
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
        for key in sorted(k for k in env if k.startswith("MANIFEEL_")):
            if key in {
                "MANIFEEL_TASK_CONFIG", "MANIFEEL_DATASET_PATH", "MANIFEEL_ISAACGYM_CONFIG",
                "MANIFEEL_RUN_NAME", "MANIFEEL_NUM_DEMOS", "MANIFEEL_NUM_EPOCHS",
                "MANIFEEL_SEED", "MANIFEEL_ROLLOUT_EVERY", "MANIFEEL_CHECKPOINT_EVERY",
                "MANIFEEL_VAL_EVERY",
            }:
                print(f"{key}={env[key]}")
        print("command=" + " ".join(command))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
