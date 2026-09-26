#!/usr/bin/env python3
"""Launch the trainable ResNet + HyperNet fusion policy for a ManiFeel task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Dict

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs" / "manifeel"

TASKS: Dict[str, Dict[str, object]] = {
    "peg_insertion": {
        "dataset": "pih_quan_June06",
        "isaacgym": "isaacgym_config.yaml",
        "action_dim": 6,
    },
    "usb_insertion": {
        "dataset": "usb_quan_Aug05",
        "isaacgym": "isaacgym_config_usb.yaml",
        "action_dim": 6,
    },
    "gear_assembly": {
        "dataset": "gear_quan_Sep15",
        "isaacgym": "isaacgym_config_gear.yaml",
        "action_dim": 6,
    },
    "nut_bolt_assembly": {
        "dataset": "nutbolt_quan_July1",
        "isaacgym": "isaacgym_config_nut.yaml",
        "action_dim": 7,
    },
    "bulb_installation": {
        "dataset": "bulb_quan_Sep19",
        "isaacgym": "isaacgym_config_bulb.yaml",
        "action_dim": 7,
    },
    "peg_reorientation": {
        "dataset": "blindinsert_quan_Aug15",
        "isaacgym": "isaacgym_config_peg_reorientation.yaml",
        "action_dim": 6,
    },
    "object_search": {
        "dataset": "explore_quan_June17",
        "isaacgym": "isaacgym_config_object_search.yaml",
        "action_dim": 7,
    },
    "ball_sorting": {
        "dataset": "sorting_quan_Aug8",
        "isaacgym": "isaacgym_config_ball_sorting.yaml",
        "action_dim": 7,
    },
}


def load_task(task_id: str):
    """Build and validate the shared 301-epoch task profile."""
    try:
        task = TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"unknown task: {task_id}") from exc
    dataset_path = Path("/data/wangzihao/datasets/manifeel") / str(task["dataset"])
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset missing: {dataset_path}")
    shared = OmegaConf.load(CONFIG_ROOT / "hyper_resnet_fusion_tasks.yaml")
    return OmegaConf.merge(
        shared,
        OmegaConf.create(
            {
                "task": "vistac_wrist",
                "dataset_path": str(dataset_path),
                "isaacgym_cfg_name": task["isaacgym"],
                "action_dim": task["action_dim"],
                "run_name": f"dp_hrf_{task_id}_batch8_ep301_seed42",
            }
        ),
    )


def main() -> int:
    """Pass the fixed policy and training protocol to the upstream launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(TASKS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_task(args.task)

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
        "WANDB_PROJECT": "manifeel_hyper_resnet_fusion_tasks",
    }.items():
        env[f"MANIFEEL_{key}"] = str(value)

    command = [
        "bash",
        str(REPO_ROOT / "scripts/manifeel/train_single_task_dp.sh"),
        "_target_=adavip.manifeel.retained_dp_workspace.RetainedDiffusionUnetImageWorkspace",
        "policy.obs_encoder._target_=adavip.manifeel.hyper_resnet_obs_encoder.HyperResNetObsEncoder",
        f"+policy.obs_encoder.use_fusion={str(bool(cfg.use_fusion)).lower()}",
        f"+policy.obs_encoder.use_fusion_hypernet={str(bool(cfg.use_fusion_hypernet)).lower()}",
        f"task.shape_meta.action.shape=[{cfg.action_dim}]",
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
                "MANIFEEL_TASK_CONFIG",
                "MANIFEEL_DATASET_PATH",
                "MANIFEEL_ISAACGYM_CONFIG",
                "MANIFEEL_RUN_NAME",
                "MANIFEEL_NUM_DEMOS",
                "MANIFEEL_NUM_EPOCHS",
                "MANIFEEL_SEED",
                "MANIFEEL_ROLLOUT_EVERY",
                "MANIFEEL_CHECKPOINT_EVERY",
                "MANIFEEL_VAL_EVERY",
            }:
                print(f"{key}={env[key]}")
        print("command=" + " ".join(command))
        return 0
    return subprocess.call(command, cwd=REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
