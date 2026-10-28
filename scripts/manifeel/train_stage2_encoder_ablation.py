"""Train one approved S1--S4 strategy on its own allocated H200."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from adavip.manifeel.full_sensor_workspace import normalizer_fingerprint
from adavip.manifeel.stage2_ablation_workspace import Stage2AblationTrainer, tensor_fingerprint
from adavip.manifeel.stage2_data import (
    load_complete_ram_cache, make_stage2_dataset, validate_complete_ram_cache,
)
from adavip.manifeel.training_runtime import check_dataloader_ipc
from scripts.manifeel.train_stage2_t2 import build_policy, load_teacher


def parse_args() -> argparse.Namespace:
    """Parse the fixed training protocol and explicitly isolated artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=("s1", "s2", "s3", "s4"))
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-config", required=True)
    parser.add_argument("--dataset-root", default="/data/wangzihao/datasets/manifeel")
    parser.add_argument("--task-embeddings", default="/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=8)
    return parser.parse_args()


def source_fingerprint() -> str:
    """Bind continuation to all policy/data/training source used by the ablation."""
    root = Path(__file__).resolve().parents[2]
    names = ("adavip/manifeel/stage2_policy.py", "adavip/manifeel/stage2_data.py",
             "adavip/manifeel/stage2_ablation_workspace.py", "adavip/manifeel/full_sensor_obs_encoder.py",
             "adavip/manifeel/reference_diffusion_policy.py", "adavip/manifeel/full_sensor_dataset.py",
             "adavip/manifeel/decoded_sensor_cache.py", "adavip/manifeel/stage1_dataset.py",
             "adavip/manifeel/full_sensor_workspace.py", "scripts/manifeel/train_stage2_t2.py",
             "scripts/manifeel/train_stage2_encoder_ablation.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def main() -> None:
    """Validate inputs, initialize E500, train once, then load the entire RAM cache."""
    args = parse_args()
    if args.workers < 0 or args.cache_workers < 1:
        raise ValueError("Invalid CPU worker count")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Formal ablation requires exactly one allocated H200")
    check_dataloader_ipc()
    cv2.setNumThreads(0)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    dataset = make_stage2_dataset(args.labels, args.dataset_root, args.task_embeddings)
    if len(dataset) != 65265 or len(dataset.task_specs) != 9:
        raise ValueError("Expected 65,265 complete K64 windows across nine tasks")
    if dataset.store.manifest["teacher"]["metadata"] != {
            "epoch": 500, "global_step": 547400, "batch_in_epoch": 0, "weights": "ema"}:
        raise ValueError("Wrong Utility label Teacher")
    cache_manifest = validate_complete_ram_cache(dataset.source, args.cache, args.cache_config)
    utility = np.asarray(dataset.store.utility, dtype=np.float64)
    rms = np.maximum(np.sqrt(np.square(utility).mean(axis=0)), 1e-6)
    teacher = load_teacher(args.teacher, args.labels)
    cfg = OmegaConf.to_container(teacher.cfg, resolve=True)
    cfg["policy"]["_target_"] = "adavip.manifeel.stage2_policy.Stage2Policy"
    cfg["policy"]["encoder_strategy"] = args.strategy
    policy = build_policy(teacher, rms, args.strategy)
    # Compare the entire inherited path while the trusted Teacher is still alive.
    if tensor_fingerprint(policy.obs_encoder.state_dict()) != tensor_fingerprint(teacher.ema_model.obs_encoder.state_dict()):
        raise RuntimeError("E500 initialization differs")
    del teacher
    contract = {
        "variant": "stage2_transformer", "encoder_strategy": args.strategy,
        "teacher": str(Path(args.teacher).resolve()),
        "labels_sha256": dataset.store.manifest["sha256"],
        "cache_contract": cache_manifest["contract"], "cache_bytes": cache_manifest["nbytes"],
        "task_embedding_sha256": dataset.source.manifest["task_embedding_sha256"],
        "batch_size": 32, "epochs": 600, "windows": 65265, "seed": 42,
        "utility_weight": 0.05, "rms": rms.tolist(),
        "normalizer": normalizer_fingerprint(policy.normalizer), "source_sha256": source_fingerprint(),
        "encoder_lr": 8e-5, "resnet_lr": 1e-5 if args.strategy == "s2" else 8e-5,
        "utility_lr": 1e-4, "action_lr": 1e-4,
        "encoder_start_epoch": 2 if policy.is_dual_encoder else 7,
        "encoder_ramp_updates": 500, "warmup_updates": 500,
        "gradient_clip_norm": 1.0, "ema_decay": 0.9999,
        "checkpoint_epochs": [1, 6, 10, 20, 50, 100, 200, 300, 400, 500],
    }
    trainer = Stage2AblationTrainer(
        policy, dataset, args.output, contract, cfg,
        lambda: load_complete_ram_cache(dataset.source, args.cache, args.cache_config, args.cache_workers),
        workers=args.workers,
    )
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: setattr(trainer, "stop_requested", True))
    print(json.dumps({"event": "ablation_ready", "strategy": args.strategy,
                      "gpu": torch.cuda.get_device_name(0), "windows": len(dataset),
                      "workers": args.workers, "cache_workers": args.cache_workers,
                      "available_cpus": len(os.sched_getaffinity(0)),
                      "common_initialization": trainer.initialization["common_sha256"]}), flush=True)
    trainer.run(args.resume)


if __name__ == "__main__":
    main()
