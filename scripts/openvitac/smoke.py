#!/usr/bin/env python3
"""Bounded real-data loading, DP forward/backward, paired masks and stage checks."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from adavip.openvitac.dataset import OpenViTacDataset
from adavip.openvitac.schema import IMAGE_KEYS, make_shape_meta
from adavip.openvitac.stages import Stage1Heads, Stage2Policy, UtilityLabelStore, paired_utility_labels, utility_contract


def move(value, device):
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value.to(device) if isinstance(value, torch.Tensor) else value


def make_policy(cfg, small=False):
    cfg = copy.deepcopy(cfg)
    if small:
        cfg.down_dims = [32, 64, 128]
        cfg.diffusion_step_embed_dim = 32
    return hydra.utils.instantiate(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config", default=str(Path(__file__).parents[2] / "configs/openvitac/stage0.yaml"))
    args = parser.parse_args()
    if (args.output / "report.json").exists():
        raise FileExistsError("Use a fresh smoke output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.manual_seed(42)
    cfg = OmegaConf.load(args.config)
    start = time.monotonic()
    dataset = OpenViTacDataset(args.root)
    chosen = [int(np.flatnonzero(dataset.index[:, 0] == task)[0]) for task in range(len(dataset.tasks))]
    # Include all three instruction variants from the same benchmark task.
    block_task = next(i for i, t in enumerate(dataset.tasks) if t["spec"]["task_id"] == "insert_block_v2")
    for language in sorted({e["language_id"] for e in dataset.tasks[block_task]["episodes"]}):
        episodes = {e["episode_index"] for e in dataset.tasks[block_task]["episodes"] if e["language_id"] == language and e["split"] == "train"}
        chosen.append(int(np.flatnonzero((dataset.index[:, 0] == block_task) & np.isin(dataset.index[:, 1], list(episodes)))[0]))
    loader = DataLoader(Subset(dataset, chosen), batch_size=2, num_workers=args.workers,
                        persistent_workers=False, timeout=180 if args.workers else 0)
    coverage, languages, first, loaded, worker_affinities = set(), set(), None, [], {}
    for batch in loader:
        coverage.update(batch["task_index"].tolist())
        languages.update(batch["language_id"].tolist())
        worker_affinities[str(int(batch["loader_pid"][0]))] = batch["loader_affinity"][0].tolist()
        for i in range(len(batch["action"])):
            loaded.append({"task": int(batch["task_index"][i]), "episode": int(batch["episode_index"][i]),
                           "language": int(batch["language_id"][i]), "action_shape": list(batch["action"].shape)})
        if first is None:
            first = batch
    report = {"device": args.device, "full_unet_widths": args.full_model,
              "train_windows": len(dataset), "task_coverage": sorted(coverage),
              "language_coverage": sorted(languages), "samples": loaded,
              "obs_shapes": {key: list(value.shape) for key, value in first["obs"].items()},
              "loader_seconds": time.monotonic() - start,
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "optimizer_steps": 0}
    report["loader_worker_affinities"] = worker_affinities
    if len(coverage) != 11:
        raise RuntimeError("DataLoader failed to cover all eleven tasks")
    device = torch.device(args.device)
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
        torch.cuda.reset_peak_memory_stats(device)
    policy = make_policy(cfg.policy, small=not args.full_model).to(device).train()
    policy.set_normalizer(dataset.get_normalizer())
    batch = move(first, device)
    batch_size = len(batch["action"])
    condition = policy.obs_encoder(policy.normalizer.normalize(batch["obs"]))
    if condition.shape != (batch_size, 256):
        raise RuntimeError("Wrong observation condition dimension")
    loss = policy.compute_loss(batch, condition_ids=torch.zeros(batch_size, dtype=torch.long, device=device))
    loss.backward()
    gradients = {}
    for key, encoder in policy.obs_encoder.image_encoders.items():
        gradients[key] = float(sum(p.grad.square().sum() for p in encoder.parameters() if p.grad is not None).sqrt())
    for key, module in (("proprio", policy.obs_encoder.state_projection), ("language", policy.obs_encoder.language_projection), ("denoiser", policy.model)):
        gradients[key] = float(sum(p.grad.square().sum() for p in module.parameters() if p.grad is not None).sqrt())
    if not torch.isfinite(loss) or not all(np.isfinite(v) and v > 0 for v in gradients.values()):
        raise RuntimeError("Invalid loss/branch gradient")
    report.update(action_loss=float(loss), gradient_norms=gradients,
                  parameters=sum(p.numel() for p in policy.parameters()), forward_backward_passed=True,
                  backward_task_indices=first["task_index"].tolist())
    policy.zero_grad(set_to_none=True)
    policy.eval()
    # Two draws on one small batch only: no large-scale Utility archive.
    labels = paired_utility_labels(policy, batch, draws=2)
    contract = utility_contract(dataset.manifest, "random_teacher_smoke_only", "train_split_fitted")
    label_path = args.output / "bounded_labels.npz"
    UtilityLabelStore.save(label_path, contract, batch["window_id"].cpu().numpy(), labels)
    UtilityLabelStore.load(label_path, contract)
    report["stage05"] = {"windows": batch_size, "draws": 2, "conditions": 9, "utility_shape": list(labels["utility"].shape)}
    # Small stage heads on real tokens and real future tactile observations.
    normalized = policy.normalizer.normalize(batch["obs"])
    z_t = policy.obs_encoder(normalized).detach().requires_grad_()
    futures = [dataset.get_future(index, offsets=(4,)) for index in chosen[:batch_size]]
    heads = Stage1Heads(resolution=8).to(device)
    future_obs = {key: torch.cat([f["obs"][key] for f in futures]).to(device) for key in IMAGE_KEYS}
    future_valid = torch.cat([f["valid_mask"] for f in futures]).to(device)
    stage1 = heads(z_t, labels["utility"].detach(), future_obs, future_valid)
    (stage1["utility_loss"] + stage1["future_loss"]).backward()
    if z_t.grad is None or not torch.isfinite(z_t.grad).all():
        raise RuntimeError("Stage 1 representation gradient failed")
    report["stage1"] = {"utility_loss": float(stage1["utility_loss"]), "future_loss": float(stage1["future_loss"]), "passed": True}
    # Release the full production U-Net before testing an independent Stage-2 DP.
    encoder_state = {k: v.detach().cpu().clone() for k, v in policy.obs_encoder.state_dict().items()}
    normalizer = policy.normalizer
    del policy, heads, stage1, z_t, normalized, condition, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    stage2_base = make_policy(cfg.policy, small=True).to(device)
    stage2_base.obs_encoder.load_state_dict(encoder_state)
    stage2_base.set_normalizer(normalizer)
    stage2 = Stage2Policy(stage2_base).to(device)
    stage2.configure_phase("C")
    batch["utility"] = labels["utility"].detach()
    stage2_losses = stage2.training_losses(batch)
    stage2_losses["loss"].backward()
    if len(stage2.film_hypernets) != 4 or not torch.isfinite(stage2_losses["loss"]):
        raise RuntimeError("Stage 2 failed")
    report["stage2"] = {"hypernets": 4, "utility_dim": 8, "loss": float(stage2_losses["loss"]), "passed": True}
    if device.type == "cuda":
        report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["seconds"] = time.monotonic() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
