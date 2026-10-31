"""Fixed-batch gradient calibration without changing the Stage-1 initial state."""

from __future__ import annotations

import math
import statistics

import torch
from torch.utils.data import DataLoader

from adavip.manifeel.language_conditioning_workspace import BalancedEpochSampler
from adavip.manifeel.stage1_future import gradient_diagnostics, stage1_losses
from diffusion_policy.common.pytorch_util import dict_apply


def calibrate_future_weight(policy: torch.nn.Module, auxiliary: torch.nn.Module,
                            dataset: object, device: torch.device, batch_size: int = 32,
                            batches: int = 8, target_ratio: float = 0.1,
                            seed: int = 42) -> dict:
    """Use the median fusion-gradient ratio; perform no optimizer or EMA updates."""
    if not auxiliary.sensor_keys:
        return {"weight": 0.0, "batches": [], "target_ratio": target_ratio}
    if batches < 1 or not 0 < target_ratio < 1:
        raise ValueError("Invalid calibration budget or gradient target")
    sampler = BalancedEpochSampler(dataset, seed + 810000, batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0,
                        generator=torch.Generator().manual_seed(seed + 820000))
    noise = torch.Generator(device=device).manual_seed(seed + 830000)
    records = []
    devices = [device.index or 0] if device.type == "cuda" else []
    mode = policy.training
    with torch.random.fork_rng(devices=devices):
        policy.train()
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            batch = dict_apply(batch, lambda tensor: tensor.to(device))
            sample = policy.prepare_denoising_sample(batch, generator=noise)
            losses = stage1_losses(policy, auxiliary, batch, sample)
            record = gradient_diagnostics(losses, policy)
            record.update(batch=index, action_loss=float(losses["action"].detach()),
                          future_loss=float(losses["future"].detach()),
                          valid_count=losses["valid_count"])
            if losses["valid_count"] and record["future_fusion_grad_norm"] > 1e-20:
                record["suggested_weight"] = (target_ratio * record["action_fusion_grad_norm"]
                                              / record["future_fusion_grad_norm"])
            records.append(record)
            del losses, sample, batch
    policy.train(mode)
    weights = [record["suggested_weight"] for record in records if "suggested_weight" in record]
    if not weights or not all(math.isfinite(value) and value > 0 for value in weights):
        raise ValueError("Calibration requires finite nonzero action and future gradients")
    weight = statistics.median(weights)
    for record in records:
        record["calibrated_ratio"] = weight * record["weighted_future_action_grad_ratio"]
    return {"weight": weight, "target_ratio": target_ratio, "batches": records,
            "shared_parameters": "obs_encoder.fusion", "optimizer_steps": 0,
            "target_ema_updates": int(auxiliary.ema_updates)}
