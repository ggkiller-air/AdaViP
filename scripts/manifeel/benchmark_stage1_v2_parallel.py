"""Measure real Stage 1 DDP updates without touching the production run."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from adavip.manifeel.stage1_parallel import GlobalBatchShard, LossModule
from adavip.manifeel.stage1_v2_training import BalancedTaskSampler, source_fingerprint
from diffusion_policy.common.pytorch_util import dict_apply


def main() -> None:
    """Compare resident compute and real cached-loader throughput at global batch32."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        distributed.init_process_group("nccl")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    cfg = OmegaConf.load(arguments.config)
    cfg.dataset._target_ = "adavip.manifeel.stage1_parallel_data.CachedStage1Dataset"
    dataset = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(dataset.get_normalizer(mode=str(cfg.training.normalizer_mode)))
    model.to(device).train()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=[
        parameter for parameter in model.parameters() if parameter.requires_grad
    ])
    module = LossModule(model)
    if world_size > 1:
        module = DistributedDataParallel(module, device_ids=[local_rank],
                                         broadcast_buffers=False, find_unused_parameters=True,
                                         gradient_as_bucket_view=True)
    torch.manual_seed(42 + rank)
    sampler = GlobalBatchShard(BalancedTaskSampler(dataset, 42), 32, rank, world_size)
    options = {"num_workers": arguments.workers, "pin_memory": True}
    if arguments.workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, batch_sampler=sampler, **options)
    iterator = iter(loader)
    cpu_batch = next(iterator)
    resident = dict_apply(cpu_batch, lambda value: value.to(device, non_blocking=True))

    def update(batch: dict) -> None:
        optimizer.zero_grad(set_to_none=True)
        losses = module(batch)
        loss = losses["action_loss"] + 0.5168797374060089 * losses["future_vision_loss"] \
            + 0.4334887418589181 * losses["future_tactile_loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite benchmark loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.update_target_encoder()

    records = []
    for mode in ("resident_compute", "cached_loader"):
        for _ in range(3):
            update(resident)
        if world_size > 1:
            distributed.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        waits = []
        for _ in range(arguments.steps):
            wait_started = time.perf_counter()
            if mode == "cached_loader":
                batch = dict_apply(next(iterator), lambda value: value.to(device, non_blocking=True))
            else:
                batch = resident
            waits.append(time.perf_counter() - wait_started)
            update(batch)
        torch.cuda.synchronize()
        elapsed = torch.tensor(time.perf_counter() - started, device=device)
        if world_size > 1:
            distributed.all_reduce(elapsed, op=distributed.ReduceOp.MAX)
        record = {"mode": mode, "world_size": world_size, "global_batch_size": 32,
                  "steps": arguments.steps, "seconds": float(elapsed),
                  "samples_per_second": arguments.steps * 32 / float(elapsed),
                  "mean_wait_seconds": float(np.mean(waits)),
                  "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                  "source_sha256": source_fingerprint(), "workers_per_rank": arguments.workers}
        records.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
    if rank == 0:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(records, indent=2))
    if world_size > 1:
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
