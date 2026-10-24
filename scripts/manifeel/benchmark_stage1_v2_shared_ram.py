"""Load verified data once into RAM, then fork CUDA/DDP benchmark workers."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from adavip.manifeel.stage1_parallel import GlobalBatchShard, LossModule
from adavip.manifeel.stage1_v2_training import BalancedTaskSampler, source_fingerprint
from diffusion_policy.common.pytorch_util import dict_apply


def benchmark_worker(rank: int, world_size: int, address: str, dataset: Any,
                     normalizer: torch.nn.Module, cfg: DictConfig, output: Path,
                     steps: int, workers: int) -> None:
    """Initialize CUDA only in children; inherited read-only arrays stay shared."""
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if world_size > 1:
        distributed.init_process_group("nccl", init_method=address, rank=rank, world_size=world_size)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(normalizer)
    model.to(device).train()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=[
        parameter for parameter in model.parameters() if parameter.requires_grad
    ])
    module = LossModule(model)
    if world_size > 1:
        module = DistributedDataParallel(module, device_ids=[rank], broadcast_buffers=False,
                                         find_unused_parameters=True, gradient_as_bucket_view=True)
    torch.manual_seed(42 + rank)
    sampler = GlobalBatchShard(BalancedTaskSampler(dataset, 42), 32, rank, world_size)
    options = {"num_workers": workers, "pin_memory": True,
               "generator": torch.Generator().manual_seed(42)}
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    iterator = iter(DataLoader(dataset, batch_sampler=sampler, **options))
    resident = dict_apply(next(iterator), lambda value: value.to(device, non_blocking=True))

    def update(batch: dict) -> None:
        optimizer.zero_grad(set_to_none=True)
        losses = module(batch)
        loss = losses["action_loss"] + 0.5168797374060089 * losses["future_vision_loss"] \
            + 0.4334887418589181 * losses["future_tactile_loss"]
        finite = torch.isfinite(loss).to(dtype=torch.int32)
        if world_size > 1:
            distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
        if not finite:
            raise FloatingPointError("Nonfinite shared-RAM benchmark loss")
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
        for _ in range(steps):
            wait_started = time.perf_counter()
            batch = dict_apply(next(iterator), lambda value: value.to(device, non_blocking=True)) \
                if mode == "cached_loader" else resident
            waits.append(time.perf_counter() - wait_started)
            update(batch)
        torch.cuda.synchronize()
        statistics = torch.tensor([time.perf_counter() - started, np.mean(waits),
                                   torch.cuda.max_memory_allocated()], device=device, dtype=torch.float64)
        if world_size > 1:
            distributed.all_reduce(statistics, op=distributed.ReduceOp.MAX)
        record = {"mode": mode, "world_size": world_size, "global_batch_size": 32,
                  "steps": steps, "seconds": float(statistics[0]),
                  "samples_per_second": steps * 32 / float(statistics[0]),
                  "max_rank_mean_wait_seconds": float(statistics[1]),
                  "peak_memory_bytes": int(statistics[2]), "cache": "fork_shared_readonly_ram",
                  "source_sha256": source_fingerprint(), "workers_per_rank": workers}
        records.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
            output.mkdir(parents=True, exist_ok=True)
            temporary = output / f"world{world_size}.partial"
            temporary.write_text(json.dumps(records, indent=2))
            temporary.replace(output / f"world{world_size}.json")
    if world_size > 1:
        distributed.destroy_process_group()


def main() -> None:
    """Keep the parent CUDA-free, sharing one verified RAM cache across all ranks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    cfg = OmegaConf.load(arguments.config)
    cfg.dataset._target_ = "adavip.manifeel.stage1_parallel_data.CachedStage1Dataset"
    cfg.dataset.decoded_cache_mode = "ram"
    dataset = hydra.utils.instantiate(cfg.dataset)
    normalizer = dataset.get_normalizer(mode=str(cfg.training.normalizer_mode))
    if torch.cuda.is_initialized():
        raise RuntimeError("Parent initialized CUDA before fork; refusing unsafe GPU launch")
    for world_size in arguments.world_sizes:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        torch.multiprocessing.start_processes(
            benchmark_worker,
            args=(world_size, f"tcp://127.0.0.1:{port}", dataset, normalizer, cfg,
                  arguments.output_dir, arguments.steps, arguments.workers),
            nprocs=world_size, start_method="fork", join=True,
        )


if __name__ == "__main__":
    main()
