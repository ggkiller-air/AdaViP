"""Isolated 2000-step H200 benchmarks with DDP state and resume audits."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import threading
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

from adavip.manifeel.full_sensor_workspace import manifest_fingerprint, normalizer_fingerprint
from adavip.manifeel.stage1_benchmark_data import BalancedStreamBatches, PinnedPrefetch
from adavip.manifeel.stage1_cuda_prefetch import (
    CudaBatchPrefetch, backward_with_prefetch, transfer_on_current_stream,
)
from adavip.manifeel.stage1_parallel import LossModule
from adavip.manifeel.stage1_v2_training import source_fingerprint
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


def write_json(path: Path, value: Any) -> None:
    """Publish complete JSON records atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def state_digest(value: Any) -> str:
    """Hash nested optimizer/model/RNG state without retaining a full CPU copy."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str((array.dtype, array.shape)).encode())
            digest.update(memoryview(array).cast("B"))
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(memoryview(np.ascontiguousarray(item)).cast("B"))
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str(type(item)).encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def local_rng(device: torch.device) -> dict:
    """Read only this rank's CUDA RNG, avoiding contexts on other GPUs."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)}


def restore_local_rng(state: dict, device: torch.device) -> None:
    """Restore this rank's independent stochastic sequence."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def assert_replicated(tensors: list[torch.Tensor], world_size: int, label: str) -> float:
    """Compare every element to rank0, not just a parameter checksum sample."""
    maximum = torch.zeros((), device=tensors[0].device)
    if world_size > 1:
        with torch.no_grad():
            for value in tensors:
                reference = value.detach().clone()
                distributed.broadcast(reference, src=0)
                difference = (reference.to(torch.float32) - value.detach().to(torch.float32)).abs().max()
                maximum = torch.maximum(maximum, difference)
            distributed.all_reduce(maximum, op=distributed.ReduceOp.MAX)
    error = float(maximum)
    if error != 0:
        raise AssertionError(f"{label} diverged across ranks: {error}")
    return error


def verify_training_state(model: torch.nn.Module, world_size: int, expected_normalizer: str) -> dict:
    """Audit synchronized gradients, model/EMA replicas, and frozen normalization."""
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError("No training gradients")
    gradient_error = assert_replicated(gradients, world_size, "gradients")
    model_error = assert_replicated(list(model.state_dict().values()), world_size, "model/EMA/normalizer")
    required = ("image_encoders", "image_temporal", "proprio_frame", "proprio_temporal",
                "language_projection", "action_history", "state_tokenizer", "cross_attention",
                "model", "future_action_projection", "future_projection",
                "future_vision_head", "future_tactile_head")
    for name in required:
        if not any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                   for parameter in getattr(model, name).parameters()):
            raise AssertionError(f"No gradient for {name}")
    if any(parameter.requires_grad or parameter.grad is not None
           for parameter in model.target_image_encoders.parameters()):
        raise AssertionError("EMA encoder is not frozen")
    if any(parameter.requires_grad or parameter.grad is not None
           for parameter in model.normalizer.parameters()):
        raise AssertionError("Normalizer is not frozen")
    if normalizer_fingerprint(model.normalizer) != expected_normalizer:
        raise AssertionError("Normalizer changed")
    return {"gradient_max_rank_difference": gradient_error,
            "model_ema_normalizer_max_rank_difference": model_error,
            "all_online_module_gradients": True, "ema_frozen": True, "normalizer_unchanged": True}


def checkpoint_probe(model: torch.nn.Module, optimizer: Any, scheduler: Any, cfg: DictConfig,
                     batch: dict, update: Any, rank: int, world_size: int,
                     device: torch.device, directory: Path, cursor: int, step: int,
                     provenance: dict) -> dict:
    """Write one rank0 checkpoint and reproduce the next update bit for bit."""
    rng = local_rng(device)
    rng_states = [None] * world_size
    if world_size > 1:
        distributed.all_gather_object(rng_states, rng)
    else:
        rng_states[0] = rng
    path = directory / "checkpoint_probe.ckpt"
    if rank == 0:
        if path.exists():
            raise FileExistsError(path)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"cfg": cfg, "model": model.state_dict(),
                   "normalizer": model.normalizer.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "rank_rng": rng_states,
                   "global_step": step, "sampler": {"seed": 42, "consumed": cursor,
                   "version": "balanced_persistent_permutation_stream_v1"},
                   "world_size": world_size, "global_batch": 32 * world_size,
                   "provenance": provenance, "benchmark_only": True}
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    if world_size > 1:
        distributed.barrier()
    expected_metrics, _ = update(batch, audit_ema=True)
    expected = {"model": state_digest(model.state_dict()),
                "optimizer": state_digest(optimizer.state_dict()),
                "scheduler": state_digest(scheduler.state_dict()),
                "rng": state_digest(local_rng(device)),
                "metrics": state_digest(expected_metrics)}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["provenance"] != provenance or payload["sampler"]["consumed"] != cursor:
        raise AssertionError("Checkpoint provenance/sampler changed")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device)
    optimizer.load_state_dict(payload["optimizer"])
    optimizer_to(optimizer, device)
    scheduler.load_state_dict(payload["scheduler"])
    restore_local_rng(payload["rank_rng"][rank], device)
    del payload
    actual_metrics, _ = update(batch, audit_ema=True)
    actual = {"model": state_digest(model.state_dict()),
              "optimizer": state_digest(optimizer.state_dict()),
              "scheduler": state_digest(scheduler.state_dict()),
              "rng": state_digest(local_rng(device)), "metrics": state_digest(actual_metrics)}
    if actual != expected:
        raise AssertionError(f"Exact checkpoint replay failed on rank{rank}")
    return {"passed": True, "path": str(path), "writer_rank": 0,
            "exact_model_optimizer_scheduler_rng_metrics": True, "sampler_consumed": cursor}


def worker_init(worker_id: int) -> None:
    """Keep nested image/Torch CPU thread pools bounded."""
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def run_worker(rank: int, world_size: int, address: str, dataset: Any, normalizer: Any,
               cfg: DictConfig, output: Path, steps: int, warmup: int, workers: int,
               provenance: dict, pipeline: str = "pinned", warm_workers: int = 0,
               cpu_threads: int = 4, synchronize_steps: bool = True) -> None:
    """Benchmark complete cached-data training steps and audit outside the window."""
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before forking the benchmark rank")
    if pipeline not in ("pinned", "copy_stream"):
        raise ValueError("Unknown input pipeline")
    batches = BalancedStreamBatches(dataset, 42, 32, rank, world_size, None)
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=workers, pin_memory=False,
                        worker_init_fn=worker_init, prefetch_factor=2 if workers else None,
                        generator=torch.Generator().manual_seed(42), timeout=600 if workers else 0,
                        multiprocessing_context="fork" if workers else None)
    iterator = iter(loader)
    torch.set_num_threads(cpu_threads)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if "H200" not in torch.cuda.get_device_name(device):
        raise RuntimeError("Real H200 GPUs are required")
    if world_size > 1:
        distributed.init_process_group("nccl", init_method=address, rank=rank,
                                       world_size=world_size, timeout=datetime.timedelta(minutes=30))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(normalizer)
    model.to(device).train()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=[
        parameter for parameter in model.parameters() if parameter.requires_grad
    ])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    module = LossModule(model)
    if world_size > 1:
        module = DistributedDataParallel(module, device_ids=[rank], broadcast_buffers=False,
                                         find_unused_parameters=True, gradient_as_bucket_view=True)
    random.seed(42 + rank)
    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    prefetch = PinnedPrefetch(iterator, device)
    cuda_prefetch = CudaBatchPrefetch(prefetch, device) if pipeline == "copy_stream" else None

    def acquire_batch():
        return cuda_prefetch.acquire() if cuda_prefetch else transfer_on_current_stream(prefetch, device)
    directory = output / f"world{world_size}_global{32 * world_size}"
    directory.mkdir(parents=True, exist_ok=True)
    cfg.dataloader.batch_size = 32
    cfg.dataloader.num_workers = workers
    cfg.output_dir = str(directory)
    cfg.benchmark = {"world_size": world_size, "local_batch": 32,
                     "global_batch": 32 * world_size, "steps": steps, "warmup": warmup,
                     "lr_schedule": "constant_for_benchmark_only", "production_migration": False,
                     "pipeline": pipeline, "cpu_threads": cpu_threads,
                     "synchronize_each_step": synchronize_steps}
    lambda_vision, lambda_tactile = provenance["lambda_vision"], provenance["lambda_tactile"]

    def update(batch: dict, audit_ema: bool = False, preload_next=None) -> tuple[dict, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        losses = module(batch)
        loss = losses["action_loss"] + lambda_vision * losses["future_vision_loss"] \
            + lambda_tactile * losses["future_tactile_loss"]
        finite = torch.isfinite(loss).to(dtype=torch.int32)
        if world_size > 1:
            distributed.all_reduce(finite, op=distributed.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError("Nonfinite benchmark loss")
        backward_with_prefetch(loss, preload_next)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if audit_ema:
            key = next(iter(model.image_encoders))
            before = next(model.target_image_encoders[key].parameters()).detach().clone()
            online = next(model.image_encoders[key].parameters()).detach()
            expected = before.mul_(model.future_ema_decay).add_(online, alpha=1 - model.future_ema_decay)
        model.update_target_encoder()
        if audit_ema:
            actual = next(model.target_image_encoders[key].parameters()).detach()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        scheduler.step()
        return {name: value.detach() for name, value in losses.items()}, norm.detach()

    first_transfer = acquire_batch()
    first_cpu = first_transfer.host
    identities = first_cpu["dataset_index"].to(device)
    all_identities = [torch.empty_like(identities) for _ in range(world_size)]
    if world_size > 1:
        distributed.all_gather(all_identities, identities)
    else:
        all_identities[0] = identities
    reference = next(iter(BalancedStreamBatches(dataset, 42, 32 * world_size, 0, 1, 1)))
    if torch.cat(all_identities).cpu().tolist() != reference:
        raise AssertionError("Global DDP batch is not the original balanced sequence")
    completed_warmup = 0
    warm_thread = None
    warm_errors = []
    first_metrics = []

    def warm_cache() -> None:
        try:
            dataset.warm_cache(warm_workers)
        except BaseException as error:
            warm_errors.append(error)

    while True:
        transfer = first_transfer if completed_warmup == 0 else acquire_batch()
        warm_losses, warm_norm = update(
            transfer.batch, audit_ema=completed_warmup == warmup - 1,
            preload_next=cuda_prefetch.preload if cuda_prefetch else None,
        )
        completed_warmup += 1
        if rank == 0 and completed_warmup == 1:
            print(json.dumps({"event": "first_real_update", "timestamp": time.time(),
                              "pipeline": pipeline, "world_size": world_size}), flush=True)
            if warm_workers and hasattr(dataset, "warm_cache"):
                warm_thread = threading.Thread(target=warm_cache, daemon=True)
                warm_thread.start()
        if completed_warmup <= 3:
            first_metrics.append({**{name: float(value) for name, value in warm_losses.items()},
                                  "gradient_norm": float(warm_norm)})
        if warm_errors:
            raise RuntimeError("Background cache verification failed") from warm_errors[0]
        cache_status = dataset.cache_status() if hasattr(dataset, "cache_status") else {"complete": True}
        if rank == 0 and (completed_warmup == 1 or completed_warmup % 20 == 0):
            print(json.dumps({"event": "productive_warmup", "updates": completed_warmup,
                              "cache": cache_status}), flush=True)
        if completed_warmup >= warmup:
            ready = torch.tensor(int(cache_status["complete"]), device=device)
            if world_size > 1:
                distributed.all_reduce(ready, op=distributed.ReduceOp.MIN)
            if bool(ready):
                break
    if warm_thread:
        warm_thread.join()
    if warm_errors:
        raise RuntimeError("Background cache verification failed") from warm_errors[0]
    warmup = completed_warmup
    cfg.benchmark.warmup = warmup
    before_audit = verify_training_state(model, world_size, provenance["normalizer_sha256"])
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if world_size > 1:
        distributed.barrier()
    torch.cuda.reset_peak_memory_stats()
    waits, computes, batch_durations = [], [], []
    task_counts = np.zeros(9, dtype=np.int64)
    metrics = torch.zeros(5, device=device, dtype=torch.float64)
    metric_names = ("action_loss", "future_vision_loss", "future_tactile_loss",
                    "vision_latent_variance", "tactile_latent_variance")
    copy_events, compute_events, input_events = [], [], []
    if rank == 0:
        write_json(directory / "status.json", {"phase": "measuring", "steps": steps,
                   "warmup_steps": warmup, "timestamp": time.time(), "global_batch": 32 * world_size})
    if world_size > 1:
        distributed.barrier()
    start_timestamp = time.time()
    started = time.perf_counter()
    for step_index in range(steps):
        batch_started = time.perf_counter()
        transfer = acquire_batch()
        cpu_batch = transfer.host
        task_counts += np.bincount(cpu_batch["task_index"].numpy(), minlength=9)
        wait_finished = time.perf_counter()
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_start.record()
        losses, norm = update(
            transfer.batch,
            preload_next=cuda_prefetch.preload if cuda_prefetch and step_index + 1 < steps else None,
        )
        metrics += torch.stack([losses[name].double() for name in metric_names])
        compute_end.record()
        copy_events.append((transfer.copy_start, transfer.copy_end))
        compute_events.append((compute_start, compute_end))
        if transfer.wait_start is not None:
            input_events.append((transfer.wait_start, transfer.wait_end))
        if synchronize_steps:
            torch.cuda.synchronize()
        finished = time.perf_counter()
        waits.append(transfer.host_wait_seconds)
        computes.append(finished - wait_finished)
        batch_durations.append(finished - batch_started)
        if rank == 0 and (step_index + 1) % 100 == 0:
            print(json.dumps({"event": "stable_progress", "steps": step_index + 1,
                              "pipeline": pipeline, "timestamp": time.time()}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    end_timestamp = time.time()
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    samples = steps * 32 * world_size
    elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
    counts_tensor = torch.as_tensor(task_counts, device=device)
    if world_size > 1:
        distributed.all_reduce(elapsed_tensor, op=distributed.ReduceOp.MAX)
        distributed.all_reduce(counts_tensor)
        distributed.all_reduce(metrics)
    counts = counts_tensor.cpu().tolist()
    if sum(counts) != samples or max(counts) - min(counts) > 1:
        raise AssertionError(f"Nine-task uniform sampling failed: {counts}")
    rank_record = {"rank": rank, "gpu": torch.cuda.get_device_name(device),
                   "local_batch": 32, "mean_loader_wait_seconds": float(np.mean(waits)),
                   "p95_loader_wait_seconds": float(np.percentile(waits, 95)),
                   "mean_compute_h2d_seconds": float(np.mean(computes)),
                   "p50_step_seconds": float(np.percentile(batch_durations, 50)),
                   "p95_step_seconds": float(np.percentile(batch_durations, 95)),
                   "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                   "local_task_counts": task_counts.tolist(), "cpu_threads": torch.get_num_threads(),
                   "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "mean_h2d_cuda_ms": float(np.mean([start.elapsed_time(end) for start, end in copy_events])),
                   "mean_compute_span_cuda_ms": float(np.mean([start.elapsed_time(end) for start, end in compute_events])),
                   "mean_cuda_input_wait_ms": float(np.mean([start.elapsed_time(end) for start, end in input_events]))
                       if input_events else None,
                   "first_three_warmup_metrics": first_metrics}
    ranks = [None] * world_size
    if world_size > 1:
        distributed.all_gather_object(ranks, rank_record)
    else:
        ranks[0] = rank_record
    throughput = {"world_size": world_size, "global_batch": 32 * world_size,
                  "steps": steps, "warmup_steps": warmup, "measured_samples": samples,
                  "elapsed_seconds": float(elapsed_tensor),
                  "samples_per_second": samples / float(elapsed_tensor),
                  "steps_per_second": steps / float(elapsed_tensor),
                  "start_timestamp": start_timestamp, "end_timestamp": end_timestamp,
                  "task_counts": counts, "uniform_task_sampling": True, "ranks": ranks,
                  "mean_metrics": dict(zip(metric_names, (metrics / (steps * world_size)).cpu().tolist())),
                  "precision": "FP32", "lr": 1e-4, "lr_schedule": "constant_for_benchmark_only",
                  "provenance": provenance, "pipeline": pipeline,
                  "synchronize_each_step": synchronize_steps,
                  "cache_fully_verified_before_timing": cache_status["complete"],
                  "host_step_durations_are_enqueue_spans": not synchronize_steps}
    if rank == 0:
        write_json(directory / "throughput.json", throughput)
        print(json.dumps({"event": "throughput_complete", **throughput}), flush=True)
    after_audit = verify_training_state(model, world_size, provenance["normalizer_sha256"])
    cpu_probe = next(prefetch)
    cursor = (warmup + steps) * 32 * world_size
    expected_probe = next(iter(BalancedStreamBatches(dataset, 42, 32, rank, world_size,
                                                     1, consumed=cursor)))
    if cpu_probe["dataset_index"].tolist() != expected_probe:
        raise AssertionError("Checkpoint sampler cursor does not reproduce the next batch")
    prefetch.close()
    if cuda_prefetch:
        cuda_prefetch.close()
    probe = dict_apply(cpu_probe, lambda value: value.to(device, non_blocking=True))
    checkpoint = checkpoint_probe(model, optimizer, scheduler, cfg, probe, update, rank, world_size,
                                  device, directory, cursor,
                                  warmup + steps, provenance)
    if rank == 0:
        report = {**throughput, "passed": True, "checks": {"initial": before_audit,
                  "final": after_audit, "global_batch_identity": True, "sampler_exact_resume": True,
                  "ema_formula": True, "checkpoint": checkpoint}}
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", {"phase": "complete", "timestamp": time.time()})
    if world_size > 1:
        distributed.barrier()
        distributed.destroy_process_group()


def monitor_gpus(output: Path) -> None:
    """Collect low-frequency NVML telemetry in a separate non-CUDA process."""
    output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,"
                                     "memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=15, check=True)
            gpus = []
            for row in csv.reader(result.stdout.splitlines()):
                index, uuid, name, total, used, utilization = [value.strip() for value in row]
                gpus.append({"index": int(index), "uuid": uuid, "name": name,
                             "memory_total_mib": float(total), "memory_used_mib": float(used),
                             "utilization_percent": float(utilization)})
            record = {"timestamp": time.time(), "gpus": gpus}
        except Exception as error:
            record = {"timestamp": time.time(), "error": str(error)}
        with output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        time.sleep(5)


def build_report(output: Path, baseline: float = 39.7) -> dict:
    """Summarize completed measurements without counting checkpoint I/O or warmup."""
    reports = [json.loads(path.read_text()) for path in sorted(output.glob("world*/report.json"))]
    if not reports:
        raise RuntimeError("No successfully audited measurements exist")
    telemetry_path = output / "gpu_monitor.jsonl"
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()] \
        if telemetry_path.exists() else []
    serial = next((item["samples_per_second"] for item in reports if item["world_size"] == 1), None)
    for report in reports:
        report["speedup_vs_production_39_7"] = report["samples_per_second"] / baseline
        report["speedup_vs_matched_single_gpu"] = report["samples_per_second"] / serial if serial else None
        samples = [record for record in telemetry if report["start_timestamp"] <= record["timestamp"]
                   <= report["end_timestamp"] and "gpus" in record]
        by_gpu: dict[str, list] = {}
        for sample in samples:
            for gpu in sample["gpus"]:
                by_gpu.setdefault(gpu["uuid"], []).append(gpu)
        report["gpu_telemetry"] = [{"uuid": uuid, "index": values[0]["index"],
                "mean_utilization_percent": float(np.mean([item["utilization_percent"] for item in values])),
                "max_utilization_percent": max(item["utilization_percent"] for item in values),
                "peak_memory_used_mib": max(item["memory_used_mib"] for item in values),
                "samples": len(values)} for uuid, values in by_gpu.items()]
    result = {"reports": reports, "production_untouched": True,
              "notes": ["Global128 is a benchmark protocol change, not a production migration.",
                        "Timed window excludes initialization, full-cache verification, warmup, audits and checkpoint I/O.",
                        "39.7 samples/s is an observed production rate, not a controlled baseline.",
                        "Matched single/4GPU measurements both use lossless optimized RAM reads.",
                        "Training dynamics/convergence equivalence is not established by throughput."]}
    write_json(output / "report.json", result)
    lines = ["# Stage1 Teacher H200 DDP benchmark", "", "Production training/checkpoints remain untouched.",
             "", "| GPUs | Global batch | Steps | samples/s | steps/s | vs 39.7 | vs matched1GPU |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        speedup = report["speedup_vs_matched_single_gpu"]
        lines.append(f"| {report['world_size']} | {report['global_batch']} | {report['steps']} | "
                     f"{report['samples_per_second']:.2f} | {report['steps_per_second']:.3f} | "
                     f"{report['speedup_vs_production_39_7']:.2f}x | "
                     + (f"{speedup:.2f}x |" if speedup is not None else "N/A |"))
    lines += ["", "## Memory and input wait", "",
              "| GPUs | Rank | Peak allocated GiB | Peak reserved GiB | Loader wait mean ms | Loader wait p95 ms |",
              "|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        for rank in report.get("ranks", []):
            lines.append(f"| {report['world_size']} | {rank['rank']} | "
                         f"{rank['peak_allocated_bytes'] / 1024 ** 3:.2f} | "
                         f"{rank['peak_reserved_bytes'] / 1024 ** 3:.2f} | "
                         f"{rank['mean_loader_wait_seconds'] * 1000:.2f} | "
                         f"{rank['p95_loader_wait_seconds'] * 1000:.2f} |")
    lines += ["", "## GPU telemetry (timed window only)", "",
              "| Measurement GPUs | GPU index | Utilization mean % | Peak used MiB | Telemetry samples |",
              "|---:|---:|---:|---:|---:|"]
    for report in reports:
        for gpu in report["gpu_telemetry"]:
            lines.append(f"| {report['world_size']} | {gpu['index']} | "
                         f"{gpu['mean_utilization_percent']:.2f} | "
                         f"{gpu['peak_memory_used_mib']:.0f} | {gpu['samples']} |")
    lines += ["", "## Correctness", ""]
    for report in reports:
        lines += [f"- {report['world_size']}GPU: task draws {report.get('task_counts', [])}; "
                  "uniform within one draw. All trainable gradient replicas and full model/EMA/Normalizer "
                  "replicas agree exactly. EMA is frozen and its update formula is verified. "
                  "Rank0-only atomic checkpoint restores optimizer/scheduler, per-rank RNG and sampler; "
                  "the next update replays bit for bit."]
    lines += ["", *[f"- {note}" for note in result["notes"]]]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    """Share one sequentially verified RAM cache, then measure 1/4GPU independently."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-preflight", type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--monitor", action="store_true")
    arguments = parser.parse_args()
    if arguments.monitor:
        monitor_gpus(arguments.output_dir / "gpu_monitor.jsonl")
        return
    if arguments.steps < 1 or arguments.warmup < 1 or not arguments.reference_preflight:
        parser.error("Positive steps/warmup and a reference preflight report are required")
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("world*")):
        raise FileExistsError("Benchmark output is not empty; refusing to overwrite")
    torch.set_num_threads(1)
    cfg = OmegaConf.load(arguments.config)
    cfg.dataset._target_ = "adavip.manifeel.stage1_benchmark_data.BenchmarkDataset"
    cfg.dataset.decoded_cache_mode = "ram"
    reference = json.loads(arguments.reference_preflight.read_text())
    dataset = hydra.utils.instantiate(cfg.dataset)
    if len(dataset.task_specs) != 9 or sum(map(len, dataset.train_episodes)) != 400:
        raise AssertionError("Benchmark requires all nine tasks and 400 demonstrations")
    normalizer = dataset.get_normalizer(mode=str(cfg.training.normalizer_mode))
    provenance = {"source_sha256": source_fingerprint(),
                  "data_sha256": manifest_fingerprint(dataset.manifest),
                  "normalizer_sha256": normalizer_fingerprint(normalizer),
                  "lambda_vision": reference["lambda_vision"],
                  "lambda_tactile": reference["lambda_tactile"],
                  "world_sizes": arguments.world_sizes, "seed": 42}
    root = Path(__file__).resolve().parents[2]
    provenance["benchmark_source_hashes"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("adavip/manifeel/stage1_ddp_benchmark.py", "adavip/manifeel/stage1_benchmark_data.py",
                     "adavip/manifeel/stage1_parallel.py", "adavip/manifeel/stage1_parallel_data.py",
                     "slurm/manifeel/benchmark_stage1_v2_ddp128.sbatch")}
    for name in ("source_sha256", "data_sha256", "normalizer_sha256"):
        if reference[name] != provenance[name]:
            raise AssertionError(f"Benchmark reference mismatch: {name}")
    if torch.cuda.is_initialized():
        raise RuntimeError("Parent initialized CUDA before fork")
    write_json(output / "provenance.json", {**provenance, "task_windows": dataset.task_lengths,
               "steps": arguments.steps, "warmup": arguments.warmup, "workers_per_rank": arguments.workers,
               "config": OmegaConf.to_container(cfg, resolve=True)})
    monitor = subprocess.Popen([sys.executable, "-m", "adavip.manifeel.stage1_ddp_benchmark",
                                "--monitor", "--output-dir", str(output)])
    try:
        for world_size in arguments.world_sizes:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            torch.multiprocessing.start_processes(
                run_worker, args=(world_size, f"tcp://127.0.0.1:{port}", dataset, normalizer, cfg,
                                 output, arguments.steps, arguments.warmup, arguments.workers, provenance),
                nprocs=world_size, start_method="fork", join=True,
            )
            build_report(output)
    finally:
        monitor.terminate()
        monitor.wait(timeout=20)


if __name__ == "__main__":
    main()
