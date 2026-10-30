"""Bounded, isolated H200 infrastructure comparisons on real Stage2-shaped batches."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import signal
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
from adavip.manifeel.stage2_data import make_stage2_dataset, validate_complete_ram_cache
from scripts.manifeel.train_stage2_t2 import (
    build_policy, load_teacher, make_optimizer_scheduler, set_stage_trainability, update_ema,
)

STOP = False


def request_stop(*_) -> None:
    """Exit after the current benchmark update, retaining completed measurements."""
    global STOP
    STOP = True


def should_stop() -> bool:
    """Observe the batch shell stop marker without signalling the srun launcher."""
    marker = os.environ.get("INFRA_STOP_FILE")
    return STOP or bool(marker and Path(marker).exists())


def bind_current_threads(cpus: list[int]) -> None:
    """Apply a test mask to every existing main-process thread."""
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            pass


def atomic_json(path: Path, value: dict) -> None:
    """Keep an independently resumable benchmark report on persistent storage."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def physical_cpus(allowed: list[int]) -> list[int]:
    """Choose one logical CPU for each physical core inside this allocation."""
    result, seen = [], set()
    for cpu in allowed:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        identity = ((root / "physical_package_id").read_text(), (root / "core_id").read_text())
        if identity not in seen:
            seen.add(identity)
            result.append(cpu)
    return result


class SamplePool(Dataset):
    """Repeat a fixed balanced real-data RAM pool without source filesystem I/O."""

    def __init__(self, samples: list[dict], steps: int) -> None:
        self.samples, self.steps = samples, steps

    def __len__(self) -> int:
        return self.steps * 32

    def __getitem__(self, index: int) -> dict:
        return self.samples[index % len(self.samples)]


class BindWorker:
    """Place DataLoader workers on the cores selected by this test case."""

    def __init__(self, cpus: list[int]) -> None:
        self.cpus = cpus

    def __call__(self, worker: int) -> None:
        torch.set_num_threads(1)
        cv2.setNumThreads(0)
        os.sched_setaffinity(0, {self.cpus[worker % len(self.cpus)]})


def move_batch(batch: dict, device: str = "cuda") -> dict:
    """Transfer the same complete batch contract used by formal training."""
    return {key: ({sensor: tensor.to(device, non_blocking=True) for sensor, tensor in value.items()}
                  if isinstance(value, dict) else value.to(device, non_blocking=True))
            for key, value in batch.items()}


def benchmark(policy: torch.nn.Module, pool: SamplePool, allowed: list[int],
              case: dict, args: argparse.Namespace) -> dict:
    """Time warm and steady updates including loss, clipping, AdamW and EMA."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    cpus = allowed
    if case.get("two_cpu"):
        siblings = Path(f"/sys/devices/system/cpu/cpu{allowed[0]}/topology/thread_siblings_list").read_text().strip()
        candidates = set()
        for group in siblings.split(","):
            ends = list(map(int, group.split("-")))
            candidates.update(range(ends[0], ends[-1] + 1))
        cpus = sorted(candidates.intersection(allowed))
    torch.set_num_threads(4)
    bind_current_threads(cpus)
    torch.backends.cuda.matmul.allow_tf32 = bool(case.get("tf32"))
    # Match the existing PyTorch 2.4 formal run: matmul TF32 off, cuDNN TF32 on.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(case.get("cudnn"))
    model = copy.deepcopy(policy).to("cuda").train()
    model.normalizer.to("cuda")
    set_stage_trainability(model, "C")
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer, scheduler = make_optimizer_scheduler(model, 2040, 600)
    if case.get("fused"):
        optimizer = torch.optim.AdamW(optimizer.param_groups, betas=(0.95, 0.999),
                                     eps=1e-8, weight_decay=1e-6, fused=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler.lr_lambdas)
    # Exercise the actual C-stage optimizer rates instead of the epoch1 encoder gate.
    scheduler.last_epoch = 130000
    for group, base_lr, schedule in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
        group["lr"] = base_lr * schedule(scheduler.last_epoch)
    if case.get("channels_last"):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.to(memory_format=torch.channels_last)
    compute = model.training_losses
    compile_started = time.perf_counter()
    if case.get("compile"):
        # Bound compilation to the shared U-Net, leaving validation and RNG explicit.
        model.model = torch.compile(model.model, mode="default")
    setup_seconds = time.perf_counter() - compile_started
    workers = case.get("workers", 8)
    options = dict(batch_size=32, num_workers=workers, pin_memory=True,
                   worker_init_fn=BindWorker(cpus if case.get("two_cpu") else physical_cpus(cpus)),
                   generator=torch.Generator().manual_seed(700042))
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="fork")
    loader = DataLoader(pool, **options)
    iterator = iter(loader)
    gpu_batch = move_batch(next(iterator)) if case.get("gpu_resident") else None
    wall_start, data_wait, losses, warmup_seconds = None, 0.0, [], 0.0
    requested = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        if should_stop():
            raise InterruptedError("Benchmark preempted; completed cases remain in report")
        if step == args.warmup:
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - requested
            wall_start = time.perf_counter()
            data_wait = 0.0
        started = time.perf_counter()
        batch = gpu_batch if gpu_batch is not None else move_batch(next(iterator))
        if step >= args.warmup:
            data_wait += time.perf_counter() - started
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(case.get("bf16"))):
            result = compute(batch, "C")
            loss = result["action_loss"] + 0.05 * result["utility_loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite benchmark loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                             1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        # Compiled U-Net wrappers alter parameter ordering; use the original module for EMA.
        if case.get("foreach_ema"):
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - 0.9999)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
        elif case.get("compile"):
            compiled = model.model
            model.model = compiled._orig_mod
            update_ema(ema, model, 0.9999)
            model.model = compiled
        else:
            update_ema(ema, model, 0.9999)
        # Match the formal trainer's per-step CPU metric synchronizations.
        if not case.get("deferred_metrics"):
            _ = result["action_loss_per_sample"].detach().double().cpu()
            _ = result["utility_loss_per_sample"].detach().double().cpu()
        if step >= args.warmup:
            losses.append(loss.detach() if case.get("deferred_metrics") else float(loss.detach()))
    torch.cuda.synchronize()
    seconds = time.perf_counter() - wall_start
    record = {"name": case["name"], "settings": case, "strategy": args.strategy,
              "batch_size": 32, "steps": args.steps, "seconds": seconds,
              "samples_per_second": args.steps * 32 / seconds,
              "data_and_h2d_seconds": data_wait, "warmup_seconds": warmup_seconds,
              "setup_seconds": setup_seconds, "first_loss": float(losses[0]), "last_loss": float(losses[-1]),
              "final_gradient_norm": float(norm), "allowed_cpus": cpus,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    if case["name"] == "allocated_cpu_workers8":
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as profile:
                for _ in range(2):
                    measured = move_batch(next(iterator))
                    optimizer.zero_grad(set_to_none=True)
                    result = model.training_losses(measured, "C")
                    (result["action_loss"] + 0.05 * result["utility_loss"]).backward()
                    optimizer.step()
                    profile.step()
            profile.export_chrome_trace(str(args.output / "cpu_gpu_trace.json"))
            (args.output / "profile_top_ops.txt").write_text(
                profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
        except Exception as error:
            record["profile_error"] = repr(error)
    del iterator, loader, optimizer, scheduler, model, ema, compute, gpu_batch
    gc.collect()
    torch.cuda.empty_cache()
    bind_current_threads(allowed)
    return record


def main() -> None:
    """Prepare a small balanced real-data pool and persist each independent comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("s2", "s3"), required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate real subset inputs on CPU without a GPU performance claim")
    parser.add_argument("--teacher", default="/data/wangzihao/outputs/manifeel/checkpoint_archive/full_sensor_20261019_epochs300_400_500/package/checkpoints/epoch500.ckpt")
    parser.add_argument("--labels", default="/public/home/wangzihao/projects/AdaViP/outputs/stage1_utility_labels_ep500_k64")
    parser.add_argument("--cache", default="/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017")
    parser.add_argument("--cache-config", default="/data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for signum in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(signum, request_stop)
    if not args.check_inputs and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("A separate single-GPU allocation is required")
    allowed = sorted(os.sched_getaffinity(0))
    if not args.check_inputs and len(allowed) < 16:
        raise RuntimeError(f"Benchmark step has insufficient CPU affinity: {allowed}")
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"cases": []}
    report.update(torch=torch.__version__, gpu="CPU input check" if args.check_inputs else torch.cuda.get_device_name(0), allowed_cpus=allowed,
                  job_id=os.environ.get("SLURM_JOB_ID"), protocol="independent infrastructure benchmark; no production checkpoints")
    atomic_json(report_path, report)
    dataset = make_stage2_dataset(args.labels, "/data/wangzihao/datasets/manifeel",
                                 "/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz")
    manifest = validate_complete_ram_cache(dataset.source, args.cache, args.cache_config,
                                           require_full_ram=False)
    groups = [{"data": {}} for _ in dataset.task_specs]
    for record in manifest["arrays"]:
        task = record["task_index"]
        values = np.load(Path(args.cache) / record["filename"], mmap_mode="r", allow_pickle=False)
        groups[task]["data"][record["key"]] = SourceIndexedArray(values, np.arange(len(values)), len(values))
    dataset.source.groups, dataset.source.verified_ram_cache = groups, True
    indices = np.concatenate([np.flatnonzero(dataset.task_indices == task)[:8] for task in range(9)])
    pool_samples = [dataset[int(index)] for index in indices]
    rms = np.maximum(np.sqrt(np.square(np.asarray(dataset.store.utility, dtype=np.float64)).mean(0)), 1e-6)
    report["pool"] = {"windows": len(indices), "task_counts": np.bincount(dataset.task_indices[indices]).tolist(),
                      "source": "small deterministic real-data RAM pool; not a full-epoch throughput measurement"}
    if args.check_inputs:
        report.update(input_check_only=True, complete=True,
                      observation_shapes={key: list(value.shape) for key, value in pool_samples[0]["obs"].items()},
                      action_shape=list(pool_samples[0]["action"].shape))
        atomic_json(report_path, report)
        print(json.dumps(report), flush=True)
        return
    teacher = load_teacher(args.teacher, args.labels)
    torch.manual_seed(42)
    policy = build_policy(teacher, rms, args.strategy)
    del teacher, dataset, groups
    cases = [{"name": "two_cpu_workers8", "two_cpu": True},
             {"name": "allocated_cpu_workers8"},
             {"name": "allocated_cpu_workers4", "workers": 4},
             {"name": "allocated_cpu_workers0", "workers": 0},
             {"name": "deferred_metrics", "deferred_metrics": True},
             {"name": "foreach_ema", "foreach_ema": True},
             {"name": "gpu_resident", "gpu_resident": True, "workers": 0},
             {"name": "tf32_cudnn", "tf32": True, "cudnn": True},
             {"name": "tf32_channels_last", "tf32": True, "cudnn": True, "channels_last": True},
             {"name": "bf16_channels_last", "tf32": True, "cudnn": True, "channels_last": True, "bf16": True},
             {"name": "bf16_fused_adamw", "tf32": True, "cudnn": True, "channels_last": True, "bf16": True, "fused": True},
             {"name": "compile_unet_bf16", "tf32": True, "cudnn": True, "channels_last": True, "bf16": True, "fused": True, "compile": True}]
    pool = SamplePool(pool_samples, args.warmup + args.steps + 2)
    completed = {x["name"] for x in report["cases"]}
    for case in cases:
        if case["name"] in completed:
            continue
        try:
            record = benchmark(policy, pool, allowed, case, args)
        except InterruptedError:
            report["interrupted"] = True
            atomic_json(report_path, report)
            return
        except Exception as error:
            record = {"name": case["name"], "error": repr(error)}
            bind_current_threads(allowed)
            gc.collect()
            torch.cuda.empty_cache()
        report["cases"].append(record)
        atomic_json(report_path, report)
        print(json.dumps(record), flush=True)
    report["complete"] = True
    atomic_json(report_path, report)


if __name__ == "__main__":
    main()
