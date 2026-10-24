"""Profile a frozen Stage 1 v2 teacher on one independent H200."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS
from adavip.manifeel.full_sensor_workspace import optimizer_to
from adavip.manifeel.stage1_v2_training import BalancedTaskSampler, capture_rng, restore_rng
from diffusion_policy.common.pytorch_util import dict_apply


def write_json(path: Path, value: dict) -> None:
    """Write a diagnostic artifact atomically outside the production directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def digest_array(values: np.ndarray) -> str:
    """Hash contiguous float32 fixture values without converting precision."""
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def required_rows(dataset, indices: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Collect exact causal/future image rows and history/action rows for a fixture."""
    images = [set() for _ in dataset.task_specs]
    actions = [set() for _ in dataset.task_specs]
    for index in indices:
        task, episode, start, end, offset = dataset._resolve_index(index)
        anchor = start + 8 + offset
        if anchor - 8 < start or anchor + 14 >= end:
            raise ValueError("Fixture crosses an episode boundary")
        images[task].update(range(anchor - 5, anchor + 5))
        actions[task].update(range(anchor - 8, anchor + 15))
    return ([np.asarray(sorted(rows), dtype=np.int64) for rows in images],
            [np.asarray(sorted(rows), dtype=np.int64) for rows in actions])


def prepare_fixture(run_dir: Path, root: Path, cache_dir: Path, sample_count: int = 32) -> None:
    """Prepare a bounded real-data RAM fixture on CPU before requesting a GPU."""
    if sample_count < 32 or sample_count % 32:
        raise ValueError("Fixture size must be a positive multiple of batch32")
    if root.resolve() == run_dir.resolve() or (root / "fixture.json").exists():
        raise FileExistsError("Use a new, separate profiling directory")
    root.mkdir(parents=True, exist_ok=True)
    os.link(run_dir / "checkpoints/latest.ckpt", root / "reference.ckpt")
    payload = torch.load(root / "reference.ckpt", map_location="cpu", mmap=True, weights_only=False)
    cfg = payload["cfg"]
    dataset = hydra.utils.instantiate(cfg.dataset, decoded_cache_dir=None)
    contract = json.loads((cache_dir / "contract.json").read_text())
    cache = json.loads((cache_dir / "manifest.json").read_text())
    if dataset.manifest["shape_meta"] != contract["manifest"]["shape_meta"]:
        raise ValueError("Fixture cache changes observation shapes")
    indices = list(itertools.islice(BalancedTaskSampler(dataset, int(cfg.training.seed)), sample_count))
    image_rows, action_rows = required_rows(dataset, indices)
    records = []
    for record in cache["arrays"]:
        task, key = record["task_index"], record["key"]
        original = contract["manifest"]["stores"][task]
        current = dataset.manifest["stores"][task]
        for name in ("task", "path", "arrays", "episode_ends"):
            if current[name] != original[name]:
                raise ValueError(f"Fixture changes cached source: {name}")
        if sorted(original["train_episodes"] + original["val_episodes"]) != list(
                range(len(original["episode_ends"]))):
            raise ValueError("Fixture source cache does not cover every demonstration")
        rows = image_rows[task] if key in IMAGE_KEYS else action_rows[task]
        source = np.load(cache_dir / record["filename"], mmap_mode="r", allow_pickle=False)
        if list(source.shape) != record["shape"] or source.shape[0] != current["episode_ends"][-1]:
            raise ValueError("Fixture requires the existing full-coverage decoded cache")
        values = np.array(source[rows], dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite fixture")
        if key in IMAGE_KEYS and (values.min() < 0 or values.max() > 1):
            raise ValueError("Image fixture is outside [0,1]")
        destination = root / "arrays" / record["filename"]
        destination.parent.mkdir(exist_ok=True)
        np.save(destination, values, allow_pickle=False)
        records.append({"task": task, "key": key, "filename": record["filename"],
                        "rows": rows.tolist(), "sha256": digest_array(values),
                        "source_array_sha256": record["sha256"], "nbytes": values.nbytes})
        print(json.dumps({"event": "fixture_array", "task": task, "key": key,
                          "nbytes": values.nbytes}), flush=True)
    OmegaConf.save(cfg, root / "reference_config.yaml")
    task_counts = np.bincount([dataset._resolve_index(index)[0] for index in indices], minlength=9)
    write_json(root / "fixture.json", {
        "indices": indices, "task_counts": task_counts.tolist(), "arrays": records,
        "nbytes": sum(record["nbytes"] for record in records), "production_run": str(run_dir),
        "reference_epoch": payload["epoch"], "reference_step": payload["global_step"],
        "provenance": payload["provenance"], "cache_contract_sha256": cache["contract"],
        "scope": f"{sample_count} fixed real windows covering nine tasks; not a full-data throughput benchmark",
        "validation": "Original contract/source metadata and selected float32 values; no full-cache rehash",
    })


def load_fixture(root: Path, cfg) -> tuple:
    """Load every selected array into anonymous RAM before creating workers."""
    fixture = json.loads((root / "fixture.json").read_text())
    dataset = hydra.utils.instantiate(cfg.dataset, decoded_cache_dir=None)
    groups = [{"data": {}} for _ in dataset.task_specs]
    for record in fixture["arrays"]:
        values = np.load(root / "arrays" / record["filename"], allow_pickle=False)
        if digest_array(values) != record["sha256"]:
            raise ValueError("Corrupt profiling fixture")
        task = record["task"]
        groups[task]["data"][record["key"]] = SourceIndexedArray(
            values, np.asarray(record["rows"], dtype=np.int64), int(dataset.episode_ends[task][-1]),
        )
    dataset.groups = groups
    return dataset, fixture


class RepeatedFixtureBatches:
    """Repeat one bounded real-data batch solely for controlled profiling."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __iter__(self):
        while True:
            yield list(self.indices)


def batch_nbytes(batch: dict) -> int:
    """Count all CPU-to-GPU tensor bytes without touching their values."""
    return sum(batch_nbytes(value) if isinstance(value, dict) else value.numel() * value.element_size()
               for value in batch.values())


def summarize(values: list[float]) -> dict:
    """Summarize directly observed timings without assuming normality."""
    return {"mean": statistics.mean(values), "p50": float(np.quantile(values, 0.5)),
            "p95": float(np.quantile(values, 0.95)), "min": min(values), "max": max(values)}


def optimized_read(array: SourceIndexedArray, indices: np.ndarray) -> np.ndarray:
    """Mirror the existing verified-cache fast path for a CPU-only comparison."""
    mapped = array.row_map[indices]
    if np.any(mapped < 0):
        raise IndexError("Missing fixture row")
    if np.all(np.diff(mapped) == 1):
        values = array.values[int(mapped[0]):int(mapped[-1]) + 1]
    else:
        values = array.values[mapped]
    return np.ascontiguousarray(np.moveaxis(values, -1, 1))


def cpu_read_probe(dataset, indices: list[int]) -> dict:
    """Compare verified RAM image-read paths with bitwise equality checks."""
    timings = defaultdict(list)
    for index in indices:
        task, episode, start, end, offset = dataset._resolve_index(index)
        anchor = start + 8 + offset
        for length, first in ((6, anchor - 5), (4, anchor + 1)):
            rows = np.arange(first, first + length)
            for key in IMAGE_KEYS:
                array = dataset.groups[task]["data"][key]
                started = time.perf_counter()
                original = dataset._read_images(array, rows, key)
                timings["original_seconds"].append(time.perf_counter() - started)
                started = time.perf_counter()
                optimized = optimized_read(array, rows)
                timings["optimized_seconds"].append(time.perf_counter() - started)
                if not np.array_equal(original, optimized):
                    raise AssertionError("Fast read changes decoded values")
    return {name: summarize(values) for name, values in timings.items()}


class ModuleTimings:
    """Collect non-overlapping forward module CUDA-event intervals and trace ranges."""

    def __init__(self, model) -> None:
        self.events = defaultdict(list)
        self.handles = []
        modules = [("online_resnet", value) for value in model.image_encoders.values()]
        modules += [("temporal", value) for value in model.image_temporal.values()]
        modules += [("temporal", model.proprio_temporal), ("diffusion_unet", model.model)]
        modules += [("future_ema_vision" if key in IMAGE_KEYS[:3] else "future_ema_tactile", value)
                    for key, value in model.target_image_encoders.items()]
        for name, module in modules:
            def before(current, inputs, label=name):
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                scope = torch.profiler.record_function(f"stage1/{label}")
                scope.__enter__()
                begin.record()
                current._profile_interval = (begin, finish, scope)

            def after(current, inputs, output, label=name):
                begin, finish, scope = current._profile_interval
                finish.record()
                scope.__exit__(None, None, None)
                self.events[label].append((begin, finish))

            self.handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]

    def reset(self) -> None:
        """Clear completed intervals between steps."""
        self.events.clear()

    def seconds(self) -> dict:
        """Read elapsed device timeline intervals after the caller synchronizes."""
        return {name: sum(begin.elapsed_time(finish) / 1000 for begin, finish in pairs)
                for name, pairs in self.events.items()}

    def close(self) -> None:
        """Remove diagnostic hooks before uninstrumented timing."""
        for handle in self.handles:
            handle.remove()


def precision_context(mode: str):
    """Limit BF16 autocast to an isolated forward pass with FP32 master weights."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mode == "bf16" \
        else contextlib.nullcontext()


def gradient_comparison(reference: list, parameters: list) -> dict:
    """Compare all available gradient elements on device with a single result transfer."""
    products, expected_squares, actual_squares = [], [], []
    for expected, parameter in zip(reference, parameters):
        actual = parameter.grad
        if (expected is None) != (actual is None):
            raise AssertionError("Precision modes change gradient connectivity")
        if expected is not None:
            products.append((expected * actual).sum())
            expected_squares.append(expected.square().sum())
            actual_squares.append(actual.square().sum())
    product, expected_norm, actual_norm = torch.stack([
        torch.stack(products).sum(), torch.stack(expected_squares).sum().sqrt(),
        torch.stack(actual_squares).sum().sqrt(),
    ]).tolist()
    return {"cosine": product / max(expected_norm * actual_norm, 1e-30),
            "norm_ratio": actual_norm / max(expected_norm, 1e-30)}


def run_profile(root: Path, output: Path, steps: int, warmup: int, workers: int) -> None:
    """Measure isolated single-H200 compute, real RAM loading, components and BF16."""
    output.mkdir(parents=True, exist_ok=False)
    payload = torch.load(root / "reference.ckpt", map_location="cpu", mmap=True, weights_only=False)
    cfg = payload["cfg"]
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset, fixture = load_fixture(root, cfg)
    cpu_reads = cpu_read_probe(dataset, fixture["indices"])
    write_json(output / "cpu_reads.json", cpu_reads)
    if torch.cuda.is_initialized():
        raise RuntimeError("DataLoader workers must fork before CUDA initialization")
    loader = DataLoader(dataset, batch_sampler=RepeatedFixtureBatches(fixture["indices"]),
                        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                        **({"prefetch_factor": 2} if workers else {}))
    iterator = iter(loader)
    device = torch.device("cuda:0")
    if "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Profiling requires an independent H200")
    cpu_batch = next(iterator)
    if not cpu_batch["obs"]["front"].is_pinned():
        raise AssertionError("CPU input is not pinned")
    resident = dict_apply(cpu_batch, lambda value: value.to(device, non_blocking=True))
    model = hydra.utils.instantiate(cfg.policy)
    model.normalizer.load_state_dict(payload["normalizer"], strict=True)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)

    def reset() -> None:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        optimizer_to(optimizer, device)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(42)

    def forward(batch, mode):
        with precision_context(mode):
            losses = model.compute_loss_with_metrics(batch)
            total = losses["action_loss"] + payload["lambda_vision"] * losses["future_vision_loss"] \
                + payload["lambda_tactile"] * losses["future_tactile_loss"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite {mode} loss")
        return losses, total

    def update(batch, mode):
        optimizer.zero_grad(set_to_none=True)
        losses, total = forward(batch, mode)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.update_target_encoder()
        metrics = {name: float(losses[name].detach()) for name in (
            "action_loss", "future_vision_loss", "future_tactile_loss",
            "vision_latent_variance", "tactile_latent_variance",
        )}
        return metrics, float(norm)

    environment = {
        "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "master_parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "batch_bytes": batch_nbytes(cpu_batch), "workers": workers, "parent_cpu_threads": torch.get_num_threads(),
        "batch_size": 32, "fixture": fixture, "cpu_reads": cpu_reads,
    }
    write_json(output / "environment.json", environment)
    reports = []
    try:
        for mode in ("fp32", "bf16"):
            if mode == "bf16" and not environment["bf16_supported"]:
                raise RuntimeError("H200 BF16 support check failed")
            for source in ("resident", "ram_loader"):
                reset()
                for _ in range(warmup):
                    batch = resident if source == "resident" else dict_apply(
                        next(iterator), lambda value: value.to(device, non_blocking=True),
                    )
                    update(batch, mode)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                waits, enqueue_times = [], []
                started_timestamp, started = time.time(), time.perf_counter()
                for _ in range(steps):
                    wait_started = time.perf_counter()
                    host = next(iterator) if source == "ram_loader" else None
                    waits.append(time.perf_counter() - wait_started)
                    transfer_started = time.perf_counter()
                    batch = resident if host is None else dict_apply(
                        host, lambda value: value.to(device, non_blocking=True),
                    )
                    enqueue_times.append(time.perf_counter() - transfer_started)
                    metrics, norm = update(batch, mode)
                torch.cuda.synchronize()
                elapsed, finished_timestamp = time.perf_counter() - started, time.time()
                report = {"precision": mode, "source": source, "steps": steps, "warmup_steps": warmup,
                          "started_timestamp": started_timestamp, "finished_timestamp": finished_timestamp,
                          "seconds": elapsed, "samples_per_second": steps * 32 / elapsed,
                          "steps_per_second": steps / elapsed, "loader_wait_seconds": summarize(waits),
                          "h2d_host_enqueue_seconds": summarize(enqueue_times),
                          "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "last_losses": metrics,
                          "last_gradient_norm": norm}
                reports.append(report)
                write_json(output / f"throughput_{mode}_{source}.json", report)
                print(json.dumps({"event": "throughput", **report}), flush=True)

        reset()
        rng = capture_rng()
        reference_losses, reference_total = forward(resident, "fp32")
        reference_total.backward()
        reference = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]
        expected = {name: float(reference_losses[name].detach()) for name in (
            "action_loss", "future_vision_loss", "future_tactile_loss",
        )}
        del reference_losses, reference_total
        optimizer.zero_grad(set_to_none=True)
        restore_rng(rng)
        actual_losses, actual_total = forward(resident, "bf16")
        actual_total.backward()
        actual = {name: float(actual_losses[name].detach()) for name in expected}
        numeric = {"fp32": expected, "bf16": actual,
                   "relative_loss_difference": {name: abs(actual[name] - expected[name]) / max(abs(expected[name]), 1e-12)
                                                for name in expected},
                   "gradients": gradient_comparison(reference, parameters)}
        del reference, actual_losses, actual_total
        reset()
        stable_losses, stable_norms = [], []
        for _ in range(200):
            metrics, norm = update(resident, "bf16")
            stable_losses.append(metrics["action_loss"])
            stable_norms.append(norm)
        numeric.update({"bf16_stability_steps": 200, "finite_loss_and_gradients": True,
                        "bf16_action_loss": summarize(stable_losses), "bf16_gradient_norm": summarize(stable_norms),
                        "master_parameter_dtypes_after": sorted({str(parameter.dtype) for parameter in model.parameters()}),
                        "scope": "Short fixed-batch diagnostics only; no claim about long-run convergence"})
        write_json(output / "bf16_numeric.json", numeric)

        components = {}
        for mode in ("fp32", "bf16"):
            reset()
            for _ in range(warmup):
                update(resident, mode)
            hooks = ModuleTimings(model)
            timings = defaultdict(list)
            try:
                for _ in range(20):
                    hooks.reset()
                    wait_started = time.perf_counter()
                    host = next(iterator)
                    timings["loader_wait"].append(time.perf_counter() - wait_started)
                    intervals = {}

                    @contextlib.contextmanager
                    def interval(name):
                        begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record()
                        with torch.profiler.record_function(f"stage1/{name}"):
                            yield
                        finish.record()
                        intervals[name] = (begin, finish)

                    with interval("h2d"):
                        batch = dict_apply(host, lambda value: value.to(device, non_blocking=True))
                    optimizer.zero_grad(set_to_none=True)
                    with interval("forward"):
                        losses, total = forward(batch, mode)
                    with interval("backward"):
                        total.backward()
                    with interval("clip_adam"):
                        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                        optimizer.step()
                    with interval("ema_update"):
                        model.update_target_encoder()
                    with interval("metric_sync"):
                        for name in ("action_loss", "future_vision_loss", "future_tactile_loss",
                                     "vision_latent_variance", "tactile_latent_variance"):
                            float(losses[name].detach())
                        float(norm)
                    torch.cuda.synchronize()
                    for name, (begin, finish) in intervals.items():
                        timings[name].append(begin.elapsed_time(finish) / 1000)
                    for name, seconds in hooks.seconds().items():
                        timings[name].append(seconds)
                components[mode] = {name: summarize(values) for name, values in timings.items()}
                write_json(output / f"components_{mode}.json", components[mode])
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=True) as profile:
                    for _ in range(3):
                        hooks.reset()
                        update(resident, mode)
                profile.export_chrome_trace(str(output / f"trace_{mode}.json"))
                (output / f"operators_{mode}.txt").write_text(
                    profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=40),
                )
            finally:
                hooks.close()
        write_json(output / "report.json", {"environment": environment, "throughput": reports,
                                            "bf16_numeric": numeric, "components": components})
    finally:
        iterator._shutdown_workers()


def main() -> None:
    """Prepare fixtures on CPU or run the separately allocated H200 diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.prepare:
        if arguments.run_dir is None or arguments.cache_dir is None:
            parser.error("Preparation requires --run-dir and --cache-dir")
        torch.set_num_threads(1)
        prepare_fixture(arguments.run_dir, arguments.fixture_dir, arguments.cache_dir)
    else:
        if arguments.output_dir is None:
            parser.error("Profiling requires --output-dir")
        run_profile(arguments.fixture_dir, arguments.output_dir, arguments.steps,
                    arguments.warmup, arguments.workers)


if __name__ == "__main__":
    main()
