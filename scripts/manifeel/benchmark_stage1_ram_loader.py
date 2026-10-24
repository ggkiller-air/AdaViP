"""Compare unchanged FP32 Stage 1 training with lossless RAM loader variants."""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate

from adavip.manifeel.full_sensor_workspace import optimizer_to
from adavip.manifeel.stage1_ram_loader import RamBatchDataset, make_ram_dataloader
from diffusion_policy.common.pytorch_util import dict_apply
from profile_stage1_v2 import batch_nbytes, load_fixture, prepare_fixture, summarize, write_json


class FixtureBatches:
    """Replay the same original balanced-sampler prefix in every benchmark case."""

    def __init__(self, indices: list[int]) -> None:
        if not indices or len(indices) % 32:
            raise ValueError("Expected complete batch32 fixture windows")
        self.indices = indices

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            for offset in range(0, len(self.indices), 32):
                yield self.indices[offset:offset + 32]


def original_dataset_class(source: Path) -> tuple[type, str]:
    """Import the untouched production dataset for a genuine original-path baseline."""
    name = "_stage1_original_ram_baseline"
    path = source / "adavip/manifeel/stage1_v2_dataset.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.Stage1V2ManiFeelDataset, hashlib.sha256(path.read_bytes()).hexdigest()


def compare_tensors(actual: dict, expected: dict) -> None:
    """Require bitwise equality, exact dtypes, shapes, dictionary order and masks."""
    if list(actual) != list(expected):
        raise AssertionError("Batch dictionary keys/order changed")
    for key in expected:
        if isinstance(expected[key], dict):
            compare_tensors(actual[key], expected[key])
        elif actual[key].dtype != expected[key].dtype or not torch.equal(actual[key], expected[key]):
            raise AssertionError(f"Batch differs: {key}")


def summarize_results(
    root: Path, labels: tuple[str, ...] = ("original_a", "cached", "batched", "original_b"),
) -> dict:
    """Combine measured GPU telemetry and matched-case throughput without extrapolation."""
    records = {}
    samples = []
    for path in root.glob("gpu_*.csv"):
        with path.open() as stream:
            for row in csv.DictReader(stream, skipinitialspace=True):
                stamp = datetime.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                samples.append((stamp.timestamp(), float(row["utilization.gpu [%]"].split()[0]),
                                float(row["memory.used [MiB]"].split()[0])))
    for label in labels:
        report = json.loads((root / label / "report.json").read_text())
        selected = [sample for sample in samples
                    if report["started_timestamp"] <= sample[0] <= report["finished_timestamp"]]
        report["telemetry_samples"] = len(selected)
        if selected:
            report["gpu_util_percent"] = summarize([sample[1] for sample in selected])
            report["nvml_memory_mib"] = summarize([sample[2] for sample in selected])
        records[label] = report
    baselines = [record["samples_per_second"] for record in records.values() if record["case"] == "original"]
    if len(baselines) < 2:
        raise ValueError("Need at least two original baselines")
    baseline = sum(baselines) / len(baselines)
    drift = max(baselines) / min(baselines)
    stable = drift <= 1.1
    for record in records.values():
        record["speedup_vs_mean_original"] = record["samples_per_second"] / baseline if stable else None
        record["ratio_vs_fastest_original"] = record["samples_per_second"] / max(baselines)
    reference_record = next(record for record in records.values() if record["case"] == "original")
    reference = reference_record["first_losses"]
    differences = {
        label: max(abs(value - reference[index][key])
                   for index, metrics in enumerate(record["first_losses"])
                   for key, value in metrics.items())
        for label, record in records.items()
    }
    final_reference = reference_record["last_losses"]
    final_differences = {
        label: max(abs(value - final_reference[key]) for key, value in record["last_losses"].items())
        for label, record in records.items()
    }
    summary = {
        "cases": records, "initial_loss_and_grad_max_abs_difference": differences,
        "final_loss_and_grad_max_abs_difference": final_differences,
        "baseline_drift_ratio": drift, "baseline_stable_within_10pct": stable,
        "speedup_interpretable": stable,
        "baseline_mean_samples_per_second": baseline,
        "scope": "256-window RAM fixture only; not full-dataset production speedup",
    }
    write_json(root / "summary.json", summary)
    return summary


def run_case(root: Path, frozen: Path, case: str, label: str, steps: int, warmup: int) -> None:
    """Measure warmed single-GPU updates without checkpoint I/O or source writes."""
    output = root / label
    output.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    payload = torch.load(root / "reference.ckpt", map_location="cpu", mmap=True, weights_only=False)
    cfg = payload["cfg"]
    original_class, original_sha = original_dataset_class(frozen)
    fixture_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    fixture_cfg.dataset._target_ = (
        "_stage1_original_ram_baseline.Stage1V2ManiFeelDataset" if case == "original"
        else "adavip.manifeel.stage1_parallel_data.CachedStage1Dataset"
    )
    dataset, fixture = load_fixture(root, fixture_cfg)
    if len(fixture["indices"]) < 256:
        raise ValueError("Use at least 256 real windows for this comparison")
    if case != "original":
        original = copy.copy(dataset)
        original.__class__ = original_class
        for offset in range(0, len(fixture["indices"]), 32):
            indices = fixture["indices"][offset:offset + 32]
            expected = default_collate([original[index] for index in indices])
            actual = (RamBatchDataset(dataset).__getitems__(indices) if case == "batched"
                      else default_collate([dataset[index] for index in indices]))
            compare_tensors(actual, expected)
            del actual, expected
        del original
    write_json(output / "validation.json", {
        "case": case, "original_dataset_sha256": original_sha,
        "bitwise_windows_checked": 0 if case == "original" else len(fixture["indices"]),
        "fixture": fixture["scope"], "task_counts": fixture["task_counts"],
        "reference_step": payload["global_step"], "provenance": payload["provenance"],
    })
    if torch.cuda.is_initialized():
        raise RuntimeError("Fork workers before initializing CUDA")
    options = OmegaConf.to_container(cfg.dataloader, resolve=True)
    if options.pop("batch_size") != 32:
        raise ValueError("This benchmark must preserve batch32")
    options.update(batch_sampler=FixtureBatches(fixture["indices"]),
                   generator=torch.Generator().manual_seed(int(cfg.training.seed)))
    loader = make_ram_dataloader(dataset, **options) if case == "batched" else DataLoader(dataset, **options)
    iterator = iter(loader)
    try:
        if "H200" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This benchmark requires one independent H200")
        device = torch.device("cuda:0")
        model = hydra.utils.instantiate(cfg.policy)
        model.normalizer.load_state_dict(payload["normalizer"], strict=True)
        model.normalizer.requires_grad_(False)
        model.load_state_dict(payload["model"], strict=True)
        model.to(device).train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=parameters)
        optimizer.load_state_dict(payload["optimizer"])
        optimizer_to(optimizer, device)
        torch.manual_seed(42)
        waits, transfers, losses_at_start = [], [], []
        first_batch_bytes = None

        def update(measured: bool) -> dict:
            nonlocal first_batch_bytes
            wait_started = time.perf_counter()
            host = next(iterator)
            wait_seconds = time.perf_counter() - wait_started
            if not host["obs"]["front"].is_pinned():
                raise AssertionError("Native pin-memory path is required")
            if first_batch_bytes is None:
                first_batch_bytes = batch_nbytes(host)
            if measured:
                waits.append(wait_seconds)
                transfer_start = torch.cuda.Event(enable_timing=True)
                transfer_end = torch.cuda.Event(enable_timing=True)
                transfer_start.record()
            batch = dict_apply(host, lambda value: value.to(device, non_blocking=True))
            if measured:
                transfer_end.record()
                transfers.append((transfer_start, transfer_end))
            optimizer.zero_grad(set_to_none=True)
            losses = model.compute_loss_with_metrics(batch)
            total = losses["action_loss"] + payload["lambda_vision"] * losses["future_vision_loss"] \
                + payload["lambda_tactile"] * losses["future_tactile_loss"]
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(cfg.training.grad_clip_norm), error_if_nonfinite=True,
            )
            optimizer.step()
            model.update_target_encoder()
            metrics = {name: float(losses[name].detach()) for name in (
                "action_loss", "future_vision_loss", "future_tactile_loss",
                "vision_latent_variance", "tactile_latent_variance",
            )}
            metrics["gradient_norm"] = float(gradient_norm)
            metrics["total"] = float(total.detach())
            return metrics

        for index in range(warmup):
            metrics = update(False)
            if index < 3:
                losses_at_start.append(metrics)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started_timestamp, started = time.time(), time.perf_counter()
        for index in range(steps):
            metrics = update(True)
            if (index + 1) % 40 == 0:
                print(json.dumps({"event": "benchmark_progress", "case": label, "step": index + 1}), flush=True)
        torch.cuda.synchronize()
        elapsed, finished_timestamp = time.perf_counter() - started, time.time()
        report = {
            "case": case, "label": label, "steps": steps, "warmup_steps": warmup,
            "started_timestamp": started_timestamp, "finished_timestamp": finished_timestamp,
            "seconds": elapsed, "samples_per_second": steps * 32 / elapsed,
            "steps_per_second": steps / elapsed, "loader_wait_seconds": summarize(waits),
            "h2d_cuda_ms": summarize([start.elapsed_time(end) for start, end in transfers]),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "first_losses": losses_at_start, "last_losses": metrics,
            "batch_bytes": first_batch_bytes, "workers": options["num_workers"],
            "prefetch_factor": options["prefetch_factor"], "pin_memory": options["pin_memory"],
            "batch_size": 32, "precision": "fp32", "checkpoint_io_in_timing": False,
            "scheduler": "fixed checkpoint LR, identical for all diagnostic cases",
            "gpu": torch.cuda.get_device_name(0), "torch_threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "fixture_nbytes": fixture["nbytes"],
            "scope": fixture["scope"], "production_changes": False,
        }
        write_json(output / "report.json", report)
        print(json.dumps({"event": "ram_loader_report", **report}), flush=True)
    finally:
        iterator._shutdown_workers()


def main() -> None:
    """Prepare data on CPU or run a bounded single-H200 benchmark case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--labels", nargs="+", default=["original_a", "cached", "batched", "original_b"])
    parser.add_argument("--case", choices=("original", "cached", "batched"), default="original")
    parser.add_argument("--label")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=24)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("Steps and warmup must be positive")
    if args.summarize:
        print(json.dumps(summarize_results(args.fixture_dir, tuple(args.labels))), flush=True)
    elif args.prepare:
        cfg = OmegaConf.load(args.production_dir / "config.yaml")
        prepare_fixture(args.production_dir, args.fixture_dir, Path(cfg.dataset.decoded_cache_dir), 256)
    else:
        run_case(args.fixture_dir, args.production_dir / "source", args.case, args.label or args.case,
                 args.steps, args.warmup)


if __name__ == "__main__":
    main()
