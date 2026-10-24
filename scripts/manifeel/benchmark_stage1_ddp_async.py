"""Independent full-data H200 DDP tests for streaming RAM and CUDA prefetch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from adavip.manifeel.full_sensor_workspace import manifest_fingerprint, normalizer_fingerprint
from adavip.manifeel.stage1_benchmark_data import BalancedStreamBatches
from adavip.manifeel.stage1_ddp_benchmark import build_report, run_worker, write_json
from adavip.manifeel.stage1_v2_training import source_fingerprint


def verify_source_scope(source: Path, production: Path) -> dict:
    """Permit only an explicitly recorded dataset change, never resume bypasses."""
    current = source / "adavip" / "manifeel"
    original = production / "source" / "adavip" / "manifeel"
    paths = sorted(current.glob("stage1_v2_*.py"))
    paths += [current / name for name in (
        "full_sensor_dataset.py", "full_sensor_schema.py", "decoded_sensor_cache.py",
        "task_protocol.py", "task_embeddings.py", "action_adapter.py",
    )]
    records = {}
    for path in paths:
        previous = original / path.name
        before = hashlib.sha256(previous.read_bytes()).hexdigest()
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after and path.name != "stage1_v2_dataset.py":
            raise ValueError(f"Non-dataset production source changed: {path.name}")
        records[path.name] = {"original_sha256": before, "experiment_sha256": after,
                              "identical": before == after}
    return {"files": records, "production_resume": False,
            "allowed_difference": "stage1_v2_dataset.py only; original sample values checked bitwise"}


def compare_sample(actual: dict, expected: dict) -> None:
    """Compare every original tensor without accepting missing or changed fields."""
    if list(actual) != list(expected):
        raise AssertionError("Sample dictionary order/keys changed")
    for key, value in expected.items():
        if isinstance(value, dict):
            compare_sample(actual[key], value)
        elif actual[key].dtype != value.dtype or not torch.equal(actual[key], value):
            raise AssertionError(f"Sample tensor changed: {key}")


def fit_reference_normalizer(dataset, mode: str, expected_hash: str,
                             cpu_threads: int = 4):
    """Match the reference reduction settings while keeping the fork parent serial."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(cpu_threads)
        normalizer = dataset.get_normalizer(mode=mode)
        actual_hash = normalizer_fingerprint(normalizer)
        if actual_hash != expected_hash:
            raise AssertionError(f"Normalizer fingerprint mismatch with {cpu_threads} CPU threads: "
                                 f"expected {expected_hash}, got {actual_hash}")
        return normalizer
    finally:
        torch.set_num_threads(previous_threads)


def verify_original_windows(dataset, production: Path, count: int = 32) -> dict:
    """Check real streaming-cache windows against the untouched frozen dataset."""
    path = production / "source" / "adavip" / "manifeel" / "stage1_v2_dataset.py"
    specification = importlib.util.spec_from_file_location("_stage1_async_frozen_dataset", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    original = copy.copy(dataset)
    original.__class__ = module.Stage1V2ManiFeelDataset
    indices = next(iter(BalancedStreamBatches(dataset, 42, count, 0, 1, 1)))
    for index in indices:
        actual = dataset[index]
        if int(actual.pop("dataset_index")) != index:
            raise AssertionError("Dataset identity changed")
        compare_sample(actual, original[index])
    return {"bitwise_windows_checked": count, "indices": indices,
            "frozen_dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize_experiment(root: Path, production_rate: float = 39.7) -> dict:
    """Keep full-data audited cases separate from preflight and old fixtures."""
    cases = []
    for path in sorted(root.glob("*/world*/report.json")):
        report = json.loads(path.read_text())
        cases.append({**report, "report_path": str(path)})
    serial = next((case for case in cases if case["world_size"] == 1
                   and case["pipeline"] == "copy_stream"), None)
    pinned = next((case for case in cases if case["world_size"] == 4
                   and case["pipeline"] == "pinned"), None)
    asynchronous = next((case for case in cases if case["world_size"] == 4
                         and case["pipeline"] == "copy_stream"), None)
    difference = None
    if pinned and asynchronous:
        differences = []
        for original_rank, async_rank in zip(pinned["ranks"], asynchronous["ranks"]):
            if original_rank["cpu_threads"] != async_rank["cpu_threads"]:
                raise AssertionError("Pipeline comparison has different CPU thread settings")
            for original_metrics, async_metrics in zip(original_rank["first_three_warmup_metrics"],
                                                       async_rank["first_three_warmup_metrics"]):
                if original_metrics.keys() != async_metrics.keys():
                    raise AssertionError("Pipeline comparison metric coverage changed")
                differences.extend(abs(original_metrics[key] - async_metrics[key]) for key in original_metrics)
        difference = max(differences)
        if difference > 1e-6:
            raise AssertionError(f"Initial losses/gradient norms changed across input pipelines: {difference}")
    for case in cases:
        case["ratio_vs_observed_production"] = case["samples_per_second"] / production_rate
        case["ratio_vs_matched_copy_stream_single"] = (
            case["samples_per_second"] / serial["samples_per_second"] if serial else None
        )
        case["ratio_vs_four_gpu_pinned"] = (
            case["samples_per_second"] / pinned["samples_per_second"] if pinned else None
        )
    result = {"cases": cases, "production_untouched": True,
              "full_data": True, "global_batch_four_gpu": 128,
              "target_samples_per_second": 120,
              "four_gpu_initial_metric_max_abs_difference": difference,
              "notes": ["No production checkpoint is resumed or overwritten.",
                        "All raw cache blocks are verified and resident before stable timing.",
                        "Cache fill overlaps real warmup updates, not artificial GPU keepalive.",
                        "39.7 samples/s is historical production throughput, not a controlled baseline.",
                        "Global128 throughput does not prove unchanged convergence."]}
    write_json(root / "summary.json", result)
    lines = ["# Independent full-data DDP asynchronous-input benchmark", "",
             "| Pipeline | GPUs | Global batch | Stable steps | samples/s | vs observed39.7 |",
             "|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        lines.append(f"| {case['pipeline']} | {case['world_size']} | {case['global_batch']} | "
                     f"{case['steps']} | {case['samples_per_second']:.3f} | "
                     f"{case['ratio_vs_observed_production']:.3f} |")
    lines += ["", *[f"- {note}" for note in result["notes"]]]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> None:
    """Warm shared RAM productively, then run independently audited 4/1GPU cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-index", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cases", nargs="+", default=["copy_stream:4", "pinned:4", "copy_stream:1"])
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.warmup, arguments.workers,
           arguments.cache_workers, arguments.cpu_threads) < 1:
        parser.error("Positive step, worker and thread counts are required")
    cases = []
    for specification in arguments.cases:
        pipeline, world = specification.split(":")
        world_size = int(world)
        if pipeline not in ("pinned", "copy_stream") or world_size not in (1, 4):
            parser.error("Cases must be pinned/copy_stream with 1 or 4 GPUs")
        cases.append((pipeline, world_size))
    if len(set(cases)) != len(cases):
        parser.error("Cases cannot be repeated within one output directory")
    root = arguments.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("*/world*")):
        raise FileExistsError("Refusing to overwrite previous benchmark cases")
    torch.set_num_threads(1)
    production = arguments.production_dir.resolve()
    source = Path(__file__).resolve().parents[2]
    scope = verify_source_scope(source, production)
    write_json(root / "source_change_contract.json", scope)
    cfg = OmegaConf.load(production / "config.yaml")
    cfg.dataset._target_ = "adavip.manifeel.stage1_streaming_cache.StreamingBenchmarkDataset"
    cfg.dataset.decoded_cache_mode = "ram"
    cfg.dataset.block_index_path = str(arguments.block_index.resolve())
    dataset = hydra.utils.instantiate(cfg.dataset)
    if len(dataset.task_specs) != 9 or sum(map(len, dataset.train_episodes)) != 400:
        raise AssertionError("All nine tasks and 400 demonstrations are required")
    reference = json.loads((production / "preflight" / "report.json").read_text())
    normalizer = fit_reference_normalizer(
        dataset, str(cfg.training.normalizer_mode), reference["normalizer_sha256"],
    )
    provenance = {"source_sha256": source_fingerprint(),
                  "original_source_sha256": reference["source_sha256"],
                  "data_sha256": manifest_fingerprint(dataset.manifest),
                  "normalizer_sha256": normalizer_fingerprint(normalizer),
                  "lambda_vision": reference["lambda_vision"],
                  "lambda_tactile": reference["lambda_tactile"],
                  "block_index_sha256": hashlib.sha256(arguments.block_index.read_bytes()).hexdigest(),
                  "source_change_contract": scope, "seed": 42, "production_resume": False,
                  "normalizer_fit_cpu_threads": 4}
    for name in ("data_sha256", "normalizer_sha256"):
        if provenance[name] != reference[name]:
            raise AssertionError(f"Original data/normalizer contract changed: {name}")
    write_json(root / "window_validation.json", verify_original_windows(dataset, production))
    write_json(root / "provenance.json", provenance)
    if torch.cuda.is_initialized():
        raise RuntimeError("The shared-RAM parent must remain CUDA-free")
    monitor = subprocess.Popen([sys.executable, "-m", "adavip.manifeel.stage1_ddp_benchmark",
                                "--monitor", "--output-dir", str(root)])
    minimum_warmup = arguments.warmup
    try:
        for pipeline, world_size in cases:
            output = root / pipeline
            output.mkdir(exist_ok=True)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            print(json.dumps({"event": "case_start", "pipeline": pipeline,
                              "world_size": world_size, "timestamp": time.time()}), flush=True)
            torch.multiprocessing.start_processes(
                run_worker, args=(world_size, f"tcp://127.0.0.1:{port}", dataset, normalizer,
                                 copy.deepcopy(cfg), output, arguments.steps, minimum_warmup,
                                 arguments.workers, provenance, pipeline, arguments.cache_workers,
                                 arguments.cpu_threads, False),
                nprocs=world_size, start_method="fork", join=True,
            )
            report_path = output / f"world{world_size}_global{world_size * 32}" / "report.json"
            report = json.loads(report_path.read_text())
            minimum_warmup = max(minimum_warmup, report["warmup_steps"])
            monitor_path = output / "gpu_monitor.jsonl"
            if not monitor_path.exists():
                monitor_path.symlink_to(root / "gpu_monitor.jsonl")
            build_report(output)
            summarize_experiment(root)
    finally:
        monitor.terminate()
        monitor.wait(timeout=20)


if __name__ == "__main__":
    main()
