"""Exercise the exact frozen GPU entrypoint through its CPU startup boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    """Validate real metadata, RAM samples, Normalizer and policy before allocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    sys.path.insert(0, str(source))
    import hydra
    import torch
    from adavip.manifeel.full_sensor_workspace import normalizer_fingerprint

    entrypoint = source / "scripts/manifeel/benchmark_stage1_ddp_async.py"
    specification = importlib.util.spec_from_file_location("stage1_frozen_startup_review", entrypoint)
    benchmark = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(benchmark)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Exact frozen entrypoint CPU startup; not CUDA/NCCL or throughput validation",
              "entrypoint_sha256": hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
              "gpu_requested": False, "passed": False}

    class StartupValidated(Exception):
        pass

    class CpuMonitor:
        def terminate(self):
            return None

        def wait(self, timeout):
            return 0

    original_popen = benchmark.subprocess.Popen

    def cpu_monitor(command, *args, **kwargs):
        if "--monitor" in command:
            return CpuMonitor()
        return original_popen(command, *args, **kwargs)

    def inspect_launch(function, args, nprocs, start_method, join):
        bound = inspect.signature(function).bind(0, *args)
        parameters = bound.arguments
        if nprocs != 4 or start_method != "fork" or not join:
            raise AssertionError("Unexpected four-rank launch protocol")
        if torch.cuda.is_initialized():
            raise AssertionError("CUDA initialized in the CPU startup parent")
        if torch.get_num_threads() != 1:
            raise AssertionError("Fork parent did not return to serial Torch")
        model = hydra.utils.instantiate(parameters["cfg"].policy)
        model.set_normalizer(parameters["normalizer"])
        model.train()
        if normalizer_fingerprint(model.normalizer) != parameters["provenance"]["normalizer_sha256"]:
            raise AssertionError("Policy construction changed the exact Normalizer")
        if any(parameter.requires_grad for parameter in model.normalizer.parameters()):
            raise AssertionError("Policy Normalizer is not frozen")
        if any(parameter.requires_grad for parameter in model.target_image_encoders.parameters()):
            raise AssertionError("EMA target is not frozen")
        optimizer = hydra.utils.instantiate(parameters["cfg"].optimizer, params=[
            parameter for parameter in model.parameters() if parameter.requires_grad
        ])
        if optimizer.param_groups[0]["lr"] != 1e-4:
            raise AssertionError("Optimizer protocol changed")
        from adavip.manifeel.stage1_benchmark_data import BalancedStreamBatches
        dataset = parameters["dataset"]
        shards = [next(iter(BalancedStreamBatches(dataset, 42, 32, rank, 4, 1)))
                  for rank in range(4)]
        expected = next(iter(BalancedStreamBatches(dataset, 42, 128, 0, 1, 1)))
        if sum(shards, []) != expected:
            raise AssertionError("Four-rank initial sample ordering changed")
        report.update({"passed": True, "parent_threads": torch.get_num_threads(),
                       "cuda_initialized": torch.cuda.is_initialized(),
                       "model_parameters": model.parameter_counts(),
                       "normalizer_sha256": normalizer_fingerprint(model.normalizer),
                       "policy_optimizer_construction": True, "ema_and_normalizer_frozen": True,
                       "global128_sampler_identity": True, "data_manifest_verified": True,
                       "cache_status": dataset.cache_status()})
        raise StartupValidated()

    command = [str(entrypoint), "--production-dir", str(arguments.production.resolve()),
               "--output-dir", str(output / "bootstrap"),
               "--block-index", str(arguments.experiment.resolve() / "block_index.json"),
               "--steps", "24", "--warmup", "8", "--cases", "copy_stream:4", "pinned:4"]
    previous_argv = sys.argv
    try:
        sys.argv = command
        with patch.object(benchmark.subprocess, "Popen", side_effect=cpu_monitor), \
                patch.object(benchmark.torch.multiprocessing, "start_processes", side_effect=inspect_launch):
            try:
                benchmark.main()
            except StartupValidated:
                pass
        if not report["passed"]:
            raise AssertionError("The exact entrypoint did not reach the rank-launch boundary")
    finally:
        sys.argv = previous_argv
        (output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
