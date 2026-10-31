"""Independent H200 runtime experiments; never change formal Stage-1 runs."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import time
import traceback

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from adavip.manifeel.language_conditioning_workspace import BalancedEpochSampler
from adavip.manifeel.stage1_future import gradient_diagnostics, stage1_losses
from adavip.manifeel.stage1_future_dataset import initialize_future_worker
from adavip.manifeel.stage1_future_workspace import Stage1FutureWorkspace, future_source_fingerprint
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from scripts.manifeel.language_runtime import configure_profile


PROFILES = {
    "baseline": {},
    "bf16": {"bf16": True},
    "channels_last": {"channels_last": True},
    "cudnn_autotune": {"autotune": True},
    "matmul_tf32": {"matmul_tf32": True},
    "fused_adamw": {"fused": True},
    "compile_unet": {"compile": True},
    "bf16_channels_last": {"bf16": True, "channels_last": True},
    "bf16_channels_last_fused_compile": {
        "bf16": True, "channels_last": True, "fused": True, "compile": True},
}


def atomic_json(path: Path, value: dict) -> None:
    """Publish only complete JSON reports."""
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def image_layout_hook(module: torch.nn.Module, inputs: tuple) -> tuple:
    """Convert only 4D image tensors to channels_last, preserving every other input."""
    return tuple(value.contiguous(memory_format=torch.channels_last)
                 if isinstance(value, torch.Tensor) and value.ndim == 4 else value for value in inputs)


def float_output_hook(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
    """Keep MSE/cosine reductions in FP32 while BF16 accelerates feature computation."""
    return output.float()


def apply_image_layout(policy: torch.nn.Module, auxiliary: torch.nn.Module) -> None:
    """Convert online/target image backbones and raw CNN decoder outputs consistently."""
    modules = list(policy.obs_encoder.image_encoders.values()) + list(auxiliary.target_encoders.values())
    if auxiliary.target_space == "raw":
        modules += [head.decoder for head in auxiliary.heads.values()]
    for module in modules:
        module.to(memory_format=torch.channels_last)
        module.register_forward_pre_hook(image_layout_hook)


def check_production_isolation(output: Path, production: Path) -> None:
    """Reject writes inside a formal experiment directory."""
    output, production = output.resolve(), production.resolve()
    if output == production or production in output.parents:
        raise ValueError("Benchmark output must be outside the formal training root")


def run_case(base: object, shared: object, output: Path, profile: str, repeat: int,
             warmup: int, steps: int) -> dict:
    """Use identical initial model/data/noise; time optimizer, EMA, H2D and loader waits."""
    flags = {key: bool(PROFILES[profile].get(key, False))
             for key in ("bf16", "channels_last", "autotune", "matmul_tf32", "fused", "compile")}
    configure_profile("tf32_deterministic")
    torch.backends.cudnn.benchmark = flags["autotune"]
    torch.backends.cuda.matmul.allow_tf32 = flags["matmul_tf32"]
    cfg = copy.deepcopy(base)
    cfg.resume, cfg.output_dir = None, str(output)
    if flags["fused"]:
        cfg.optimizer.fused = True
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    workspace = Stage1FutureWorkspace(cfg, cfg.output_dir)
    dataset = hydra.utils.instantiate(cfg.dataset)
    dataset.groups, dataset.verified_ram_cache = shared.groups, True
    normalizer = dataset.get_normalizer()
    device = torch.device("cuda:0")
    for policy in (workspace.model, workspace.ema_model):
        policy.set_normalizer(normalizer)
        policy.to(device)
        policy.normalizer.to(device)
    workspace.future.to(device)
    optimizer_to(workspace.optimizer, device)
    policy, auxiliary = workspace.model, workspace.future
    if flags["channels_last"]:
        apply_image_layout(policy, auxiliary)
        if workspace.ema_model is not None:
            for backbone in workspace.ema_model.obs_encoder.image_encoders.values():
                backbone.to(memory_format=torch.channels_last)
    # These hooks are identity conversions in the production-precision control.
    for module in (*auxiliary.heads.values(), *auxiliary.target_encoders.values()):
        module.register_forward_hook(float_output_hook)
    if flags["compile"]:
        # Replacing the forward callable preserves the module tree used by the original EMA.
        policy.model.forward = torch.compile(policy.model.forward, mode="default", dynamic=False)
    ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
    policy.train()
    auxiliary.train()
    sampler = BalancedEpochSampler(dataset, 910042 + repeat, 32)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=8,
                        pin_memory=True, persistent_workers=True, prefetch_factor=1,
                        timeout=180, worker_init_fn=initialize_future_worker,
                        generator=torch.Generator().manual_seed(920042 + repeat))
    noise = torch.Generator(device=device).manual_seed(930042 + repeat)
    iterator = None
    first_record, first_gradient = None, None
    action_losses, future_losses, total_losses = [], [], []
    data_wait, compute_seconds = 0.0, 0.0
    started = measured_start = None
    warmup_started = time.perf_counter()
    parameters = workspace.trainable_parameters()
    try:
        iterator = iter(loader)
        torch.cuda.reset_peak_memory_stats()
        for index in range(warmup + steps):
            if index == warmup:
                torch.cuda.synchronize()
                warmup_seconds = time.perf_counter() - warmup_started
                measured_start = time.perf_counter()
            requested = time.perf_counter()
            batch = next(iterator)
            waiting = time.perf_counter() - requested
            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
            workspace.optimizer.zero_grad(set_to_none=True)
            for group in workspace.optimizer.param_groups:
                group["lr"] = cfg.optimizer.lr * workspace._lr_multiplier(
                    math.ceil(len(dataset) / 32) * 100)
            sample = policy.prepare_denoising_sample(batch, generator=noise)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=flags["bf16"]):
                terms = stage1_losses(policy, auxiliary, batch, sample)
                total = terms["action"] + float(cfg.future.loss_weight) * terms["future"]
            if index == 0:
                first_record = gradient_diagnostics(terms, policy, float(cfg.future.loss_weight),
                                                     include_resnets=True)
                gradients = torch.autograd.grad(total, list(policy.obs_encoder.fusion.parameters()),
                                                 retain_graph=True)
                first_gradient = torch.cat([g.detach().flatten().float().cpu() for g in gradients])
                first_record.update(action_loss=float(terms["action"].detach()),
                                    future_loss=float(terms["future"].detach()),
                                    total_loss=float(total.detach()),
                                    action_loss_dtype=str(terms["action"].dtype),
                                    future_loss_dtype=str(terms["future"].dtype))
                assert all(p.grad is None for p in auxiliary.target_encoders.parameters())
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite benchmark objective")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            workspace.optimizer.step()
            auxiliary.update_targets(policy.obs_encoder)
            ema.step(policy)
            workspace.global_step += 1
            # Match production host scalar reads; include their synchronization cost in timing.
            loss_values = (float(terms["action"].detach()), float(terms["future"].detach()),
                           float(total.detach()))
            if index >= warmup:
                action_losses.append(loss_values[0])
                future_losses.append(loss_values[1])
                total_losses.append(loss_values[2])
                data_wait += waiting
                compute_seconds += time.perf_counter() - requested - waiting
            del batch, sample, terms, total
        torch.cuda.synchronize()
        seconds = time.perf_counter() - measured_start
        workers = [{"pid": worker.pid, "affinity": sorted(os.sched_getaffinity(worker.pid))}
                   for worker in iterator._workers]
        torch.save(first_gradient, output / "first_fusion_gradient.pt")
        record = {"profile": profile, "repeat": repeat, "experiment": cfg.experiment,
                  "flags": flags, "batch_size": 32, "warmup_updates": warmup, "updates": steps,
                  "initialization_sha256": workspace.policy_initialization,
                  "future_weight": float(cfg.future.loss_weight),
                  "warmup_seconds_including_compile": warmup_seconds,
                  "seconds": seconds, "samples_per_second": steps * 32 / seconds,
                  "data_wait_seconds": data_wait, "data_wait_fraction": data_wait / seconds,
                  "mean_action_loss": statistics.mean(action_losses),
                  "mean_future_loss": statistics.mean(future_losses),
                  "mean_total_loss": statistics.mean(total_losses),
                  "first_step": first_record, "last_gradient_norm": float(norm),
                  "target_ema_updates": int(auxiliary.ema_updates),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                  "workers": workers, "passed_finite_checks": True,
                  "parameter_precision": "FP32", "loss_reduction_precision": "FP32"}
        atomic_json(output / "result.json", record)
        return record
    finally:
        if iterator is not None:
            iterator._shutdown_workers()
        del iterator, loader, workspace, policy, auxiliary, ema, parameters, normalizer
        gc.collect()
        if flags["compile"]:
            torch._dynamo.reset()
        torch.cuda.empty_cache()


def summarize(cases: list[dict]) -> dict:
    """Compare repeated profiles with their own experiment's baseline."""
    summary = {}
    for experiment in ("B", "E"):
        groups = {}
        for case in cases:
            if case["experiment"] == experiment and case.get("passed_finite_checks"):
                groups.setdefault(case["profile"], []).append(case)
        if not groups.get("baseline"):
            continue
        baseline = statistics.median(row["samples_per_second"] for row in groups["baseline"])
        summary[experiment] = {name: {
            "repeats": len(rows), "median_samples_per_second": statistics.median(
                row["samples_per_second"] for row in rows),
            "samples_per_second": [row["samples_per_second"] for row in rows],
            "speedup": statistics.median(row["samples_per_second"] for row in rows) / baseline,
            "median_warmup_seconds": statistics.median(row["warmup_seconds_including_compile"] for row in rows),
        } for name, rows in groups.items()}
    return summary


def main() -> None:
    """Load RAM once, run randomized repeat cases and retain errors for experience."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    check_production_isolation(args.output, args.production_root)
    if args.warmup < 1 or args.steps < 1 or args.repeats < 1:
        raise ValueError("Benchmark budgets must be positive")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "H200" not in torch.cuda.get_device_name(0)):
        raise RuntimeError("Requires an independent single-H200 allocation")
    if len(os.sched_getaffinity(0)) < 32:
        raise RuntimeError("Expected all32allocated CPUs in the srun step")
    args.output.mkdir(parents=True, exist_ok=True)
    stop_file = args.output / "STOP_REQUEST"

    def request_stop(signum: int, frame: object) -> None:
        stop_file.write_text(signal.Signals(signum).name + "\n")

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    os.environ["STAGE1_FUTURE_AFFINITY_DIR"] = str(args.output / "affinity")
    report_path = args.output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"cases": [], "errors": []}
    report.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  job_id=os.environ.get("SLURM_JOB_ID"), source_fingerprint=future_source_fingerprint(),
                  production_changed=False, batch_size=32, experiments=["B", "E"],
                  profiles=PROFILES, warmup=args.warmup, steps=args.steps, repeats=args.repeats,
                  trainer_affinity=sorted(os.sched_getaffinity(0)))
    atomic_json(report_path, report)
    bases = {e: OmegaConf.load(args.production_root / e / "launch_config.yaml") for e in ("B", "E")}
    sanity = json.loads((args.production_root / "sanity/report.json").read_text())
    expected_initialization = sanity["experiments"]["B"]["initialization"]
    dataset = hydra.utils.instantiate(bases["B"].dataset)
    started = time.perf_counter()
    dataset.load_ram_cache()
    report["cache_load_seconds"] = time.perf_counter() - started
    atomic_json(report_path, report)
    for repeat in range(args.repeats):
        order = [(e, profile) for e in ("B", "E") for profile in PROFILES]
        random.Random(970042 + repeat).shuffle(order)
        for experiment, profile in order:
            if stop_file.exists():
                atomic_json(args.output / "SAFE_STOP.json", {"completed_cases": len(report["cases"])})
                return
            name = f"r{repeat}_{experiment}_{profile}"
            if any(row["name"] == name for row in report["cases"] + report["errors"]):
                continue
            print(json.dumps({"event": "benchmark_case_start", "name": name}), flush=True)
            try:
                result = run_case(bases[experiment], dataset, args.output / name, profile, repeat,
                                  args.warmup, args.steps)
                if result["initialization_sha256"] != expected_initialization:
                    raise RuntimeError("Runtime case changed the common Stage-1 initialization")
                result["name"] = name
                report["cases"].append(result)
                print(json.dumps({"event": "benchmark_case_pass", "name": name,
                                  "samples_per_second": result["samples_per_second"]}), flush=True)
            except Exception:
                record = {"name": name, "experiment": experiment, "profile": profile,
                          "traceback": traceback.format_exc()}
                report["errors"].append(record)
                print(json.dumps({"event": "benchmark_case_error", **record}), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            report["summary"] = summarize(report["cases"])
            atomic_json(report_path, report)
    # Numerical comparisons are deliberately reported, not treated as policy-quality evidence.
    for row in report["cases"]:
        reference = next((r for r in report["cases"] if r["experiment"] == row["experiment"]
                          and r["repeat"] == row["repeat"] and r["profile"] == "baseline"), None)
        if reference is None:
            continue
        left = torch.load(args.output / row["name"] / "first_fusion_gradient.pt", weights_only=True)
        right = torch.load(args.output / reference["name"] / "first_fusion_gradient.pt", weights_only=True)
        row["first_fusion_gradient_cosine_vs_baseline"] = float(
            torch.nn.functional.cosine_similarity(left, right, dim=0))
        row["first_total_loss_relative_difference"] = abs(
            row["first_step"]["total_loss"] - reference["first_step"]["total_loss"]) / max(
                abs(reference["first_step"]["total_loss"]), 1e-20)
    report["complete"] = True
    report["summary"] = summarize(report["cases"])
    atomic_json(report_path, report)
    atomic_json(args.output / "COMPLETE.json", {"cases": len(report["cases"]), "errors": len(report["errors"])})


if __name__ == "__main__":
    main()
