"""Fail closed unless a real four-GPU short run passes correctness and speed gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from omegaconf import OmegaConf

from adavip.manifeel.full_sensor_workspace import training_protocol


def read_records(path: Path) -> list[dict]:
    """Read complete JSONL records while another job may be appending a final line."""
    content = path.read_text()
    return [json.loads(line) for line in content.splitlines(keepends=True) if line.endswith("\n")]


def assess(run: Path, baseline: Path, min_speedup: float = 1.15) -> dict:
    """Check evidence without stopping, signaling or modifying the baseline job."""
    progress = read_records(run / "progress.jsonl")
    metrics = read_records(run / "metrics.jsonl")
    baseline_progress = read_records(baseline / "progress.jsonl")
    baseline_metrics = read_records(baseline / "metrics.jsonl")
    errors = []
    gates = [row for row in progress if row["event"] == "throughput_preflight"]
    windows = [row for row in baseline_progress if row["event"] == "train_step"][-20:]
    baseline_rate = len(windows) / sum(1 / row["samples_per_second"] for row in windows) if windows else 0
    ddp_rate = min((row["samples_per_second"] for row in gates), default=0)
    speedup = ddp_rate / baseline_rate if baseline_rate else 0
    if len(gates) < 2 or not all(row["passed"] for row in gates):
        errors.append("Both initial and resumed phases must pass the throughput gate")
    if speedup < min_speedup:
        errors.append("Measured global throughput did not meet the required speedup")
    cfg, baseline_cfg = OmegaConf.load(run / "config.yaml"), OmegaConf.load(baseline / "config.yaml")
    if training_protocol(cfg) != training_protocol(baseline_cfg):
        errors.append("DDP changed the scientific protocol or optimizer update budget")
    if cfg.dataloader.batch_size != 64 or cfg.optimizer.lr != 0.00008:
        errors.append("Expected global batch64 and unchanged LR8e-5")
    for row in [row for row in progress if row["event"] == "train_step"]:
        if (row.get("world_size"), row.get("per_rank_batch_size"), row.get("global_batch_size")) != (4, 16, 64):
            errors.append("Unexpected effective DDP batch layout")
            break
    cache_reports = [row for row in progress if row["event"] == "shared_cache_verified"]
    memory = []
    if len(cache_reports) < 2:
        errors.append("Missing shared-cache proof before or after resume")
    for row in cache_reports:
        ranks = row["ranks"]
        if len(ranks) != 4 or len({rank["file_identity"] for rank in ranks}) != 1 or len({rank["host"] for rank in ranks}) != 1:
            errors.append("Ranks do not share one node and the same decoded file inodes")
            continue
        nbytes = ranks[0]["nbytes"]
        physical = sum(rank["mapped_bytes"]["Pss"] for rank in ranks)
        resident = statistics.median(rank["mapped_bytes"]["Rss"] for rank in ranks)
        memory.append({"logical_gib": nbytes / 1024 ** 3, "total_rank_pss_gib": physical / 1024 ** 3,
                       "median_rank_rss_gib": resident / 1024 ** 3})
        if physical > nbytes * 1.10 or resident < nbytes * 0.90:
            errors.append("Decoded pages were copied or were not resident before the measured run")
    plans = [row for row in progress if row["event"] == "global_sampler_verified"]
    if not plans or any(row["global_samples"] != 69205 or row["global_batches"] != 1082
                        or row["tail_samples_per_rank"] != [6, 5, 5, 5] for row in plans):
        errors.append("Weighted sampler budget, global steps or unpadded tail changed")
    for epoch in {row["epoch"] for row in plans}:
        if len({row["sample_order_sha256"] for row in plans if row["epoch"] == epoch}) != 1:
            errors.append("Resume changed the global weighted sample order")
    complete = [row for row in metrics if row.get("runtime/resumed_batch_cursor") == 0]
    if not complete:
        errors.append("No uninterrupted full epoch was measured")
    baseline_by_epoch = {row["epoch"]: row for row in baseline_metrics}
    matched = [(row, baseline_by_epoch[row["epoch"]]) for row in complete if row["epoch"] in baseline_by_epoch]
    full_epoch_speedup = 0.0
    full_epoch_ddp_rate = full_epoch_baseline_rate = 0.0
    if matched:
        ddp_samples = sum(sum(value for key, value in row.items() if key.startswith("train/count/"))
                          for row, reference in matched)
        baseline_samples = sum(sum(value for key, value in reference.items() if key.startswith("train/count/"))
                               for row, reference in matched)
        full_epoch_ddp_rate = ddp_samples / sum(row["train/seconds"] for row, reference in matched)
        full_epoch_baseline_rate = baseline_samples / sum(reference["train/seconds"] for row, reference in matched)
        full_epoch_speedup = full_epoch_ddp_rate / full_epoch_baseline_rate if full_epoch_baseline_rate else 0.0
    if full_epoch_speedup < min_speedup:
        errors.append("Matched full-epoch training throughput, including loader startup, did not meet required speedup")
    for row in metrics:
        if (row.get("validation/task_coverage") != 9 or row.get("validation/sample_count") != 2060
                or sum(value for key, value in row.items() if key.startswith("train/count/")) != 69205):
            errors.append("Validation reduction or global training sample accounting is incomplete")
        if any(f"validation/macro/{condition}" not in row for condition in
               ("full", "V1", "V2", "V3", "T1", "T2", "P", "L", "vision", "tactile")):
            errors.append("Missing paired validation condition")
    initialized = [row for row in progress if row["event"] == "replicas_verified" and row["phase"] == "initialized"]
    committed = [row for row in progress if row["event"] == "replicas_verified" and row["phase"] == "epoch_complete"]
    signature = ("model", "ema_model", "global_step", "ema_step", "batch_in_epoch")
    if len(initialized) < 2 or not committed or any(initialized[-1][key] != committed[-1][key] for key in signature):
        errors.append("Checkpoint reload did not preserve raw/EMA replicas and update cursor")
    if not any(row["event"] == "resume_rng_verified" and row["source_world_size"] == 4 for row in progress):
        errors.append("Missing verified per-rank RNG restore")
    if not progress or progress[-1].get("reason") != "bounded_step_preflight":
        errors.append("Resumed phase did not finish its bounded optimizer updates safely")
    for tag in ("latest", "interrupt"):
        if not (run / "checkpoints" / f"{tag}.ckpt").is_file():
            errors.append(f"Missing {tag} checkpoint")
    return {"passed": not errors, "errors": errors, "baseline_run": str(baseline), "ddp_run": str(run),
            "baseline_samples_per_second": baseline_rate, "ddp_samples_per_second": ddp_rate,
            "speedup": speedup, "required_speedup": min_speedup, "memory": memory,
            "matched_full_epochs": [row["epoch"] for row, reference in matched],
            "full_epoch_ddp_samples_per_second": full_epoch_ddp_rate,
            "full_epoch_baseline_samples_per_second": full_epoch_baseline_rate,
            "full_epoch_speedup": full_epoch_speedup,
            "full_epoch_wall_seconds": [row["runtime/epoch_wall_seconds"] for row in complete],
            "last_global_step": progress[-1].get("global_step") if progress else None}


def main() -> None:
    """Persist a machine-readable pass/fail decision for subsequent long-run review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=1.15)
    args = parser.parse_args()
    report = assess(args.run, args.baseline_run, args.min_speedup)
    temporary = args.run / "ddp_preflight_report.partial"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.run / "ddp_preflight_report.json")
    print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
    if not report["passed"]:
        raise SystemExit("DDP preflight failed; leave the single-GPU baseline running")


if __name__ == "__main__":
    main()
