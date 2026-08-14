#!/usr/bin/env python3
"""Run an offline inference smoke test for the no-tactile DP checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import dill
import hydra
import torch
from omegaconf import OmegaConf


DEFAULT_CHECKPOINT = Path(
    "/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/seed_42/"
    "checkpoints/epoch=0190-train_loss=0.022.ckpt"
)
ARTIFACT_FORMAT = "adavip.dp_sanity.inference.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sample-index", type=int, default=300)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timed-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA before reading a wall-clock inference duration."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_policy(
    policy: torch.nn.Module,
    obs: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], float]:
    """Run one policy sample and return its synchronized latency in ms."""
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        result = policy.predict_action(obs)
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    """Return JSON-compatible shape, finite, and range diagnostics."""
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def load_policy(
    checkpoint: Path,
) -> tuple[torch.nn.Module, Any, str, str]:
    """Restore a policy from a full workspace or compact inference artifact."""
    with checkpoint.open("rb") as checkpoint_file:
        payload = torch.load(
            checkpoint_file,
            map_location="cpu",
            pickle_module=dill,
        )

    if payload.get("format") == ARTIFACT_FORMAT:
        policy_config = OmegaConf.create(payload["policy_config"])
        policy = hydra.utils.instantiate(policy_config)
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
        dataset_config = OmegaConf.create(
            payload["offline_validation_dataset_config"]
        )
        weights = str(payload["weights"])
        del payload
        return policy, dataset_config, weights, ARTIFACT_FORMAT

    cfg = payload["cfg"]
    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace = workspace_class(cfg, output_dir=str(checkpoint.parent.parent))
    excluded = tuple(
        key
        for key in payload["state_dicts"]
        if key not in {"model", "ema_model"}
    )
    workspace.load_payload(payload, exclude_keys=excluded)
    use_ema = bool(cfg.training.use_ema)
    policy = workspace.ema_model if use_ema else workspace.model
    if policy is None:
        raise RuntimeError("Checkpoint config selected EMA but no EMA model was restored")
    dataset_config = cfg.task.dataset
    del payload
    return policy, dataset_config, "ema_model" if use_ema else "model", "workspace"


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    if args.warmup_runs < 0 or args.timed_runs < 1:
        raise ValueError("Require non-negative warmups and at least one timed run")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested but no GPU is visible")

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    print(f"[inference] loading checkpoint={args.checkpoint}", flush=True)
    policy, dataset_config, weights, checkpoint_format = load_policy(args.checkpoint)
    policy.num_inference_steps = args.num_inference_steps
    policy.eval().to(device)

    dataset = hydra.utils.instantiate(dataset_config)
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"sample index {args.sample_index} is outside dataset length {len(dataset)}"
        )
    sample = dataset[args.sample_index]
    obs = {
        key: value.unsqueeze(0).to(device, non_blocking=True)
        for key, value in sample["obs"].items()
    }
    shape_meta = OmegaConf.to_container(dataset_config.shape_meta, resolve=True)
    expected_obs_keys = tuple(shape_meta["obs"])
    if tuple(obs) != expected_obs_keys:
        raise RuntimeError(f"Unexpected observation keys: {tuple(obs)}")

    observation_steps = int(dataset_config.n_obs_steps)
    expected_obs_shapes = {
        key: [1, observation_steps, *attributes["shape"]]
        for key, attributes in shape_meta["obs"].items()
    }
    actual_obs_shapes = {key: list(value.shape) for key, value in obs.items()}
    if actual_obs_shapes != expected_obs_shapes:
        raise RuntimeError(f"Unexpected observation shapes: {actual_obs_shapes}")

    ground_truth = sample["action"].unsqueeze(0).to(device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    result, quality_latency_ms = run_policy(policy, obs, device)

    action = result["action"]
    action_pred = result["action_pred"]
    action_dim = int(shape_meta["action"]["shape"][0])
    n_action_steps = int(policy.n_action_steps)
    horizon = int(dataset_config.horizon)
    if list(action.shape) != [1, n_action_steps, action_dim]:
        raise RuntimeError(f"Unexpected rollout action shape: {list(action.shape)}")
    if list(action_pred.shape) != [1, horizon, action_dim]:
        raise RuntimeError(f"Unexpected full action shape: {list(action_pred.shape)}")
    if not torch.isfinite(action).all() or not torch.isfinite(action_pred).all():
        raise RuntimeError("Policy produced NaN or infinite action values")

    for _ in range(args.warmup_runs):
        run_policy(policy, obs, device)
    latencies_ms = [
        run_policy(policy, obs, device)[1] for _ in range(args.timed_runs)
    ]

    full_mse = torch.nn.functional.mse_loss(action_pred, ground_truth)
    action_start = observation_steps - 1
    chunk_target = ground_truth[:, action_start : action_start + n_action_steps]
    chunk_mse = torch.nn.functional.mse_loss(action, chunk_target)
    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_format": checkpoint_format,
        "policy_weights": weights,
        "sample_index": args.sample_index,
        "dataset_length": len(dataset),
        "num_inference_steps": args.num_inference_steps,
        "observation_shapes": actual_obs_shapes,
        "action": tensor_summary(action),
        "action_pred": tensor_summary(action_pred),
        "ground_truth": tensor_summary(ground_truth),
        "full_horizon_mse": float(full_mse.item()),
        "rollout_chunk_mse": float(chunk_mse.item()),
        "quality_sample_latency_ms": quality_latency_ms,
        "timed_latency_ms": latencies_ms,
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_mean_ms": statistics.mean(latencies_ms),
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report["gpu"] = {
            "name": properties.name,
            "total_memory_mib": round(properties.total_memory / (1024**2)),
            "peak_allocated_mib": round(
                torch.cuda.max_memory_allocated(device) / (1024**2)
            ),
        }

    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized, flush=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
        print(f"[inference] report={args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
