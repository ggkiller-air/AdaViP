#!/usr/bin/env python3
"""Restore an exported Piper RDP policy and audit offline real-data inference."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import zarr

from adavip.real_world.piper_rdp_runtime import (
    OnlineGelSightPcaProcessor,
    load_policy_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_OUTPUT = Path(
    os.environ.get("ADAVIP_RDP_OUTPUT", REPO_ROOT / "outputs" / "piper_rdp")
)
DEFAULT_BUNDLE = DEFAULT_TRAINING_OUTPUT / "deployment/piper_rdp_18_75hz_ema"
DEFAULT_DATASET = Path(
    os.environ.get("ADAVIP_RDP_DATASET", REPO_ROOT / "datasets" / "piper_rdp")
)
DEFAULT_INDICES = (20, 127, 327, 404, 550, 654)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--sample-indices",
        default=",".join(str(index) for index in DEFAULT_INDICES),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_indices(value: str) -> list[int]:
    """Parse a comma-separated non-empty list of sample indices."""
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices or any(index < 0 for index in indices):
        raise ValueError("Sample indices must be non-negative integers")
    return indices


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    """Synchronize accelerator work before recording latency."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_range(value: torch.Tensor) -> dict[str, Any]:
    """Return finite and per-dimension range diagnostics."""
    cpu = value.detach().float().cpu()
    flattened = cpu.reshape(-1, cpu.shape[-1])
    return {
        "finite": bool(torch.isfinite(cpu).all()),
        "minimum": float(cpu.min()),
        "maximum": float(cpu.max()),
        "minimum_per_dimension": flattened.min(dim=0).values.tolist(),
        "maximum_per_dimension": flattened.max(dim=0).values.tolist(),
    }


def timed_call(device: torch.device, function: Any) -> tuple[Any, float]:
    """Call a policy function and return synchronized wall-clock milliseconds."""
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def audit_gelsight_runtime(bundle: Path, dataset_path: Path) -> dict[str, Any]:
    """Compare online first-frame PCA processing with prepared real-data values."""
    replay = zarr.open(str(dataset_path / "replay_buffer.zarr"), mode="r")
    data = replay["data"]
    episode_ends = np.asarray(replay["meta/episode_ends"][:], dtype=np.int64)
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    processor = OnlineGelSightPcaProcessor.from_directory(bundle)
    frame_reports = []
    for episode_index, (start, end) in enumerate(zip(episode_starts, episode_ends)):
        processor.reset()
        processor.initialize(data["right_gelsight_img"][int(start)])
        frame_indices = sorted(
            {int(start), int((start + end - 1) // 2), int(end - 1)}
        )
        for frame_index in frame_indices:
            result = processor.process(data["right_gelsight_img"][frame_index])
            expected = np.asarray(
                data["right_gelsight_marker_offset_emb"][frame_index],
                dtype=np.float32,
            )
            max_abs_error = float(
                np.max(np.abs(result["marker_offset_emb"] - expected))
            )
            frame_reports.append(
                {
                    "episode": episode_index,
                    "frame_index": frame_index,
                    "detected_markers": int(result["detected_markers"]),
                    "assignment_max_px": float(result["assignment_max_px"]),
                    "pca_max_abs_error": max_abs_error,
                }
            )
    maximum_error = max(item["pca_max_abs_error"] for item in frame_reports)
    return {
        "passed": bool(
            all(item["detected_markers"] == 63 for item in frame_reports)
            and maximum_error <= 1e-6
        ),
        "reference": "first frame of each of 3 episodes",
        "checked_frames": len(frame_reports),
        "maximum_pca_abs_error": maximum_error,
        "frames": frame_reports,
    }


def audit_sample(
    policy: torch.nn.Module,
    sample: dict[str, Any],
    sample_index: int,
    device: torch.device,
    seed: int,
    ratio: int,
) -> dict[str, Any]:
    """Audit token generation, full decoding, and causal prefix decoding."""
    obs = {
        key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()
    }
    extended = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample["extended_obs"].items()
    }
    ground_truth = sample["action"].unsqueeze(0).to(device)[:, 3:29]
    torch.manual_seed(seed + sample_index)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + sample_index)

    token_result, token_ms = timed_call(
        device,
        lambda: policy.predict_action(
            obs,
            dataset_obs_temporal_downsample_ratio=ratio,
            return_latent_action=True,
        ),
    )
    token_schedule = token_result["action"]
    token = token_schedule[:, 0]
    repeated_max_abs_error = float(
        (token_schedule - token_schedule[:, :1]).abs().max().detach().cpu()
    )

    full_result, full_decode_ms = timed_call(
        device,
        lambda: policy.predict_from_latent_action(
            token,
            extended,
            extended_obs_last_step=29,
            dataset_obs_temporal_downsample_ratio=ratio,
        ),
    )
    predicted = full_result["action"]
    error = predicted - ground_truth
    prefix_reports = []
    for history_length in (4, 16, 29):
        prefix_extended = {
            key: value[:, :history_length] for key, value in extended.items()
        }
        prefix_result, prefix_ms = timed_call(
            device,
            lambda prefix_extended=prefix_extended, history_length=history_length: (
                policy.predict_from_latent_action(
                    token,
                    prefix_extended,
                    extended_obs_last_step=history_length,
                    dataset_obs_temporal_downsample_ratio=ratio,
                )
            ),
        )
        prefix_action = prefix_result["action"]
        expected_steps = history_length - 3
        causal_reference = full_result["action_pred"][:, :history_length]
        causal_max_abs_error = float(
            (prefix_result["action_pred"] - causal_reference)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        prefix_reports.append(
            {
                "history_length": history_length,
                "action_shape": list(prefix_action.shape),
                "expected_action_steps": expected_steps,
                "latency_ms": prefix_ms,
                "finite": bool(torch.isfinite(prefix_action).all()),
                "causal_prefix_max_abs_error": causal_max_abs_error,
                "last_action": prefix_action[0, -1].detach().float().cpu().tolist(),
                "passed": bool(
                    list(prefix_action.shape) == [1, expected_steps, 7]
                    and torch.isfinite(prefix_action).all()
                    and causal_max_abs_error <= 1e-5
                ),
            }
        )

    sample_passed = bool(
        list(token_schedule.shape) == [1, 26, 64]
        and repeated_max_abs_error == 0.0
        and list(predicted.shape) == [1, 26, 7]
        and torch.isfinite(token_schedule).all()
        and torch.isfinite(predicted).all()
        and all(item["passed"] for item in prefix_reports)
    )
    return {
        "sample_index": sample_index,
        "passed": sample_passed,
        "token_schedule_shape": list(token_schedule.shape),
        "token_schedule_finite": bool(torch.isfinite(token_schedule).all()),
        "token_rows_repeated_max_abs_error": repeated_max_abs_error,
        "token_generation_latency_ms": token_ms,
        "full_decode_shape": list(predicted.shape),
        "full_decode_latency_ms": full_decode_ms,
        "prediction_range": tensor_range(predicted),
        "ground_truth_range": tensor_range(ground_truth),
        "ground_truth_mse": float(error.square().mean().detach().cpu()),
        "ground_truth_mse_per_dimension": error.square()
        .mean(dim=(0, 1))
        .detach()
        .float()
        .cpu()
        .tolist(),
        "prefix_decodes": prefix_reports,
    }


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    artifact = bundle / "piper_rdp_ema.pt"
    output = args.output or bundle / "offline_inference_report.json"
    indices = parse_indices(args.sample_indices)
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    load_start = time.perf_counter()
    policy, metadata = load_policy_artifact(artifact, device=device)
    restore_ms = (time.perf_counter() - load_start) * 1000.0
    dataset_config = metadata["offline_dataset_config"]
    dataset_config["dataset_path"] = str(args.dataset.resolve())
    dataset = hydra.utils.instantiate(dataset_config)
    if any(index >= len(dataset) for index in indices):
        raise IndexError(f"Sample index exceeds dataset length {len(dataset)}")

    gelsight_runtime = audit_gelsight_runtime(bundle, args.dataset.resolve())
    print(
        f"gelsight_runtime_passed={gelsight_runtime['passed']} "
        f"frames={gelsight_runtime['checked_frames']} "
        f"pca_max_abs_error={gelsight_runtime['maximum_pca_abs_error']:.3e}",
        flush=True,
    )
    sample_reports = []
    with torch.inference_mode():
        for index in indices:
            sample_report = audit_sample(
                policy,
                dataset[index],
                index,
                device,
                args.seed,
                int(metadata["runtime"]["dataset_obs_temporal_downsample_ratio"]),
            )
            sample_reports.append(sample_report)
            print(
                f"sample={index} passed={sample_report['passed']} "
                f"token_ms={sample_report['token_generation_latency_ms']:.1f} "
                f"decode_ms={sample_report['full_decode_latency_ms']:.1f} "
                f"mse={sample_report['ground_truth_mse']:.8f}",
                flush=True,
            )

    report = {
        "format": "adavip.piper_rdp.offline_inference_report.v1",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "passed": bool(
            gelsight_runtime["passed"]
            and all(item["passed"] for item in sample_reports)
        ),
        "scope": "trained-data offline structural and numerical audit; not robot success",
        "weights": metadata["weights"],
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256_file(artifact),
        "device": str(device),
        "cpu_threads": args.cpu_threads,
        "policy_restore_strict": True,
        "at_normalizer_synchronized_from_policy": bool(
            metadata["at_normalizer_synchronized_from_policy"]
        ),
        "policy_restore_latency_ms": restore_ms,
        "num_inference_steps": int(metadata["runtime"]["num_inference_steps"]),
        "dataset": str(args.dataset.resolve()),
        "dataset_length": len(dataset),
        "sample_indices": indices,
        "sample_count": len(indices),
        "gelsight_runtime": gelsight_runtime,
        "mean_ground_truth_mse": float(
            np.mean([item["ground_truth_mse"] for item in sample_reports])
        ),
        "mean_token_generation_latency_ms": float(
            np.mean([item["token_generation_latency_ms"] for item in sample_reports])
        ),
        "mean_full_decode_latency_ms": float(
            np.mean([item["full_decode_latency_ms"] for item in sample_reports])
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "samples": sample_reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"report={output}")
    print(f"passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
