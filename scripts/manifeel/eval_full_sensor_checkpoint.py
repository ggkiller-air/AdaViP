#!/usr/bin/env python3
"""Evaluate Stage0 or either Stage2 fusion, isolating tasks in separate processes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    """Parse portable checkpoint, embedding and rollout settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-embedding-path", type=Path)
    parser.add_argument("--config", type=Path, help="Override the legacy Stage2 inference architecture YAML")
    parser.add_argument("--check-only", action="store_true", help="Load weights on CPU without Isaac Gym or a GPU")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--task-id", help="One ManiFeel task ID; omit to evaluate all nine")
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--n-test-vis", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--test-start-seed", type=int, default=100000)
    parser.add_argument("--policy-seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_test < 1 or args.max_steps < 1 or not 0 <= args.n_test_vis <= args.n_test:
        parser.error("Require n-test/max-steps > 0 and 0 <= n-test-vis <= n-test")
    if not args.device.startswith("cuda:"):
        parser.error("Simulator evaluation requires --device cuda:<index>")
    return args


def evaluate_task(args: argparse.Namespace) -> None:
    """Import Isaac Gym before torch, then run the complete sensor policy."""
    import isaacgym
    import hydra
    import numpy as np
    from omegaconf import OmegaConf
    import torch

    from adavip.manifeel.device_runner import DeviceAwareManifeelRunner
    from adavip.manifeel.full_sensor_evaluation import FullSensorEvaluationPolicy, load_evaluation_policy
    from adavip.manifeel.task_embeddings import load_task_embeddings
    from adavip.manifeel.task_protocol import DEFAULT_TASK_SPECS

    matching_specs = [spec for spec in DEFAULT_TASK_SPECS if spec.task_id == args.task_id]
    if not matching_specs:
        raise ValueError(f"Unknown ManiFeel task: {args.task_id}")
    spec = matching_specs[0]
    policy, cfg, metadata = load_evaluation_policy(args.checkpoint, args.device, args.weights, args.config)
    embedding_path = args.task_embedding_path
    if embedding_path is None:
        bundled = args.checkpoint.parent.parent / "task_embeddings.npz"
        if bundled.is_file():
            embedding_path = bundled
        elif cfg.get("task_embedding_path"):
            embedding_path = Path(cfg.task_embedding_path)
        else:
            raise ValueError("Supply --task-embedding-path with the frozen training task_embeddings.npz")
    embeddings = load_task_embeddings(embedding_path, [spec])
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    conditioned = FullSensorEvaluationPolicy(policy, shape_meta, embeddings[spec.task_id], spec.action_dim)
    environment_meta = {"obs": {key: value for key, value in shape_meta["obs"].items()
                                 if key != "task_embedding"}, "action": {"shape": [spec.action_dim]}}
    tactile_sizes = {tuple(shape_meta["obs"][key]["shape"][1:])
                     for key in ("left_tactile_camera_taxim", "right_tactile_camera_taxim")}
    if len(tactile_sizes) != 1:
        raise ValueError("The simulator runner requires matching left/right tactile sizes")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    os.environ["MANIFEEL_DEVICE"] = args.device
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "third_party/manifeel/manifeel/config"), version_base=None,
    ):
        runner = DeviceAwareManifeelRunner(
            output_dir=str(args.output_dir), shape_meta=environment_meta,
            isaacgym_cfg_name=spec.isaacgym_cfg_name, device=args.device,
            n_test=args.n_test, n_test_vis=args.n_test_vis, test_start_seed=args.test_start_seed,
            max_steps=args.max_steps, n_obs_steps=int(cfg.n_obs_steps),
            n_action_steps=int(cfg.n_action_steps), tactile_size=list(next(iter(tactile_sizes))),
        )
        torch.manual_seed(args.policy_seed)
        torch.cuda.manual_seed_all(args.policy_seed)
        np.random.seed(args.policy_seed)
        with torch.no_grad():
            results = runner.run(conditioned)
    mean_score = float(results["test/mean_score"])
    if not math.isfinite(mean_score):
        raise ValueError("Simulator returned nonfinite test/mean_score")
    report = {
        "task_id": args.task_id, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_metadata": metadata, "test/mean_score": mean_score,
        "n_test": args.n_test, "max_steps": args.max_steps,
        "test_start_seed": args.test_start_seed, "policy_seed": args.policy_seed,
        "task_embedding_path": str(embedding_path.resolve()),
        "protocol": "full-input; EMA by default; environment maximum reward mean",
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    """Keep simulator lifetimes separate and aggregate the official mean score."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.check_only:
        from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy

        policy, cfg, metadata = load_evaluation_policy(args.checkpoint, "cpu", args.weights, args.config)
        report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_metadata": metadata,
                  "policy_class": type(policy).__name__, "n_obs_steps": int(cfg.n_obs_steps),
                  "parameters": sum(parameter.numel() for parameter in policy.parameters()),
                  "check": "strict CPU weight/normalizer load; simulator not started"}
        (args.output_dir / "checkpoint_check.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return
    if args.task_id:
        evaluate_task(args)
        return
    from adavip.manifeel.task_protocol import DEFAULT_TASK_SPECS

    reports = {}
    for spec in DEFAULT_TASK_SPECS:
        output = args.output_dir / spec.task_id
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--checkpoint", str(args.checkpoint.resolve()), "--output-dir", str(output.resolve()),
                   "--task-id", spec.task_id, "--device", args.device, "--weights", args.weights,
                   "--n-test", str(args.n_test), "--n-test-vis", str(args.n_test_vis),
                   "--max-steps", str(args.max_steps), "--test-start-seed", str(args.test_start_seed),
                   "--policy-seed", str(args.policy_seed)]
        if args.task_embedding_path:
            command.extend(["--task-embedding-path", str(args.task_embedding_path.resolve())])
        if args.config:
            command.extend(["--config", str(args.config.resolve())])
        with (args.output_dir / f"{spec.task_id}.log").open("w") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        reports[spec.task_id] = json.loads((output / "evaluation.json").read_text())
    summary = {"tasks": reports, "test/macro_mean_score":
               sum(report["test/mean_score"] for report in reports.values()) / len(reports)}
    (args.output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
