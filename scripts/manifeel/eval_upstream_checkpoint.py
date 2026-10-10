#!/usr/bin/env python3
"""Run ManiFeel's unchanged eval.py on an explicit physical GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys
from typing import Any

import isaacgym  # noqa: F401
import hydra
from omegaconf import DictConfig, open_dict
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_EVAL = REPO_ROOT / "third_party" / "manifeel" / "eval.py"


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, rollout, and physical-device arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--n-test-vis", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--test-start-seed", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    """Keep the upstream runner and metrics, relocating only the CUDA device."""
    args = parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError(f"Expected a physical CUDA device, got {args.device!r}")
    physical_device = int(args.device.split(":", 1)[1])
    if args.output_dir.exists():
        raise FileExistsError(f"Evaluation output exists: {args.output_dir}")

    upstream_compose = hydra.compose

    def compose_on_device(*compose_args: Any, **compose_kwargs: Any) -> DictConfig:
        """Set matching CUDA and graphics ordinals on simulator configs only."""
        config = upstream_compose(*compose_args, **compose_kwargs)
        config_name = compose_kwargs.get(
            "config_name", compose_args[0] if compose_args else ""
        )
        if str(config_name).startswith("isaacgym_config"):
            with open_dict(config):
                config.sim_device = args.device
                config.rl_device = args.device
                config.graphics_device_id = physical_device
        return config

    upstream_load = torch.load

    def load_on_cpu(*load_args: Any, **load_kwargs: Any) -> Any:
        """Avoid materializing saved CUDA tensors on occupied GPU zero."""
        load_kwargs.setdefault("map_location", "cpu")
        return upstream_load(*load_args, **load_kwargs)

    hydra.compose = compose_on_device
    torch.load = load_on_cpu
    sys.argv = [
        str(UPSTREAM_EVAL),
        "--checkpoint", str(args.checkpoint),
        "--output_dir", str(args.output_dir),
        "--device", args.device,
        "--n-test", str(args.n_test),
        "--n-test-vis", str(args.n_test_vis),
        "--max-steps", str(args.max_steps),
        "--test-start-seed", str(args.test_start_seed),
    ]
    runpy.run_path(str(UPSTREAM_EVAL), run_name="__main__")


if __name__ == "__main__":
    main()
