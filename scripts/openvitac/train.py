#!/usr/bin/env python3
"""Explicit bounded-step or separately authorized formal OpenViTac workspace entry."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal

import hydra
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-updates", type=int, default=0, help="Maximum optimizer updates in this invocation")
    parser.add_argument("--allow-epoch-training", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.max_updates <= 0 and not args.allow_epoch_training:
        parser.error("Specify a bounded --max-updates; full epochs are not enabled for preparation")
    configs = Path(__file__).parents[2] / "configs/openvitac"
    cfg = OmegaConf.merge(OmegaConf.load(configs / "stage0.yaml"), OmegaConf.load(configs / "training.yaml"))
    cfg.dataset.root, cfg.output_dir, cfg.resume = args.root, args.output, args.resume
    cfg.training.device = args.device
    cfg.runtime.stop_after_steps = args.max_updates
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, lambda sig, frame: workspace.request_stop(signal.Signals(sig).name))
    workspace.run()


if __name__ == "__main__":
    main()
