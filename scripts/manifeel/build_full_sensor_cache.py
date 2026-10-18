"""Prepare lossless decoded arrays without constructing a policy or using a GPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import hydra
from numcodecs import blosc
from omegaconf import OmegaConf

from adavip.manifeel.decoded_sensor_cache import build_decoded_cache


def main() -> None:
    """Decode the exact training configuration's selected train and validation data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--block-frames", type=int, default=48)
    arguments = parser.parse_args()
    if arguments.workers > len(os.sched_getaffinity(0)):
        raise ValueError("Cache worker count exceeds allocated CPUs")
    cv2.setNumThreads(1)
    blosc.set_nthreads(1)
    cfg = OmegaConf.load(arguments.config)
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset_cfg.pop("decoded_cache_dir", None)
    dataset_cfg.pop("decoded_cache_mode", None)
    dataset = hydra.utils.instantiate(dataset_cfg)
    build_decoded_cache(dataset, arguments.cache_dir, arguments.workers, arguments.block_frames)


if __name__ == "__main__":
    main()
