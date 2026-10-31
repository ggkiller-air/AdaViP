"""Causal L1 windows with one separately indexed future frame per sensor."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, TACTILE_KEYS
from adavip.manifeel.language_conditioning_dataset import LanguageConditioningDataset
from adavip.manifeel.language_conditioning_dataset import initialize_worker


EXPERIMENTS = {
    "A": ("none", ()),
    "B": ("raw", TACTILE_KEYS),
    "C": ("latent", TACTILE_KEYS),
    "D": ("raw", IMAGE_KEYS),
    "E": ("latent", IMAGE_KEYS),
}


def initialize_future_worker(worker_id: int) -> None:
    """Retain the L1 worker setup and record actual CPU affinities once per worker."""
    inherited = sorted(os.sched_getaffinity(0))
    initialize_worker(worker_id)
    # OpenMP initialization may narrow the affinity; choose from the original srun allocation.
    os.sched_setaffinity(0, {inherited[worker_id % len(inherited)]})
    directory = os.environ.get("STAGE1_FUTURE_AFFINITY_DIR")
    if directory:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"worker_{os.getpid()}.json").write_text(json.dumps({
            "worker": worker_id, "pid": os.getpid(), "inherited_affinity": inherited,
            "actual_affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads()}))


class Stage1FutureDataset(LanguageConditioningDataset):
    """Preserve all L1 anchors; mask t+k rather than clipping across episodes."""

    def __init__(self, *args: object, experiment: str = "A", future_offset: int = 4,
                 raw_resolution: int = 64, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if experiment not in EXPERIMENTS or future_offset < 1 or raw_resolution < 8:
            raise ValueError("Invalid future experiment, offset or raw resolution")
        self.target_space, self.future_keys = EXPERIMENTS[experiment]
        self.future_offset, self.raw_resolution = future_offset, raw_resolution

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        task, _, start, end, timestep = self._resolve_index(index)
        anchor, target = start + timestep, start + timestep + self.future_offset
        valid = target < end
        targets = {}
        for key in self.future_keys:
            if valid:
                value = torch.from_numpy(self._read_images(
                    self.groups[task]["data"][key], np.array([target]), key))[0]
                if self.target_space == "raw":
                    value = F.interpolate(value[None], size=(self.raw_resolution, self.raw_resolution),
                                          mode="bilinear", align_corners=False, antialias=True)[0]
            else:
                shape = ((3, self.raw_resolution, self.raw_resolution) if self.target_space == "raw"
                         else self.shape_meta["obs"][key]["shape"])
                value = torch.zeros(tuple(shape), dtype=torch.float32)
            targets[key] = value
        sample.update(future_obs=targets, future_valid=torch.tensor(valid),
                      future_anchor=torch.tensor(anchor),
                      future_index=torch.tensor(target if valid else -1),
                      future_episode_end=torch.tensor(end))
        return sample
