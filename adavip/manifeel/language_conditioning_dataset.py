"""All-demonstration data and deferred, lossless shared-RAM cache loading."""

from __future__ import annotations

import copy
import os

import cv2
import numpy as np
import torch

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray, load_decoded_cache
from adavip.manifeel.full_sensor_dataset import FullSensorManiFeelDataset


def initialize_worker(worker_id: int) -> None:
    """Keep CPU workers on separate allocated cores and avoid nested thread pools."""
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    if hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {cores[worker_id % len(cores)]})


class LanguageConditioningDataset(FullSensorManiFeelDataset):
    """Use every real anchor, with existing episode-boundary and 6D action masks."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.cache_directory = kwargs.pop("decoded_cache_dir", None)
        kwargs.pop("decoded_cache_mode", None)
        super().__init__(*args, **kwargs)
        # The old cache covers train UNION validation, already all 400 demos.
        # Retain that exact source contract while changing only training selection.
        self._cache_source = copy.copy(self)
        self._cache_source.manifest = copy.deepcopy(self.manifest)
        self.train_episodes = [np.arange(len(ends)) for ends in self.episode_ends]
        self.val_episodes = [np.empty(0, dtype=np.int64) for _ in self.episode_ends]
        for store, episodes in zip(self.manifest["stores"], self.train_episodes):
            store["train_episodes"], store["val_episodes"] = episodes.tolist(), []
        self._build_index()
        self.manifest["training_selection"] = "all demonstrations; no validation split"
        self.verified_ram_cache = False

    def load_ram_cache(self) -> None:
        """Load/hash all arrays in parallel after the first formal update succeeds."""
        if self.cache_directory and not self.verified_ram_cache:
            self.groups = load_decoded_cache(self._cache_source, self.cache_directory, "ram")
            self.verified_ram_cache = True

    def _read_images(self, array: object, indices: np.ndarray, key: str) -> np.ndarray:
        if isinstance(array, SourceIndexedArray) and self.verified_ram_cache:
            rows = array.row_map[indices]
            if np.any(rows < 0):
                raise IndexError("Missing decoded frame")
            # A slice avoids advanced indexing copies for the normal two-frame case.
            if len(rows) > 1 and np.all(np.diff(rows) == 1):
                images = array.values[int(rows[0]):int(rows[-1]) + 1]
            else:
                images = array.values[rows]
            return np.ascontiguousarray(np.moveaxis(images, -1, 1))
        return super()._read_images(array, indices, key)
