"""Join complete T=2 utility labels to the existing lossless decoded RAM cache."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from torch.utils.data import Dataset, WeightedRandomSampler

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray, array_digest, cache_contract, fingerprint
from adavip.manifeel.full_sensor_dataset import FullSensorManiFeelDataset
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS
from adavip.manifeel.stage1_dataset import UtilityLabelStore, align_source_to_labels


class Stage2SourceDataset(FullSensorManiFeelDataset):
    """Skip redundant per-window scans after the complete cache was hash verified."""

    def _read_images(self, array: Any, indices: np.ndarray, key: str) -> np.ndarray:
        if isinstance(array, SourceIndexedArray) and getattr(self, "verified_ram_cache", False):
            rows = array.row_map[indices]
            if np.any(rows < 0):
                raise IndexError("Missing decoded sensor row")
            images = array.values[rows]
            height, width = self.shape_meta["obs"][key]["shape"][1:]
            if images.shape[1:3] != (height, width):
                raise ValueError("Decoded RAM cache image shape changed")
            return np.ascontiguousarray(np.moveaxis(images, -1, 1))
        return super()._read_images(array, indices, key)


def validate_complete_ram_cache(source: Stage2SourceDataset, directory: str,
                                cache_config: str, require_full_ram: bool = True) -> dict:
    """Check complete metadata; require full-cache RAM unless only mapping a subset.

    Benchmark subset readers can disable the RAM-capacity requirement while
    preserving all source-contract, array-set, shape and dtype validation.
    Formal cache loading retains the capacity check by default.
    """
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    frozen_config = OmegaConf.to_container(OmegaConf.load(cache_config).dataset, resolve=True)
    frozen_config.pop("decoded_cache_dir", None)
    frozen_config.pop("decoded_cache_mode", None)
    original_source = hydra.utils.instantiate(frozen_config)
    if fingerprint(cache_contract(original_source)) != manifest["contract"]:
        raise ValueError("Decoded cache does not match its original source contract")
    for original, labeled in zip(original_source.manifest["stores"], source.manifest["stores"]):
        for key in ("path", "arrays", "episode_ends"):
            if original[key] != labeled[key]:
                raise ValueError(f"Decoded cache and labels differ in source {key}")
    del original_source
    records = manifest["arrays"]
    expected = {(task, key) for task in range(len(source.task_specs))
                for key in (*IMAGE_KEYS, "state", "action")}
    if {(item["task_index"], item["key"]) for item in records} != expected or len(records) != len(expected):
        raise ValueError("RAM cache does not contain exactly the nine-task sensor set")
    required = int(manifest["nbytes"]) + 32 * 1024 ** 3
    allocated = os.environ.get("SLURM_MEM_PER_NODE")
    if require_full_ram and allocated and required > int(allocated) * 1024 ** 2:
        raise MemoryError("H200 job memory is too small for complete cache plus 32 GiB headroom")
    for record in records:
        mapped = np.load(root / record["filename"], mmap_mode="r", allow_pickle=False)
        if (mapped.dtype != np.float32 or list(mapped.shape) != record["shape"]
                or len(mapped) != int(source.episode_ends[int(record["task_index"])][-1])
                or mapped.nbytes != record["nbytes"]):
            raise ValueError(f"Wrong cache shape or dtype: {record['filename']}")
    return manifest


def load_complete_ram_cache(source: Stage2SourceDataset, directory: str,
                            cache_config: str, workers: int = 1) -> int:
    """Load/hash all decoded arrays into shared read-only parent RAM before fork."""
    manifest = validate_complete_ram_cache(source, directory, cache_config)
    root = Path(directory)
    groups = [{"data": {}} for _ in source.task_specs]

    def load(record: dict) -> tuple[int, str, SourceIndexedArray, int]:
        task, key = int(record["task_index"]), record["key"]
        expected_length = int(source.episode_ends[task][-1])
        mapped = np.load(root / record["filename"], mmap_mode="r", allow_pickle=False)
        values = np.array(mapped, copy=True)
        if array_digest(values) != record["sha256"]:
            raise ValueError(f"Corrupt decoded cache: {record['filename']}")
        values.flags.writeable = False
        print(json.dumps({"event": "stage2_ram_array", "task": task, "key": key,
                          "bytes": values.nbytes}), flush=True)
        return task, key, SourceIndexedArray(values, np.arange(expected_length), expected_length), values.nbytes

    total = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for task, key, array, size in pool.map(load, manifest["arrays"]):
            groups[task]["data"][key] = array
            total += size
    if total != manifest["nbytes"]:
        raise ValueError("Decoded RAM byte count mismatch")
    source.groups = groups
    source.verified_ram_cache = True
    return total


class Stage2UtilityDataset(Dataset):
    """Provide all complete labeled windows without future tactile decoding."""

    def __init__(self, source: Stage2SourceDataset, store: UtilityLabelStore) -> None:
        self.source = source
        self.store = store
        self.task_specs = source.task_specs
        align_source_to_labels(source, store)
        self.task_indices = np.asarray(store.index["task_index"], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.store.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.store.index[index]
        result = self.source[int(row["dataset_index"])]
        for key in ("task_index", "episode_index", "timestep", "source_index"):
            if int(result[key]) != int(row[key]):
                raise ValueError(f"Window/label alignment failed for {key}")
        if not result["observation_valid_mask"].all() or not result["action_valid_mask"].all():
            raise ValueError("Only complete labeled windows can be trained")
        result["utility"] = torch.from_numpy(np.asarray(self.store.utility[index], dtype=np.float32).copy())
        return result

    def make_task_balanced_sampler(self, generator: torch.Generator) -> WeightedRandomSampler:
        """Draw one epoch of windows with equal expected mass per task."""
        counts = np.bincount(self.task_indices)
        weights = 1.0 / counts[self.task_indices]
        return WeightedRandomSampler(torch.from_numpy(weights), len(self), replacement=True, generator=generator)


def make_stage2_dataset(label_dir: str, dataset_root: str, task_embedding_path: str) -> Stage2UtilityDataset:
    """Build the exact all-episode source specified by the immutable label manifest."""
    store = UtilityLabelStore(label_dir)
    manifest = store.manifest["dataset_manifest"]
    source = Stage2SourceDataset(
        shape_meta=manifest["shape_meta"], dataset_root=dataset_root,
        task_embedding_path=task_embedding_path, horizon=manifest["horizon"],
        n_obs_steps=manifest["n_obs_steps"], n_action_steps=manifest["n_action_steps"],
        max_train_episodes=None, dataset_version=manifest["dataset_version"],
        seed=manifest["seed"], source_frequency_hz=manifest["source_frequency_hz"],
    )
    return Stage2UtilityDataset(source, store)
