"""Verified label joins, episode splits and train-only Utility normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from adavip.manifeel.full_sensor_dataset import FullSensorManiFeelDataset
from adavip.manifeel.utility_label_store import complete_window_index, file_hash


def trajectory_split(index: np.ndarray, seed: int = 20261022, validation_ratio: float = 0.2) -> dict:
    """Split whole episodes within each task; no neighboring-window leakage."""
    if not 0 < validation_ratio < 1:
        raise ValueError("Validation ratio must lie in (0,1)")
    generator = np.random.default_rng(seed)
    validation = np.zeros(len(index), dtype=bool)
    episodes = {}
    for task_index in np.unique(index["task_index"]):
        selected = index["task_index"] == task_index
        identifiers = np.unique(index["episode_index"][selected])
        if len(identifiers) < 2:
            raise ValueError("Every task needs at least two labeled episodes")
        shuffled = generator.permutation(identifiers)
        count = min(len(identifiers) - 1, max(1, round(len(identifiers) * validation_ratio)))
        held = np.sort(shuffled[:count])
        validation |= selected & np.isin(index["episode_index"], held)
        episodes[str(int(task_index))] = {
            "train": np.sort(shuffled[count:]).tolist(), "validation": held.tolist(),
        }
    return {"seed": seed, "validation_ratio": validation_ratio, "episodes": episodes,
            "train_rows": np.flatnonzero(~validation), "validation_rows": np.flatnonzero(validation)}


def fit_utility_statistics(utility: np.ndarray, train_rows: np.ndarray, floor: float = 1e-6) -> dict:
    """Fit pooled per-dimension population statistics on training trajectories only."""
    if floor <= 0 or len(train_rows) == 0:
        raise ValueError("Need training labels and a positive scale floor")
    values = np.asarray(utility[train_rows], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError("Utility training labels must be finite [N,9]")
    return {"mean": values.mean(axis=0).tolist(), "scale": np.maximum(values.std(axis=0), floor).tolist(),
            "floor": floor, "ddof": 0, "train_windows": len(train_rows),
            "train_rows_sha256": hashlib.sha256(np.asarray(train_rows, dtype="<i8").tobytes()).hexdigest(),
            "fit_scope": "pooled training trajectories only; never per-task or validation"}


class UtilityLabelStore:
    """Read immutable final arrays and reject index, normalization or hash mismatches."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.completion = json.loads((self.root / "COMPLETE.json").read_text())
        unsigned = {key: value for key, value in self.manifest.items() if key != "sha256"}
        fingerprint = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if fingerprint != self.manifest["sha256"] or fingerprint != self.completion["contract_sha256"]:
            raise ValueError("Label manifest fingerprint mismatch")
        if self.completion["status"] != "complete":
            raise ValueError("Utility label store is incomplete")
        for name in ("window_index.npy", "task_id.npy", "utility.npy", "utility_variance.npy", "weighted_losses.npy"):
            expected = self.completion["files"][name]
            path = self.root / name
            if path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
                raise ValueError(f"Utility label file hash mismatch: {name}")
        self.index = np.load(self.root / "window_index.npy", mmap_mode="r", allow_pickle=False)
        self.task_ids = np.load(self.root / "task_id.npy", mmap_mode="r", allow_pickle=False)
        self.utility = np.load(self.root / "utility.npy", mmap_mode="r", allow_pickle=False)
        self.variance = np.load(self.root / "utility_variance.npy", mmap_mode="r", allow_pickle=False)
        losses = np.load(self.root / "weighted_losses.npy", mmap_mode="r", allow_pickle=False)
        expected_index = complete_window_index(self.manifest["dataset_manifest"])
        np.testing.assert_array_equal(self.index, expected_index)
        count = len(self.index)
        if self.completion["windows"] != count or self.manifest["windows"] != count:
            raise ValueError("Wrong label count")
        if self.utility.shape != (count, 9) or self.variance.shape != (count, 9) or losses.shape != (count, 10):
            raise ValueError("Wrong label/loss shapes")
        if not np.isfinite(self.utility).all() or not np.isfinite(self.variance).all() or (self.variance < 0).any():
            raise ValueError("Invalid utility or variance")
        np.testing.assert_allclose(self.utility, losses[:, 1:] - losses[:, :1], rtol=1e-12, atol=1e-14)
        for task in self.manifest["tasks"]:
            selected = self.index["task_index"] == task["task_index"]
            if not np.all(self.task_ids[selected] == task["task_id"]):
                raise ValueError("Task name/index mismatch")


def align_source_to_labels(source: FullSensorManiFeelDataset, store: UtilityLabelStore) -> None:
    """Rebuild only the source index; never call the dataset Normalizer fitter."""
    expected = store.manifest["dataset_manifest"]
    if source._normalizer is not None:
        raise ValueError("Source must not refit Teacher normalization")
    for key in ("shape_meta", "horizon", "n_obs_steps", "n_action_steps", "task_embedding_sha256"):
        if source.manifest[key] != expected[key]:
            raise ValueError(f"Dataset protocol mismatch: {key}")
    if len(source.task_specs) != len(expected["stores"]):
        raise ValueError("Dataset task count mismatch")
    for task_index, (actual, frozen) in enumerate(zip(source.manifest["stores"], expected["stores"])):
        for key in actual:
            if key not in ("train_episodes", "val_episodes", "path") and actual[key] != frozen[key]:
                raise ValueError(f"Dataset store mismatch: task={task_index}, key={key}")
        source.train_episodes[task_index] = np.arange(len(source.episode_ends[task_index]))
        source.val_episodes[task_index] = np.empty(0, dtype=np.int64)
    source._build_index()
    for row in store.index:
        task_index, episode_index, start, end, timestep = source._resolve_index(int(row["dataset_index"]))
        if (task_index, episode_index, timestep, start + timestep, end - start) != tuple(
            int(row[key]) for key in ("task_index", "episode_index", "timestep", "source_index", "episode_length")
        ):
            raise ValueError("Label/source index mismatch")


class Stage1UtilityDataset(Dataset):
    """Join raw signed labels and t+1..t+4 tactile targets to verified source windows."""

    def __init__(self, source: FullSensorManiFeelDataset, store: UtilityLabelStore, rows: np.ndarray) -> None:
        self.source = source
        self.store = store
        self.rows = np.asarray(rows, dtype=np.int64)
        if np.any(self.rows < 0) or np.any(self.rows >= len(store.index)):
            raise ValueError("Unknown utility row")

    def __len__(self) -> int:
        return len(self.rows)

    def make_task_balanced_sampler(self, generator: torch.Generator | None = None) -> WeightedRandomSampler:
        """Assign equal expected task mass within this split without dropping windows."""
        tasks = self.store.index["task_index"][self.rows]
        if not len(tasks):
            raise ValueError("Cannot sample an empty Stage-1 split")
        weights = np.zeros(len(tasks), dtype=np.float64)
        for task_index in np.unique(tasks):
            selected = tasks == task_index
            weights[selected] = 1.0 / selected.sum()
        return WeightedRandomSampler(torch.from_numpy(weights), len(self), replacement=True, generator=generator)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = int(self.rows[index])
        row = self.store.index[row_index]
        source_index = int(row["dataset_index"])
        sample = self.source[source_index]
        for key in ("task_index", "episode_index", "timestep", "source_index"):
            if int(sample[key]) != int(row[key]):
                raise ValueError(f"Decoded sample/index mismatch: {key}")
        if not sample["observation_valid_mask"].all() or not sample["action_valid_mask"].all():
            raise ValueError("Labeled window unexpectedly crosses an episode boundary")
        future = self.source.get_future_tactile(source_index, [1, 2, 3, 4])
        sample.update({"window_id": torch.tensor(int(row["window_id"])),
                       "utility": torch.from_numpy(self.store.utility[row_index].astype(np.float32)),
                       "utility_variance": torch.from_numpy(self.store.variance[row_index].astype(np.float32)),
                       "future_tactile": future["obs"], "future_valid_mask": future["valid_mask"]})
        return sample
