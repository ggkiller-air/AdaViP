"""Episode-indexed, train-normalized full-sensor ManiFeel datasets."""

from __future__ import annotations

import copy
import hashlib
from bisect import bisect_right
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from torch.utils.data import WeightedRandomSampler

from adavip.manifeel.action_adapter import pad_action
from adavip.manifeel.full_sensor_schema import (
    ACTION_CODEC, IMAGE_KEYS, SCHEMA_VERSION, SENSOR_KEYS, TACTILE_KEYS, validate_shape_meta,
)
from adavip.manifeel.task_embeddings import load_task_embeddings, validate_task_embedding_dim
from adavip.manifeel.task_protocol import coerce_task_specs
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class FullSensorManiFeelDataset(BaseImageDataset):
    """Expose one causal observation window per real timestep, without crossing episodes.

    The action window starts at t - n_obs_steps + 1, matching DP execution at
    action index n_obs_steps - 1. Repeated boundary values are explicitly masked.
    Validation and unused episodes never participate in fitted statistics.
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_root: str,
        task_embedding_path: str,
        task_specs: list[dict] | None = None,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        val_ratio: float = 0.02,
        max_train_episodes: int | None = 50,
        seed: int = 42,
        dataset_version: str = "unversioned",
        source_frequency_hz: float | None = None,
        decoded_cache_dir: str | None = None,
        decoded_cache_mode: str = "ram",
    ) -> None:
        super().__init__()
        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        validate_shape_meta(shape_meta)
        if n_obs_steps < 1 or n_action_steps < 1 or horizon < n_obs_steps + n_action_steps - 1:
            raise ValueError("horizon must contain observation history and all executed actions")
        if not 0 < val_ratio < 1:
            raise ValueError("An explicit trajectory validation split requires 0 < val_ratio < 1")
        if max_train_episodes is not None and max_train_episodes < 1:
            raise ValueError("max_train_episodes must be positive or null")
        if source_frequency_hz is not None and (not np.isfinite(source_frequency_hz) or source_frequency_hz <= 0):
            raise ValueError("source_frequency_hz must be positive or unknown (null)")
        self.shape_meta = copy.deepcopy(shape_meta)
        self.task_specs = coerce_task_specs(task_specs)
        if not self.task_specs or len({spec.task_id for spec in self.task_specs}) != len(self.task_specs):
            raise ValueError("Task IDs must be nonempty and unique")
        self.task_embeddings = load_task_embeddings(task_embedding_path, self.task_specs)
        validate_task_embedding_dim(shape_meta, self.task_embeddings)
        if not all(np.isfinite(value).all() for value in self.task_embeddings.values()):
            raise ValueError("Task embeddings must be finite")
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.seed = seed
        self.split = "train"
        self.groups: list[Any] = []
        self.episode_ends: list[np.ndarray] = []
        self.train_episodes: list[np.ndarray] = []
        self.val_episodes: list[np.ndarray] = []
        stores = []
        paths = [Path(dataset_root).expanduser().resolve() / spec.dataset for spec in self.task_specs]
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("Duplicate source stores require a shared trajectory split, not separate tasks")
        for task_index, (spec, path) in enumerate(zip(self.task_specs, paths)):
            if spec.action_dim not in (6, 7):
                raise ValueError(f"{spec.task_id}: only audited 6D/7D actions are supported")
            group = zarr.open_group(str(path), mode="r")
            ends = np.asarray(group["meta"]["episode_ends"][:], dtype=np.int64)
            if len(ends) < 2 or np.any(np.diff(np.r_[0, ends]) <= 0):
                raise ValueError(f"{spec.task_id}: need at least two nonempty episodes")
            metadata = {}
            for key in IMAGE_KEYS + ("state", "action"):
                array = group["data"][key]
                expected_tail = (spec.action_dim,) if key == "action" else (7,)
                if key in IMAGE_KEYS:
                    valid_shape = array.ndim == 4 and array.shape[-1] == 3
                else:
                    valid_shape = array.shape[1:] == expected_tail
                if not valid_shape or array.shape[0] != ends[-1] or array.dtype != np.dtype("float32"):
                    raise ValueError(f"{spec.task_id}/{key}: invalid shape, dtype or episode alignment")
                metadata[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            rng = np.random.default_rng(seed + task_index)
            n_val = min(max(1, round(len(ends) * val_ratio)), len(ends) - 1)
            val_ids = np.sort(rng.choice(len(ends), size=n_val, replace=False))
            train_ids = np.setdiff1d(np.arange(len(ends)), val_ids)
            if max_train_episodes is not None and len(train_ids) > max_train_episodes:
                train_ids = np.sort(rng.choice(train_ids, size=max_train_episodes, replace=False))
            self.groups.append(group)
            self.episode_ends.append(ends)
            self.train_episodes.append(train_ids)
            self.val_episodes.append(val_ids)
            stores.append({
                "task": asdict(spec), "path": str(path), "arrays": metadata,
                "episode_ends": ends.tolist(), "train_episodes": train_ids.tolist(),
                "val_episodes": val_ids.tolist(),
            })
        self.manifest = {
            "schema": SCHEMA_VERSION, "dataset_version": dataset_version,
            "source_frequency_hz": source_frequency_hz, "sensor_keys": list(SENSOR_KEYS),
            "action_codec": ACTION_CODEC, "shape_meta": self.shape_meta,
            "horizon": horizon, "n_obs_steps": n_obs_steps, "n_action_steps": n_action_steps,
            "seed": seed, "stores": stores,
            "task_embedding_sha256": hashlib.sha256(Path(task_embedding_path).read_bytes()).hexdigest(),
            "window_definition": "all real t; obs/action start=t-To+1; clipped padding has zero validity",
        }
        self._normalizer: LinearNormalizer | None = None
        self._normalizer_mode: str | None = None
        self._build_index()
        if decoded_cache_dir is not None:
            from adavip.manifeel.decoded_sensor_cache import load_decoded_cache

            self.groups = load_decoded_cache(self, decoded_cache_dir, decoded_cache_mode)

    def _build_index(self) -> None:
        self.records = []
        lengths = []
        self.task_lengths = [0] * len(self.task_specs)
        selected = self.train_episodes if self.split == "train" else self.val_episodes
        for task_index, episode_ids in enumerate(selected):
            ends = self.episode_ends[task_index]
            for episode_index in episode_ids:
                start = 0 if episode_index == 0 else int(ends[episode_index - 1])
                end = int(ends[episode_index])
                self.records.append((task_index, int(episode_index), start, end))
                lengths.append(end - start)
                self.task_lengths[task_index] += end - start
        self.cumulative_lengths = np.cumsum(lengths).tolist()

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _resolve_index(self, index: int) -> tuple[int, int, int, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect_right(self.cumulative_lengths, index)
        previous = 0 if record_index == 0 else self.cumulative_lengths[record_index - 1]
        return (*self.records[record_index], index - previous)

    def _read_images(self, array: Any, indices: np.ndarray, key: str) -> np.ndarray:
        first, last = int(indices.min()), int(indices.max())
        images = np.asarray(array[first:last + 1], dtype=np.float32)[indices - first]
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError(f"{key}: images must be finite float32 in [0,1]")
        height, width = self.shape_meta["obs"][key]["shape"][1:]
        if images.shape[1:3] != (height, width):
            images = np.stack([cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) for image in images])
        return np.ascontiguousarray(np.moveaxis(images, -1, 1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_index, episode_index, start, end, timestep = self._resolve_index(index)
        arrays = self.groups[task_index]["data"]
        anchor = start + timestep
        obs_indices = anchor - self.n_obs_steps + 1 + np.arange(self.n_obs_steps)
        action_indices = anchor - self.n_obs_steps + 1 + np.arange(self.horizon)
        obs_valid = (obs_indices >= start) & (obs_indices < end)
        action_valid = (action_indices >= start) & (action_indices < end)
        clipped_obs = np.clip(obs_indices, start, end - 1)
        observations = {
            key: torch.from_numpy(self._read_images(arrays[key], clipped_obs, key)) for key in IMAGE_KEYS
        }
        state = arrays["state"][int(clipped_obs.min()):int(clipped_obs.max()) + 1][clipped_obs - clipped_obs.min()]
        clipped_actions = np.clip(action_indices, start, end - 1)
        actions = arrays["action"][int(clipped_actions.min()):int(clipped_actions.max()) + 1][clipped_actions - clipped_actions.min()]
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite state or demonstration action")
        observations["state"] = torch.from_numpy(np.asarray(state, dtype=np.float32))
        embedding = self.task_embeddings[self.task_specs[task_index].task_id]
        observations["task_embedding"] = torch.from_numpy(np.repeat(embedding[None], self.n_obs_steps, axis=0))
        loss_mask = np.zeros((self.horizon, 7), dtype=np.float32)
        loss_mask[:, :self.task_specs[task_index].action_dim] = action_valid[:, None]
        return {
            "obs": observations, "action": torch.from_numpy(pad_action(actions)),
            "action_loss_mask": torch.from_numpy(loss_mask),
            "observation_valid_mask": torch.from_numpy(obs_valid),
            "action_valid_mask": torch.from_numpy(action_valid),
            "task_index": torch.tensor(task_index), "episode_index": torch.tensor(episode_index),
            "timestep": torch.tensor(timestep), "source_index": torch.tensor(anchor),
        }

    def get_validation_dataset(self) -> "FullSensorManiFeelDataset":
        """Return only the original held-out trajectories, never unused training data."""
        validation = copy.copy(self)
        validation.split = "validation"
        validation._build_index()
        return validation

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> LinearNormalizer:
        """Fit training trajectories only, excluding absent action dimensions."""
        if self._normalizer is not None and self._normalizer_mode == mode and not kwargs:
            return copy.deepcopy(self._normalizer)
        action_columns: list[list[np.ndarray]] = [[] for _ in range(7)]
        states = []
        for task_index, episode_ids in enumerate(self.train_episodes):
            arrays = self.groups[task_index]["data"]
            ends = self.episode_ends[task_index]
            for episode_index in episode_ids:
                start = 0 if episode_index == 0 else int(ends[episode_index - 1])
                end = int(ends[episode_index])
                actions = arrays["action"][start:end]
                states.append(arrays["state"][start:end])
                for dimension in range(actions.shape[-1]):
                    action_columns[dimension].append(actions[:, dimension])
        fields = []
        for columns in action_columns:
            values = np.concatenate(columns).reshape(-1, 1) if columns else np.zeros((2, 1), dtype=np.float32)
            if len(values) == 1:
                values = np.repeat(values, 2, axis=0)
            if not np.isfinite(values).all():
                raise ValueError("Training actions contain nonfinite values")
            fields.append(SingleFieldLinearNormalizer.create_fit(values, mode=mode, **kwargs))
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.cat([field.params_dict["scale"] for field in fields]),
            offset=torch.cat([field.params_dict["offset"] for field in fields]),
            input_stats_dict={
                name: torch.cat([field.get_input_stats()[name] for field in fields])
                for name in ("min", "max", "mean", "std")
            },
        )
        state_values = np.concatenate(states)
        if len(state_values) == 1:
            state_values = np.repeat(state_values, 2, axis=0)
        if not np.isfinite(state_values).all():
            raise ValueError("Training states contain nonfinite values")
        normalizer["state"] = SingleFieldLinearNormalizer.create_fit(state_values, mode=mode, **kwargs)
        for key in IMAGE_KEYS:
            normalizer[key] = get_image_range_normalizer()
        normalizer["task_embedding"] = SingleFieldLinearNormalizer.create_identity()
        normalizer.requires_grad_(False)
        if not kwargs:
            self._normalizer = copy.deepcopy(normalizer)
            self._normalizer_mode = mode
        return normalizer

    def make_task_balanced_sampler(self, generator: torch.Generator | None = None) -> WeightedRandomSampler:
        """Give each task equal expected mass at a fixed len(dataset) epoch budget."""
        weights = torch.cat([
            torch.full((length,), 1.0 / length, dtype=torch.double) for length in self.task_lengths
        ])
        return WeightedRandomSampler(weights, len(self), replacement=True, generator=generator)

    def get_future_tactile(self, index: int, offsets: list[int]) -> dict[str, Any]:
        """Read future targets separately; invalid episode-end targets are zero and masked."""
        if not offsets or any(offset <= 0 or offset > self.horizon - self.n_obs_steps for offset in offsets):
            raise ValueError("Future offsets must be positive and fit within the prediction window")
        task_index, _, start, end, timestep = self._resolve_index(index)
        indices = start + timestep + np.asarray(offsets, dtype=np.int64)
        valid = indices < end
        targets = {}
        for key in TACTILE_KEYS:
            values = np.zeros((len(offsets), *self.shape_meta["obs"][key]["shape"]), dtype=np.float32)
            if valid.any():
                values[valid] = self._read_images(self.groups[task_index]["data"][key], indices[valid], key)
            targets[key] = torch.from_numpy(values)
        return {"obs": targets, "valid_mask": torch.from_numpy(valid)}
