"""Episode-safe six-frame windows for the Stage 1 v2 Teacher."""

from __future__ import annotations

import copy
import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adavip.manifeel.action_adapter import pad_action
from adavip.manifeel.full_sensor_dataset import FullSensorManiFeelDataset
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS
from adavip.manifeel.stage1_v2_schema import (
    ACTION_HISTORY_STEPS,
    ACTION_HORIZON,
    FUTURE_STEPS,
    OBS_HISTORY_STEPS,
    SCHEMA_VERSION,
    validate_stage1_shape_meta,
)


class Stage1V2ManiFeelDataset(FullSensorManiFeelDataset):
    """Expose anchors with six causal observations and a t-1 action prefix.

    For an anchor t, observations are [t-5, ..., t], the action target is
    [a_(t-1), ..., a_(t+14)], and the future auxiliary targets are at
    [t+1, ..., t+4]. No index is allowed to cross an episode boundary.
    """

    def __init__(self, *args: Any, all_episodes: bool = True, **kwargs: Any) -> None:
        shape_meta = kwargs.get("shape_meta")
        if shape_meta is None and args:
            shape_meta = args[0]
        validate_stage1_shape_meta(shape_meta)
        kwargs.setdefault("horizon", ACTION_HORIZON)
        kwargs.setdefault("n_obs_steps", OBS_HISTORY_STEPS)
        kwargs.setdefault("n_action_steps", 8)
        if kwargs["horizon"] != ACTION_HORIZON or kwargs["n_obs_steps"] != OBS_HISTORY_STEPS:
            raise ValueError("Stage 1 v2 requires horizon=16 and n_obs_steps=6")
        self.all_episodes = all_episodes
        decoded_cache_dir = kwargs.pop("decoded_cache_dir", None)
        decoded_cache_mode = kwargs.pop("decoded_cache_mode", "mmap")
        super().__init__(*args, **kwargs)
        if all_episodes:
            self.train_episodes = [np.arange(len(ends)) for ends in self.episode_ends]
            self.val_episodes = [np.empty(0, dtype=np.int64) for _ in self.episode_ends]
            for store, episodes in zip(self.manifest["stores"], self.train_episodes):
                store["train_episodes"] = episodes.tolist()
                store["val_episodes"] = []
            self._build_index()
        self.manifest["schema"] = SCHEMA_VERSION
        self.manifest["window_definition"] = (
            "anchor t; observations [t-5,t]; actions [a_(t-1),a_(t+14)]; "
            "future targets [t+1,t+4]"
        )
        self.manifest["window_definition"] += "; all history/action/future indices valid; no padding"
        if decoded_cache_dir is not None:
            self._load_compatible_cache(decoded_cache_dir, decoded_cache_mode)

    def _load_compatible_cache(self, directory: str, mode: str) -> None:
        from adavip.manifeel.decoded_sensor_cache import load_decoded_cache

        contract = json.loads((Path(directory) / "contract.json").read_text())
        cached_manifest = contract["manifest"]
        for key in ("shape_meta", "task_embedding_sha256", "action_codec"):
            if self.manifest[key] != cached_manifest[key]:
                raise ValueError(f"Decoded cache changes Stage 1 v2 {key}")
        if len(self.manifest["stores"]) != len(cached_manifest["stores"]):
            raise ValueError("Decoded cache task count differs")
        for current, cached in zip(self.manifest["stores"], cached_manifest["stores"]):
            for key in ("task", "path", "arrays", "episode_ends"):
                if current[key] != cached[key]:
                    raise ValueError(f"Decoded cache source differs: {key}")
            selected = sorted(set(cached["train_episodes"] + cached["val_episodes"]))
            if selected != list(range(len(current["episode_ends"]))):
                raise ValueError("Reused decoded cache must contain every demonstration")
        proxy = copy.copy(self)
        proxy.manifest = cached_manifest
        proxy.train_episodes = [np.asarray(store["train_episodes"]) for store in cached_manifest["stores"]]
        proxy.val_episodes = [np.asarray(store["val_episodes"]) for store in cached_manifest["stores"]]
        self.groups = load_decoded_cache(proxy, directory, mode)
        self.manifest["decoded_cache_contract"] = contract["schema"]

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
                valid_length = max(0, end - start - ACTION_HISTORY_STEPS - (ACTION_HORIZON - 2))
                if valid_length == 0:
                    continue
                self.records.append((task_index, int(episode_index), start, end))
                lengths.append(valid_length)
                self.task_lengths[task_index] += valid_length
        self.cumulative_lengths = np.cumsum(lengths).tolist()
        if not self.records and self.split == "train":
            raise ValueError("No episode contains a valid six-frame Stage 1 v2 window")

    def _resolve_index(self, index: int) -> tuple[int, int, int, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect_right(self.cumulative_lengths, index)
        previous = 0 if record_index == 0 else self.cumulative_lengths[record_index - 1]
        return (*self.records[record_index], index - previous)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._get_sample(index)

    def _image_tensor(
        self, array: Any, indices: np.ndarray, key: str, destination: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self._read_images(array, indices, key)
        if destination is None:
            return torch.from_numpy(values)
        np.copyto(destination.numpy(), values, casting="no")
        return destination

    def _get_sample(
        self, index: int, image_buffers: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        task_index, episode_index, start, end, offset = self._resolve_index(index)
        arrays = self.groups[task_index]["data"]
        anchor = start + ACTION_HISTORY_STEPS + offset
        observation_indices = anchor - OBS_HISTORY_STEPS + 1 + np.arange(OBS_HISTORY_STEPS)
        action_indices = anchor - 1 + np.arange(ACTION_HORIZON)
        history_indices = anchor - ACTION_HISTORY_STEPS + np.arange(ACTION_HISTORY_STEPS)
        future_indices = anchor + np.arange(1, FUTURE_STEPS + 1)

        observation_valid = (observation_indices >= start) & (observation_indices < end)
        action_valid = (action_indices >= start) & (action_indices < end)
        history_valid = (history_indices >= start) & (history_indices < end)
        future_valid = (future_indices >= start) & (future_indices < end)
        if not all(mask.all() for mask in (observation_valid, action_valid, history_valid, future_valid)):
            raise RuntimeError("Stage 1 v2 index construction produced an invalid history")

        clipped_observations = np.clip(observation_indices, start, end - 1)
        observations = {
            key: self._image_tensor(
                arrays[key], clipped_observations, key,
                None if image_buffers is None else image_buffers["obs"][key],
            )
            for key in IMAGE_KEYS
        }
        state_values = arrays["state"][int(clipped_observations[0]):int(clipped_observations[-1]) + 1]
        clipped_actions = np.clip(action_indices, start, end - 1)
        clipped_history = np.clip(history_indices, start, end - 1)
        actions = arrays["action"][int(clipped_actions[0]):int(clipped_actions[-1]) + 1]
        action_history = arrays["action"][int(clipped_history[0]):int(clipped_history[-1]) + 1]
        if not np.isfinite(state_values).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite state or demonstration action")
        observations["state"] = torch.from_numpy(np.asarray(state_values, dtype=np.float32))
        embedding = self.task_embeddings[self.task_specs[task_index].task_id]
        observations["task_embedding"] = torch.from_numpy(
            np.repeat(embedding[None], OBS_HISTORY_STEPS, axis=0)
        )

        future_observations = {}
        clipped_future = np.clip(future_indices, start, end - 1)
        for key in IMAGE_KEYS:
            future_observations[key] = self._image_tensor(
                arrays[key], clipped_future, key,
                None if image_buffers is None else image_buffers["future_obs"][key],
            )

        return {
            "obs": observations,
            "future_obs": future_observations,
            "action": torch.from_numpy(pad_action(actions)),
            "action_history": torch.from_numpy(pad_action(action_history)),
            "action_loss_mask": torch.from_numpy(
                np.repeat(action_valid[:, None], 7, axis=1)
                * np.concatenate(
                    [
                        np.ones((1, self.task_specs[task_index].action_dim), dtype=np.float32),
                        np.zeros((1, 7 - self.task_specs[task_index].action_dim), dtype=np.float32),
                    ],
                    axis=1,
                )
            ),
            "action_history_valid_mask": torch.from_numpy(history_valid),
            "action_dimension_mask": torch.tensor(
                [1.0] * self.task_specs[task_index].action_dim
                + [0.0] * (7 - self.task_specs[task_index].action_dim),
            ),
            "future_valid_mask": torch.from_numpy(future_valid),
            "observation_valid_mask": torch.from_numpy(observation_valid),
            "action_valid_mask": torch.from_numpy(action_valid),
            "task_index": torch.tensor(task_index, dtype=torch.long),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "timestep": torch.tensor(anchor - start, dtype=torch.long),
            "source_index": torch.tensor(anchor, dtype=torch.long),
        }

    def get_validation_dataset(self) -> "Stage1V2ManiFeelDataset":
        """Return only held-out trajectories with the same v2 window contract."""
        validation = copy.copy(self)
        validation.split = "validation"
        validation._build_index()
        return validation
