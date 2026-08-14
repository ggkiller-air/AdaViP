"""Deployment-free image dataset for real-world policy training."""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.common.normalize_util import get_image_range_normalizer
from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class OfflineImageDataset(BaseImageDataset):
    """Load image, low-dimensional state, and action arrays from RDP Zarr.

    Unlike the upstream real-robot dataset, this class deliberately has no ROS,
    robot-frame transform, sensor publisher, or deployment dependency.
    """

    def __init__(
        self,
        shape_meta: dict[str, Any],
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int | None = None,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
    ) -> None:
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

        self.shape_meta = shape_meta
        self.rgb_keys: list[str] = []
        self.lowdim_keys: list[str] = []
        for key, attributes in shape_meta["obs"].items():
            obs_type = attributes.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported observation type {obs_type!r} for {key}")

        load_keys = [*self.rgb_keys, *self.lowdim_keys, "action"]
        replay_buffer = ReplayBuffer.copy_from_path(
            os.path.join(dataset_path, "replay_buffer.zarr"), keys=load_keys
        )
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = downsample_mask(
            mask=~val_mask, max_n=max_train_episodes, seed=seed
        )
        self.replay_buffer = replay_buffer
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            keys=["action"],
        )
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask

    def get_validation_dataset(self) -> "OfflineImageDataset":
        """Return a shallow copy sampling only validation episodes."""
        validation = copy.copy(self)
        validation.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        validation.val_mask = ~self.val_mask
        return validation

    def get_normalizer(self, **kwargs: Any) -> LinearNormalizer:
        """Fit generic per-dimension limit normalizers for joints and actions."""
        normalizer = LinearNormalizer()
        action_dim = self.shape_meta["action"]["shape"][0]
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"][:, :action_dim]
        )
        for key in self.lowdim_keys:
            obs_dim = self.shape_meta["obs"][key]["shape"][0]
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key][:, :obs_dim]
            )
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return every action in the replay buffer."""
        action_dim = self.shape_meta["action"]["shape"][0]
        return torch.from_numpy(self.replay_buffer["action"][:, :action_dim])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> dict[str, Any]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(index)
        buffer_start, buffer_end, sample_start, _ = self.sampler.indices[index]
        observation_steps = self.n_obs_steps or self.horizon
        obs_indices = np.arange(observation_steps) - sample_start + buffer_start
        obs_indices = np.clip(obs_indices, buffer_start, buffer_end - 1)

        obs: dict[str, torch.Tensor] = {}
        for key in self.rgb_keys:
            images = np.moveaxis(self.replay_buffer[key][obs_indices], -1, 1).astype(
                np.float32
            )
            obs[key] = torch.from_numpy(images / 255.0)
        for key in self.lowdim_keys:
            obs_dim = self.shape_meta["obs"][key]["shape"][0]
            values = self.replay_buffer[key][obs_indices, :obs_dim].astype(np.float32)
            obs[key] = torch.from_numpy(values)

        action_dim = self.shape_meta["action"]["shape"][0]
        action = torch.from_numpy(data["action"][:, :action_dim].astype(np.float32))
        return {"obs": obs, "action": action, "extended_obs": {}}
