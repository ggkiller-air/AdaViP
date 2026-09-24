"""ManiFeel dataset adapter for delta-frame Sparsh tactile observations."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from manifeel.dataset.manifeel_dataset_equidp import ManifeelDataset


class FilmFusionDataset(ManifeelDataset):
    """Add a six-channel current/past tactile pair without altering source Zarr."""

    def __init__(
        self,
        shape_meta: dict[str, Any],
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int = 2,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
        tactile_frame_delta: int = 5,
        tactile_source_key: str = "right_tactile_camera_taxim",
        tactile_pair_key: str = "right_tactile_camera_taxim_pair",
    ) -> None:
        if tactile_frame_delta < 0:
            raise ValueError("tactile_frame_delta cannot be negative")
        if n_obs_steps is None or n_obs_steps < 1:
            raise ValueError("film_fusion requires at least one observation step")
        target_shape_meta = copy.deepcopy(shape_meta)
        pair_meta = target_shape_meta.get("obs", {}).get(tactile_pair_key)
        if pair_meta is None or tuple(pair_meta.get("shape", ()))[:1] != (6,):
            raise ValueError(f"{tactile_pair_key} must be configured as a six-channel image")

        source_shape_meta = copy.deepcopy(target_shape_meta)
        source_shape_meta["obs"].pop(tactile_pair_key)
        super().__init__(
            shape_meta=source_shape_meta,
            zarr_path=zarr_path,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            n_obs_steps=n_obs_steps,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )
        self.shape_meta = target_shape_meta
        self.tactile_frame_delta = int(tactile_frame_delta)
        self.tactile_source_key = tactile_source_key
        self.tactile_pair_key = tactile_pair_key
        self.zarr_path = str(Path(zarr_path).expanduser())
        root = zarr.open(self.zarr_path, mode="r")
        if tactile_source_key not in root["data"]:
            raise KeyError(f"tactile source is absent from dataset: {tactile_source_key}")
        self._tactile_array = root["data"][tactile_source_key]
        self._episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        self._episode_starts = np.concatenate(
            (np.zeros(1, dtype=np.int64), self._episode_ends[:-1])
        )

    def get_normalizer(self, mode: str = "limits", **kwargs: Any):
        """Preserve [0, 1] image values expected by Piper preprocessing."""
        normalizer = super().get_normalizer(mode=mode, **kwargs)
        for key in (*self.rgb_keys, self.tactile_pair_key):
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def _observation_indices(self, idx: int) -> np.ndarray:
        buffer_start, buffer_end, sample_start, sample_end = (
            int(value) for value in self.sampler.indices[idx]
        )
        positions = np.arange(self.n_obs_steps, dtype=np.int64)
        indices = buffer_start + positions - sample_start
        return np.clip(indices, buffer_start, buffer_end - 1)

    def _tactile_pair(self, idx: int) -> np.ndarray:
        current_indices = self._observation_indices(idx)
        episodes = np.searchsorted(self._episode_ends, current_indices, side="right")
        episode_starts = self._episode_starts[episodes]
        past_indices = np.maximum(
            current_indices - self.tactile_frame_delta,
            episode_starts,
        )
        current = np.asarray(self._tactile_array.get_orthogonal_selection((current_indices,)))
        past = np.asarray(self._tactile_array.get_orthogonal_selection((past_indices,)))
        pair = np.concatenate((current, past), axis=-1).astype(np.float32, copy=False)
        pair = np.moveaxis(pair, -1, 1)
        expected = tuple(self.shape_meta["obs"][self.tactile_pair_key]["shape"])
        if tuple(pair.shape[1:]) != expected:
            raise ValueError(
                f"derived tactile pair shape {tuple(pair.shape[1:])} does not match {expected}"
            )
        return pair

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        data["obs"][self.tactile_pair_key] = self._tactile_pair(idx)
        return dict_apply(data, torch.from_numpy)
