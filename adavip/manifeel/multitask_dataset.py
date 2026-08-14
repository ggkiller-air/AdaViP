"""Multi-task ManiFeel dataset adapter."""

from __future__ import annotations

import copy
import os
from bisect import bisect_right
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from adavip.manifeel.action_adapter import MULTITASK_ACTION_DIM, action_loss_mask, pad_action
from adavip.manifeel.task_embeddings import load_task_embeddings, validate_task_embedding_dim
from adavip.manifeel.task_protocol import ManiFeelTaskSpec, coerce_task_specs
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class _SingleTaskZarrDataset(BaseImageDataset):
    """Lazy ManiFeel Zarr dataset for one task."""

    def __init__(
        self,
        shape_meta: dict,
        zarr_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int = 2,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
    ) -> None:
        super().__init__()
        self.shape_meta = shape_meta
        self.zarr_path = zarr_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

        self.rgb_keys: list[str] = []
        self.lowdim_keys: list[str] = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                self.rgb_keys.append(key)
            elif obs_type == "low_dim":
                self.lowdim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        data_keys = self.rgb_keys + self.lowdim_keys + ["action"]
        key_first_k = {key: n_obs_steps for key in self.rgb_keys + self.lowdim_keys}
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)
        self.train_mask = train_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=data_keys,
            key_first_k=key_first_k,
            episode_mask=train_mask,
        )

    def get_validation_dataset(self) -> "_SingleTaskZarrDataset":
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.rgb_keys + self.lowdim_keys + ["action"],
            key_first_k={key: self.n_obs_steps for key in self.rgb_keys + self.lowdim_keys},
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample: dict[str, np.ndarray]) -> dict[str, Any]:
        obs_dict: dict[str, np.ndarray] = {}
        obs_slice = slice(self.n_obs_steps)
        for key in self.rgb_keys:
            image_seq = sample[key][obs_slice]
            target_shape = self.shape_meta["obs"][key]["shape"]
            target_h, target_w = target_shape[1], target_shape[2]
            if image_seq.shape[1:3] == (target_h, target_w):
                resized = image_seq.astype(np.float32, copy=False)
            else:
                resized = np.asarray(
                    [cv2.resize(image, (target_w, target_h)) for image in image_seq],
                    dtype=np.float32,
                )
            obs_dict[key] = np.moveaxis(resized, -1, 1)

        for key in self.lowdim_keys:
            obs_dict[key] = sample[key][obs_slice].astype(np.float32)

        return {
            "obs": obs_dict,
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        return dict_apply(data, torch.from_numpy)


class MultiTaskManiFeelDataset(BaseImageDataset):
    """Concatenate ManiFeel task datasets with unified 7D actions and masks."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_root: str,
        task_embedding_path: str,
        task_specs: list[dict] | None = None,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: int = 2,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
    ) -> None:
        super().__init__()
        self.shape_meta = shape_meta
        self.dataset_root = Path(dataset_root)
        self.task_specs = coerce_task_specs(task_specs)
        self.task_embeddings = load_task_embeddings(task_embedding_path, self.task_specs)
        validate_task_embedding_dim(self.shape_meta, self.task_embeddings)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes

        self.datasets: list[_SingleTaskZarrDataset] = []
        for spec in self.task_specs:
            self.datasets.append(
                _SingleTaskZarrDataset(
                    shape_meta=self._shape_meta_for_task(spec),
                    zarr_path=str(self._dataset_path(spec)),
                    horizon=horizon,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    n_obs_steps=n_obs_steps,
                    seed=seed,
                    val_ratio=val_ratio,
                    max_train_episodes=max_train_episodes,
                )
            )
        lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum(lengths).tolist()

    def _shape_meta_for_task(self, spec: ManiFeelTaskSpec) -> dict:
        shape_meta = copy.deepcopy(self.shape_meta)
        obs_meta = shape_meta["obs"]
        obs_meta.pop("task_embedding", None)
        shape_meta["action"] = {"shape": [spec.action_dim]}
        return shape_meta

    def _dataset_path(self, spec: ManiFeelTaskSpec) -> Path:
        path = self.dataset_root / spec.dataset
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing ManiFeel dataset for {spec.task_id}: {path}")
        return path

    def get_validation_dataset(self) -> "MultiTaskManiFeelDataset":
        val_set = copy.copy(self)
        val_set.datasets = [dataset.get_validation_dataset() for dataset in self.datasets]
        lengths = [len(dataset) for dataset in val_set.datasets]
        val_set.cumulative_lengths = np.cumsum(lengths).tolist()
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> LinearNormalizer:
        action_data = []
        state_data = []
        for dataset in self.datasets:
            action_data.append(pad_action(dataset.replay_buffer["action"][:]))
            state_data.append(dataset.replay_buffer["state"][:])

        data = {
            "action": np.concatenate(action_data, axis=0),
            "state": np.concatenate(state_data, axis=0),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        for key, attr in self.shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                normalizer[key] = get_image_range_normalizer()
        normalizer["task_embedding"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        if not self.cumulative_lengths:
            return 0
        return self.cumulative_lengths[-1]

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        task_index = bisect_right(self.cumulative_lengths, idx)
        previous = 0 if task_index == 0 else self.cumulative_lengths[task_index - 1]
        return task_index, idx - previous

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        task_index, local_idx = self._resolve_index(idx)
        spec = self.task_specs[task_index]
        data = self.datasets[task_index][local_idx]

        embedding = np.asarray(self.task_embeddings[spec.task_id], dtype=np.float32)
        task_embedding = np.repeat(embedding[None, :], self.n_obs_steps, axis=0)
        data["obs"]["task_embedding"] = torch.from_numpy(task_embedding)

        padded_action = pad_action(data["action"].numpy())
        data["action"] = torch.from_numpy(padded_action)
        data["action_loss_mask"] = torch.from_numpy(
            action_loss_mask(spec.action_dim, self.horizon, MULTITASK_ACTION_DIM)
        )
        data["task_index"] = torch.tensor(task_index, dtype=torch.long)
        data["task_action_dim"] = torch.tensor(spec.action_dim, dtype=torch.long)
        return dict_apply(data, lambda x: x if isinstance(x, torch.Tensor) else torch.as_tensor(x))
