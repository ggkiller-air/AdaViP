"""Offline Piper datasets for RDP action-tokenizer and latent-policy training."""

from __future__ import annotations

import copy
import os
from typing import Any

import einops
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from reactive_diffusion_policy.common.normalize_util import get_image_range_normalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from reactive_diffusion_policy.model.vae.model import VAE


class PiperRdpDataset(BaseImageDataset):
    """Sample 18.75 Hz Piper joint actions and full-rate tactile conditioning."""

    def __init__(
        self,
        shape_meta: dict[str, Any],
        dataset_path: str,
        horizon: int = 29,
        pad_before: int = 3,
        pad_after: int = 25,
        n_obs_steps: int | None = 4,
        obs_temporal_downsample_ratio: int = 2,
        n_latency_steps: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: int | None = None,
    ) -> None:
        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
        if obs_temporal_downsample_ratio < 1:
            raise ValueError("obs_temporal_downsample_ratio must be positive")
        if n_obs_steps is not None and n_obs_steps % obs_temporal_downsample_ratio:
            raise ValueError(
                "n_obs_steps must be divisible by obs_temporal_downsample_ratio"
            )

        self.shape_meta = shape_meta
        self.rgb_keys, self.lowdim_keys = self._split_keys(shape_meta["obs"])
        self.extended_rgb_keys, self.extended_lowdim_keys = self._split_keys(
            shape_meta.get("extended_obs", {})
        )
        load_keys = list(
            dict.fromkeys(
                self.rgb_keys
                + self.lowdim_keys
                + self.extended_rgb_keys
                + self.extended_lowdim_keys
                + ["action"]
            )
        )
        replay_buffer = ReplayBuffer.copy_from_path(
            os.path.join(dataset_path, "replay_buffer.zarr"), keys=load_keys
        )
        key_first_k: dict[str, int] = {}
        if n_obs_steps is not None:
            extended_keys = set(self.extended_rgb_keys + self.extended_lowdim_keys)
            for key in self.rgb_keys + self.lowdim_keys:
                if key not in extended_keys:
                    key_first_k[key] = n_obs_steps

        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.n_latency_steps = n_latency_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
        train_mask = downsample_mask(~val_mask, max_train_episodes, seed)
        self.sampler = self._make_sampler(train_mask)
        self.val_mask = val_mask

    @staticmethod
    def _split_keys(shape_meta: dict[str, Any]) -> tuple[list[str], list[str]]:
        rgb_keys: list[str] = []
        lowdim_keys: list[str] = []
        for key, attributes in shape_meta.items():
            observation_type = attributes.get("type", "low_dim")
            if observation_type == "rgb":
                rgb_keys.append(key)
            elif observation_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise ValueError(
                    f"Unsupported observation type {observation_type!r} for {key}"
                )
        return rgb_keys, lowdim_keys

    def _make_sampler(self, episode_mask: np.ndarray) -> SequenceSampler:
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=episode_mask,
            key_first_k=self.key_first_k,
        )

    def get_validation_dataset(self) -> "PiperRdpDataset":
        """Return a shallow copy sampling validation episodes only."""
        validation = copy.copy(self)
        validation.sampler = validation._make_sampler(self.val_mask)
        validation.val_mask = ~self.val_mask
        return validation

    def get_normalizer(self, **kwargs: Any) -> LinearNormalizer:
        """Fit independent range normalizers for all joint and tactile fields."""
        del kwargs
        normalizer = LinearNormalizer()
        action_dim = int(self.shape_meta["action"]["shape"][0])
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"][:, :action_dim]
        )
        for key in sorted(set(self.lowdim_keys + self.extended_lowdim_keys)):
            if key in self.shape_meta["obs"]:
                metadata = self.shape_meta["obs"][key]
            else:
                metadata = self.shape_meta["extended_obs"][key]
            dimension = int(metadata["shape"][0])
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key][:, :dimension]
            )
        for key in sorted(set(self.rgb_keys + self.extended_rgb_keys)):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return all absolute 7D joint and gripper actions."""
        action_dim = int(self.shape_meta["action"]["shape"][0])
        return torch.from_numpy(self.replay_buffer["action"][:, :action_dim])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> dict[str, Any]:
        threadpool_limits(1)
        # Upstream's first-k optimization fills unloaded uint8 image slots with
        # NaN before slicing them away, which emits a harmless NumPy cast warning.
        with np.errstate(invalid="ignore"):
            data = self.sampler.sample_sequence(index)
        observation_slice = slice(self.n_obs_steps)
        ratio = self.obs_temporal_downsample_ratio

        obs: dict[str, torch.Tensor] = {}
        for key in self.rgb_keys:
            values = data[key][observation_slice][::-ratio][::-1]
            obs[key] = torch.from_numpy(
                np.moveaxis(values, -1, 1).astype(np.float32) / 255.0
            )
        for key in self.lowdim_keys:
            dimension = int(self.shape_meta["obs"][key]["shape"][0])
            values = data[key][:, :dimension][observation_slice][::-ratio][::-1]
            obs[key] = torch.from_numpy(values.astype(np.float32))

        extended_obs: dict[str, torch.Tensor] = {}
        for key in self.extended_rgb_keys:
            extended_obs[key] = torch.from_numpy(
                np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0
            )
        for key in self.extended_lowdim_keys:
            dimension = int(self.shape_meta["extended_obs"][key]["shape"][0])
            extended_obs[key] = torch.from_numpy(
                data[key][:, :dimension].astype(np.float32)
            )

        action_dim = int(self.shape_meta["action"]["shape"][0])
        action = data["action"][:, :action_dim].astype(np.float32)
        if self.n_latency_steps:
            action = action[self.n_latency_steps :]
        return {
            "obs": obs,
            "action": torch.from_numpy(action),
            "extended_obs": extended_obs,
        }


class PiperRdpLatentDataset(PiperRdpDataset):
    """Piper dataset that also fits an AT latent-action normalizer for LDP."""

    def __init__(
        self,
        at: VAE,
        use_latent_action_before_vq: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.at = at
        self.at.eval()
        self.use_latent_action_before_vq = use_latent_action_before_vq

    def get_normalizer(self, **kwargs: Any) -> LinearNormalizer:
        """Fit observation/action normalizers and encoded latent-action bounds."""
        normalizer = super().get_normalizer(**kwargs)
        latent_actions = []
        with torch.no_grad():
            for data in tqdm(
                self, leave=False, desc="Calculating latent action normalizer"
            ):
                action = data["action"].to(self.at.device).unsqueeze(0)
                normalized_action = normalizer["action"].normalize(action)
                latent = self.at.encoder(
                    self.at.preprocess(normalized_action / self.at.act_scale)
                )
                if self.at.use_vq:
                    if not self.use_latent_action_before_vq:
                        latent, _, _ = self.at.quant_state_with_vq(latent)
                    elif self.at.use_conv_encoder:
                        latent = einops.rearrange(latent, "N T A -> N (T A)")
                else:
                    latent, _ = self.at.quant_state_without_vq(latent)
                latent = einops.rearrange(
                    latent,
                    "N (T A) -> N T A",
                    T=self.at.downsampled_input_h,
                )
                latent_actions.append(latent[0].detach().cpu().numpy())
        all_latent_actions = np.concatenate(latent_actions, axis=0)
        normalizer["latent_action"] = SingleFieldLinearNormalizer.create_fit(
            all_latent_actions
        )
        return normalizer
