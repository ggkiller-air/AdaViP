"""Single-task sensor-dropout DP with explicit state tokenization."""

from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from torch import Tensor, nn

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from manifeel.dataset.manifeel_dataset_equidp import ManifeelDataset


def sensor_conditions(image_keys: tuple[str, ...]) -> tuple[tuple[str, ...], Tensor]:
    """Make unique single-image and, for five inputs, modality-group removals."""
    names = ["full", *image_keys]
    masks = [torch.ones(len(image_keys), dtype=torch.bool)]
    for index in range(len(image_keys)):
        mask = masks[0].clone()
        mask[index] = False
        masks.append(mask)
    if len(image_keys) == 5:
        for name, removed in (("vision", (0, 1, 2)), ("tactile", (3, 4))):
            mask = masks[0].clone()
            mask[list(removed)] = False
            names.append(name)
            masks.append(mask)
    return tuple(names), torch.stack(masks)


class SingleTaskTeacherEncoder(nn.Module):
    """Encode independent image histories and robot state before simple fusion."""

    def __init__(self, shape_meta: dict, n_obs_steps: int = 2, token_dim: int = 256,
                 rgb_model: nn.Module | None = None, image_feature_dim: int = 512) -> None:
        super().__init__()
        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.shape_meta = copy.deepcopy(shape_meta)
        self.image_keys = tuple(key for key, item in shape_meta["obs"].items() if item["type"] == "rgb")
        valid_five = ("front", "side", "wrist", "left_tactile_camera_taxim", "right_tactile_camera_taxim")
        if self.image_keys not in (valid_five, ("wrist", "right_tactile_camera_taxim")):
            raise ValueError("Expected ordered five-sensor or wrist/right-TacRGB inputs")
        if set(shape_meta["obs"]) != {*self.image_keys, "state"}:
            raise ValueError("State is required; language and extra inputs are unsupported")
        self.n_obs_steps, self.token_dim = n_obs_steps, token_dim
        self.image_feature_dim = image_feature_dim
        backbone = get_resnet("resnet18", weights=None) if rgb_model is None else copy.deepcopy(rgb_model)
        backbone = replace_submodules(backbone, lambda module: isinstance(module, nn.BatchNorm2d),
                                     lambda module: nn.GroupNorm(module.num_features // 16, module.num_features))
        self.image_encoders = nn.ModuleDict({key: copy.deepcopy(backbone) for key in self.image_keys})
        self.temporal_projections = nn.ModuleDict({key: nn.Sequential(
            nn.Linear(n_obs_steps * image_feature_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
        ) for key in self.image_keys})
        self.state_projection = nn.Sequential(nn.Linear(n_obs_steps * 7, token_dim),
                                              nn.LayerNorm(token_dim), nn.SiLU())
        self.sensor_identity = nn.Parameter(torch.randn(len(self.image_keys) + 1, token_dim) * 0.02)
        self.missing_tokens = nn.Parameter(torch.zeros(len(self.image_keys), token_dim))
        self.fusion = nn.Sequential(nn.Linear((len(self.image_keys) + 1) * token_dim + len(self.image_keys), token_dim),
                                    nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode_tokens(self, observations: dict[str, Tensor]) -> Tensor:
        """Return image-history tokens followed by the always-present state token."""
        batch_size = observations["state"].shape[0]
        if set(observations) != {*self.image_keys, "state"}:
            raise ValueError("Observation keys differ from the training contract")
        tokens = []
        for key in self.image_keys:
            images = observations[key]
            if tuple(images.shape) != (batch_size, self.n_obs_steps, *self.shape_meta["obs"][key]["shape"]):
                raise ValueError(f"Invalid image history for {key}")
            # Match the pinned official pipeline: range-normalizer then ImageNet transform.
            images = (images.flatten(0, 1) - self.image_mean) / self.image_std
            features = self.image_encoders[key](images)
            tokens.append(self.temporal_projections[key](features.reshape(batch_size, -1)))
        state = observations["state"]
        if tuple(state.shape) != (batch_size, self.n_obs_steps, 7):
            raise ValueError("state must have shape [B,To,7]")
        tokens.append(self.state_projection(state.flatten(1)))
        return torch.stack(tokens, 1) + self.sensor_identity

    def fuse_tokens(self, tokens: Tensor, presence: Tensor) -> Tensor:
        """Mask only image tokens, preserving state and explicit presence indicators."""
        if presence.shape != tokens[:, :-1, 0].shape or presence.dtype != torch.bool:
            raise ValueError("presence must be boolean [B,number_of_images]")
        missing = self.missing_tokens + self.sensor_identity[:-1]
        images = torch.where(presence.unsqueeze(-1), tokens[:, :-1], missing)
        features = torch.cat((images, tokens[:, -1:]), 1).flatten(1)
        return self.fusion(torch.cat((features, presence.to(features.dtype)), -1))

    def forward(self, observations: dict[str, Tensor], presence: Tensor) -> Tensor:
        """Produce the window's global diffusion condition."""
        return self.fuse_tokens(self.encode_tokens(observations), presence)


class SingleTaskTeacherPolicy(BaseImagePolicy):
    """Standard epsilon DDPM loss with sample-wise sensor dropout during training."""

    def __init__(self, shape_meta: dict, obs_encoder: SingleTaskTeacherEncoder, noise_scheduler: Any,
                 full_input_probability: float = 0.6, horizon: int = 16, n_obs_steps: int = 2,
                 n_action_steps: int = 8, num_inference_steps: int = 100,
                 diffusion_step_embed_dim: int = 128, down_dims: tuple[int, ...] = (512, 1024, 2048),
                 kernel_size: int = 5, n_groups: int = 8, cond_predict_scale: bool = True) -> None:
        super().__init__()
        self._dummy_variable.requires_grad_(False)
        if not 0 <= full_input_probability <= 1 or noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("Expected a valid full-input probability and epsilon prediction")
        self.obs_encoder, self.noise_scheduler = obs_encoder, noise_scheduler
        self.horizon, self.n_obs_steps, self.n_action_steps = horizon, n_obs_steps, n_action_steps
        self.num_inference_steps = num_inference_steps
        self.action_dim = int(shape_meta["action"]["shape"][0])
        if self.action_dim not in (6, 7) or obs_encoder.n_obs_steps != n_obs_steps:
            raise ValueError("Invalid task action dimension or observation history")
        self.condition_names, masks = sensor_conditions(obs_encoder.image_keys)
        self.register_buffer("condition_presence", masks)
        self.register_buffer("condition_probabilities", torch.tensor(
            [full_input_probability] + [(1 - full_input_probability) / (len(masks) - 1)] * (len(masks) - 1)))
        self.normalizer = LinearNormalizer()
        self.model = ConditionalUnet1D(input_dim=self.action_dim, global_cond_dim=obs_encoder.token_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=list(down_dims),
            kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Persist fitted statistics as frozen model parameters."""
        self.normalizer.load_state_dict(normalizer.state_dict(), strict=True)
        self.normalizer.requires_grad_(False)

    def sample_condition_ids(self, batch_size: int, generator: torch.Generator | None = None) -> Tensor:
        """Sample one intervention per history window on a separate RNG stream."""
        return torch.multinomial(self.condition_probabilities, batch_size, replacement=True, generator=generator)

    def losses(self, batch: dict, condition_ids: Tensor, noise: Tensor | None = None,
               timesteps: Tensor | None = None) -> Tensor:
        """Return per-sample epsilon MSE over the official padded action window."""
        action = self.normalizer["action"].normalize(batch["action"])
        if tuple(action.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError("Action shape differs from this task's diffusion contract")
        noise = torch.randn_like(action) if noise is None else noise
        timesteps = torch.randint(self.noise_scheduler.config.num_train_timesteps,
                                  (len(action),), device=action.device) if timesteps is None else timesteps
        condition = self.obs_encoder(self.normalizer.normalize(batch["obs"]), self.condition_presence[condition_ids])
        prediction = self.model(self.noise_scheduler.add_noise(action, noise, timesteps), timesteps,
                                global_cond=condition)
        return (prediction - noise).square().mean((1, 2))

    def compute_loss(self, batch: dict) -> Tensor:
        """Use dropout while training and complete observations during evaluation."""
        ids = self.sample_condition_ids(len(batch["action"])) if self.training else torch.zeros(
            len(batch["action"]), dtype=torch.long, device=self.device)
        return self.losses(batch, ids).mean()

    @torch.no_grad()
    def paired_denoising_losses(self, batch: dict, generator: torch.Generator) -> dict[str, Tensor]:
        """Reuse encoded tokens, noise and timesteps for all removal conditions."""
        if self.training:
            raise RuntimeError("Paired validation requires eval mode")
        action = self.normalizer["action"].normalize(batch["action"])
        noise = torch.randn(action.shape, device=action.device, generator=generator)
        timesteps = torch.randint(self.noise_scheduler.config.num_train_timesteps,
                                  (len(action),), device=action.device, generator=generator)
        noisy = self.noise_scheduler.add_noise(action, noise, timesteps)
        tokens = self.obs_encoder.encode_tokens(self.normalizer.normalize(batch["obs"]))
        losses = []
        for presence in self.condition_presence:
            condition = self.obs_encoder.fuse_tokens(tokens, presence.expand(len(action), -1))
            prediction = self.model(noisy, timesteps, global_cond=condition)
            losses.append((prediction - noise).square().mean((1, 2)))
        values = torch.stack(losses, 1)
        return {"losses": values, "utility": values[:, 1:] - values[:, :1]}

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Generate the task's native 6D/7D action chunk from complete observations."""
        presence = self.condition_presence[0].expand(len(obs_dict["state"]), -1)
        condition = self.obs_encoder(self.normalizer.normalize(obs_dict), presence)
        trajectory = torch.randn((len(condition), self.horizon, self.action_dim), device=self.device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, global_cond=condition)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory).prev_sample
        actions = self.normalizer["action"].unnormalize(trajectory)
        start = self.n_obs_steps - 1
        return {"action": actions[:, start:start + self.n_action_steps], "action_pred": actions}


class SingleTaskTeacherDataset(ManifeelDataset):
    """Use official sequence indices/splits with lazy source reads and train-only stats."""

    def __init__(self, shape_meta: dict, zarr_path: str, horizon: int = 16, n_obs_steps: int = 2,
                 n_action_steps: int = 8, seed: int = 42, val_ratio: float = 0.02,
                 max_train_episodes: int = 50) -> None:
        if OmegaConf.is_config(shape_meta):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        self.shape_meta = copy.deepcopy(shape_meta)
        self.rgb_keys = [key for key, item in shape_meta["obs"].items() if item["type"] == "rgb"]
        self.lowdim_keys = ["state"]
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        self.pad_before, self.pad_after = n_obs_steps - 1, n_action_steps - 1
        self.zarr_path = str(Path(zarr_path).resolve())
        source = zarr.open_group(self.zarr_path, mode="r")
        ends = np.asarray(source["meta/episode_ends"][:], dtype=np.int64)
        keys = [*self.rgb_keys, "state", "action"]
        self.replay_buffer = ReplayBuffer({"data": {key: source["data"][key] for key in keys},
                                           "meta": {"episode_ends": ends}})
        if source["data/action"].shape[1:] != tuple(shape_meta["action"]["shape"]):
            raise ValueError("Native dataset action dimension differs from task config")
        self.val_mask = get_val_mask(len(ends), val_ratio, seed)
        self.train_mask = downsample_mask(~self.val_mask, max_train_episodes, seed)
        self._make_sampler(self.train_mask)
        self.manifest = {"source": self.zarr_path, "shape_meta": shape_meta, "episode_ends": ends.tolist(),
                         "train_episodes": np.flatnonzero(self.train_mask).tolist(),
                         "val_episodes": np.flatnonzero(self.val_mask).tolist(), "seed": seed,
                         "horizon": horizon, "n_obs_steps": n_obs_steps, "n_action_steps": n_action_steps,
                         "source_arrays": {key: {"shape": list(source["data"][key].shape),
                             "metadata": (Path(self.zarr_path) / "data" / key / ".zarray").read_text()}
                             for key in keys}, "normalization": "train_episodes_only"}

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Read only the observed frames, exactly matching official padded windows."""
        buffer_start, buffer_end, sample_start, _ = self.sampler.indices[index]
        sample = {}
        for key in (*self.rgb_keys, "state", "action"):
            count = self.horizon if key == "action" else self.n_obs_steps
            indices = buffer_start + np.clip(np.arange(count) - sample_start, 0, buffer_end - buffer_start - 1)
            first, last = int(indices.min()), int(indices.max())
            sample[key] = np.asarray(self.replay_buffer[key][first:last + 1])[indices - first]
        return self._to_tensors(self._sample_to_data(sample))

    @staticmethod
    def _to_tensors(data: dict) -> dict:
        """Convert official field layouts into writable CPU tensors."""
        return {"obs": {key: torch.from_numpy(value) for key, value in data["obs"].items()},
                "action": torch.from_numpy(data["action"])}

    def _make_sampler(self, mask: np.ndarray) -> None:
        self.sampler = SequenceSampler(self.replay_buffer, self.horizon, self.pad_before, self.pad_after,
                                      episode_mask=mask, key_first_k={key: self.n_obs_steps
                                      for key in [*self.rgb_keys, "state"]})

    def load_to_ram(self, workers: int = 3) -> None:
        """Decode selected source arrays once and share read-only memory via fork."""
        source = self.replay_buffer.root["data"]

        def read(key: str) -> tuple[str, np.ndarray]:
            values = np.asarray(source[key][:], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite source array: {key}")
            values.flags.writeable = False
            print(json.dumps({"event": "ram_array_ready", "key": key, "nbytes": values.nbytes}), flush=True)
            return key, values

        with ThreadPoolExecutor(max_workers=workers) as executor:
            arrays = dict(executor.map(read, source))
        self.replay_buffer = ReplayBuffer({"data": arrays, "meta": self.replay_buffer.root["meta"]})
        self._make_sampler(self.train_mask)

    def get_validation_dataset(self) -> "SingleTaskTeacherDataset":
        """Keep the original held-out episodes, excluding unused training episodes."""
        result = copy.copy(self)
        result.train_mask = self.val_mask.copy()
        result._make_sampler(self.val_mask)
        return result

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> LinearNormalizer:
        """Fit only real training frames, retaining the official field transforms."""
        ends = self.replay_buffer.episode_ends[:]
        fields = {}
        for key in ("state", "action"):
            fields[key] = np.concatenate([self.replay_buffer[key][0 if index == 0 else ends[index - 1]:ends[index]]
                                          for index in np.flatnonzero(self.train_mask)])
        normalizer = LinearNormalizer()
        normalizer.fit(fields, last_n_dims=1, mode=mode, **kwargs)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        normalizer.requires_grad_(False)
        return normalizer


def initialize_teacher_worker(worker_id: int) -> None:
    """Report actual worker affinity independently of trainer-side CPU assumptions."""
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    path = os.environ.get("TEACHER_AFFINITY_LOG")
    if path:
        record = {"worker_id": worker_id, "pid": os.getpid(), "cpus": sorted(os.sched_getaffinity(0))}
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, (json.dumps(record) + "\n").encode())
        finally:
            os.close(descriptor)
