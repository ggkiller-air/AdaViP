"""Stage 1 v2 Teacher with action diffusion and future latent prediction."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from adavip.manifeel.full_sensor_schema import IMAGE_KEYS
from adavip.manifeel.stage1_v2_schema import (
    ACTION_EXECUTION_STEPS,
    ACTION_HISTORY_STEPS,
    ACTION_HORIZON,
    FUTURE_STEPS,
    IMAGE_FEATURE_DIM,
    MASK_NAMES,
    OBS_HISTORY_STEPS,
    SENSOR_TOKEN_NAMES,
    TOKEN_DIM,
    condition_presence,
    condition_probabilities,
    validate_stage1_shape_meta,
)
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class TemporalTokenEncoder(nn.Module):
    """Encode one sensor's six-frame feature sequence into one token."""

    def __init__(self, input_dim: int, token_dim: int, steps: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=max(input_dim * 2, 512),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.temporal_embedding = nn.Parameter(torch.zeros(steps, input_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, token_dim),
            nn.SiLU(),
            nn.LayerNorm(token_dim),
        )
        nn.init.normal_(self.temporal_embedding, std=0.02)

    def forward(self, features: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if features.ndim != 3 or features.shape[1] != self.temporal_embedding.shape[0]:
            raise ValueError("Temporal features must have shape [B, configured_steps, D]")
        encoded = features + self.temporal_embedding.unsqueeze(0)
        padding_mask = None if valid_mask is None else ~valid_mask.bool()
        encoded = self.temporal(encoded, src_key_padding_mask=padding_mask)
        if valid_mask is None:
            pooled = encoded[:, -1]
        else:
            weights = valid_mask.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled)


@dataclass(frozen=True)
class DenoisingSample:
    """Shared diffusion noise and target for one action batch."""

    noisy_action: Tensor
    noise: Tensor
    timesteps: Tensor
    target: Tensor
    loss_mask: Tensor


class Stage1V2TeacherPolicy(BaseImagePolicy):
    """From-scratch seven-token Teacher for the redesigned Stage 1 protocol."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: Any,
        horizon: int = ACTION_HORIZON,
        n_obs_steps: int = OBS_HISTORY_STEPS,
        n_action_steps: int = ACTION_EXECUTION_STEPS,
        num_inference_steps: int = 100,
        token_dim: int = TOKEN_DIM,
        image_feature_dim: int = IMAGE_FEATURE_DIM,
        action_history_steps: int = ACTION_HISTORY_STEPS,
        future_ema_decay: float = 0.999,
        full_input_probability: float = 0.8,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        validate_stage1_shape_meta(shape_meta)
        self._dummy_variable.requires_grad_(False)
        if horizon != ACTION_HORIZON or n_obs_steps != OBS_HISTORY_STEPS:
            raise ValueError("Stage 1 v2 requires horizon=16 and n_obs_steps=6")
        if n_action_steps != ACTION_EXECUTION_STEPS:
            raise ValueError("Stage 1 v2 requires n_action_steps=8")
        if action_history_steps != ACTION_HISTORY_STEPS:
            raise ValueError("Stage 1 v2 requires eight executed action-history steps")
        if image_feature_dim != IMAGE_FEATURE_DIM or token_dim != TOKEN_DIM:
            raise ValueError("Stage 1 v2 fixes image features at 512D and tokens at 256D")
        if not 0.0 < future_ema_decay < 1.0:
            raise ValueError("future_ema_decay must be in (0, 1)")
        if noise_scheduler.config.num_train_timesteps != 100:
            raise ValueError("Stage 1 v2 uses exactly 100 diffusion training timesteps")
        if horizon % (2 ** (len(down_dims) - 1)):
            raise ValueError("horizon must be divisible by the U-Net downsampling factor")
        self.shape_meta = copy.deepcopy(shape_meta)
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.action_history_steps = action_history_steps
        self.token_dim = token_dim
        self.image_feature_dim = image_feature_dim
        self.future_ema_decay = future_ema_decay
        self.num_inference_steps = num_inference_steps
        self.action_dim = 7
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()

        backbone = get_resnet("resnet18", weights=None)
        backbone = replace_submodules(
            root_module=backbone,
            predicate=lambda module: isinstance(module, nn.BatchNorm2d),
            func=lambda module: nn.GroupNorm(math.gcd(32, module.num_features), module.num_features),
        )
        self.image_encoders = nn.ModuleDict({key: copy.deepcopy(backbone) for key in IMAGE_KEYS})
        self.target_image_encoders = nn.ModuleDict({
            key: copy.deepcopy(self.image_encoders[key]) for key in IMAGE_KEYS
        })
        for encoder in self.target_image_encoders.values():
            encoder.requires_grad_(False)
            encoder.eval()

        self.image_temporal = nn.ModuleDict({
            key: TemporalTokenEncoder(image_feature_dim, token_dim, n_obs_steps)
            for key in IMAGE_KEYS
        })
        self.proprio_frame = nn.Sequential(
            nn.Linear(7, 128), nn.SiLU(), nn.Linear(128, token_dim), nn.LayerNorm(token_dim),
        )
        self.proprio_temporal = TemporalTokenEncoder(token_dim, token_dim, n_obs_steps)
        language_dim = int(shape_meta["obs"]["task_embedding"]["shape"][0])
        self.language_projection = nn.Sequential(
            nn.Linear(language_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
        )
        self.sensor_identity = nn.Parameter(torch.empty(len(SENSOR_TOKEN_NAMES), token_dim))
        self.missing_tokens = nn.Parameter(torch.zeros(len(SENSOR_TOKEN_NAMES), token_dim))
        self.missing_image_features = nn.Parameter(torch.zeros(len(IMAGE_KEYS), image_feature_dim))
        self.missing_proprio_feature = nn.Parameter(torch.zeros(token_dim))
        nn.init.normal_(self.sensor_identity, std=0.02)

        state_input_dim = len(IMAGE_KEYS) * image_feature_dim + token_dim + token_dim + 128
        self.action_history = nn.LSTM(input_size=7, hidden_size=128, num_layers=1, batch_first=True)
        self.state_tokenizer = nn.Sequential(
            nn.Linear(state_input_dim, 512), nn.SiLU(), nn.LayerNorm(512),
            nn.Linear(512, token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
        )
        self.cross_attention = nn.MultiheadAttention(token_dim, num_heads=8, batch_first=True)
        self.condition_projection = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim))

        self.model = ConditionalUnet1D(
            input_dim=7,
            global_cond_dim=token_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.future_action_projection = nn.Sequential(
            nn.Linear(7, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
        )
        self.future_projection = nn.Sequential(
            nn.Linear(token_dim + 128, 256), nn.SiLU(), nn.LayerNorm(256),
        )
        self.future_vision_head = nn.Linear(256, 3 * image_feature_dim)
        self.future_tactile_head = nn.Linear(256, 2 * image_feature_dim)
        self.register_buffer("condition_presence", torch.from_numpy(condition_presence()))
        self.register_buffer(
            "condition_probabilities",
            torch.as_tensor(condition_probabilities(full_input_probability), dtype=torch.float32),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Restore frozen action, state and image normalization statistics."""
        self.normalizer.load_state_dict(normalizer.state_dict(), strict=True)
        self.normalizer.requires_grad_(False)
        self.normalizer.to(self.device)

    def load_state_dict(self, state_dict: dict, strict: bool = True, **kwargs: Any) -> Any:
        """Restore upstream dynamic normalization fields before recursive module loading."""
        normalizer_state = {
            name[len("normalizer."):]: value
            for name, value in state_dict.items() if name.startswith("normalizer.")
        }
        if normalizer_state:
            self.normalizer.load_state_dict(normalizer_state, strict=True)
        result = super().load_state_dict(state_dict, strict=strict, **kwargs)
        self.normalizer.requires_grad_(False)
        self.normalizer.to(self.device)
        return result

    def update_target_encoder(self) -> None:
        """Update target image encoders only after an optimizer step."""
        with torch.no_grad():
            decay = self.future_ema_decay
            for key in IMAGE_KEYS:
                source = self.image_encoders[key]
                target = self.target_image_encoders[key]
                for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
                    target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
                for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
                    target_buffer.copy_(source_buffer)

    def sample_condition_ids(self, batch_size: int, generator: torch.Generator | None = None) -> Tensor:
        """Sample full or one of eight modality masks for each training item."""
        return torch.multinomial(
            self.condition_probabilities,
            batch_size,
            replacement=True,
            generator=generator,
        )

    def _encode_sensor_tokens(
        self,
        observations: dict[str, Tensor],
        presence: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = observations["state"].shape[0]
        expected_language = self.shape_meta["obs"]["task_embedding"]["shape"][0]
        tokens: list[Tensor] = []
        latest_features: list[Tensor] = []
        for key in IMAGE_KEYS:
            images = observations[key]
            expected = (batch_size, self.n_obs_steps, *self.shape_meta["obs"][key]["shape"])
            if tuple(images.shape) != expected:
                raise ValueError(f"{key}: expected {expected}, got {tuple(images.shape)}")
            features = self.image_encoders[key](images.flatten(0, 1)).reshape(
                batch_size, self.n_obs_steps, self.image_feature_dim,
            )
            tokens.append(self.image_temporal[key](features))
            latest_features.append(features[:, -1])

        state = observations["state"]
        if tuple(state.shape) != (batch_size, self.n_obs_steps, 7):
            raise ValueError("state must have shape [B,6,7]")
        proprio_frames = self.proprio_frame(state)
        tokens.append(self.proprio_temporal(proprio_frames))
        latest_features.append(proprio_frames[:, -1])

        language = observations["task_embedding"]
        if tuple(language.shape) != (batch_size, self.n_obs_steps, expected_language):
            raise ValueError("task_embedding must have shape [B,6,512]")
        if not torch.equal(language, language[:, :1].expand_as(language)):
            raise ValueError("Language embedding must be constant over one history window")
        language_token = self.language_projection(language[:, 0])
        tokens.append(language_token)
        latest_features.append(language_token)

        stacked_tokens = torch.stack(tokens, dim=1) + self.sensor_identity.unsqueeze(0)
        if presence.shape != (batch_size, len(SENSOR_TOKEN_NAMES)) or presence.dtype != torch.bool:
            raise ValueError("presence must be boolean [B,7]")
        if not torch.all(presence[:, -1]):
            raise ValueError("Language is never allowed to be dropped")
        missing_tokens = (self.missing_tokens + self.sensor_identity).unsqueeze(0)
        stacked_tokens = torch.where(presence.unsqueeze(-1), stacked_tokens, missing_tokens)

        latest_images = torch.stack(latest_features[:len(IMAGE_KEYS)], dim=1)
        latest_images = torch.where(
            presence[:, :len(IMAGE_KEYS)].unsqueeze(-1),
            latest_images,
            self.missing_image_features.unsqueeze(0),
        )
        latest_proprio = latest_features[len(IMAGE_KEYS)].unsqueeze(1)
        latest_proprio = torch.where(
            presence[:, len(IMAGE_KEYS):len(IMAGE_KEYS) + 1].unsqueeze(-1),
            latest_proprio,
            self.missing_proprio_feature.view(1, 1, -1),
        )
        latest = torch.cat((latest_images.flatten(1), latest_proprio[:, 0], latest_features[-1]), dim=-1)
        return stacked_tokens, latest

    def train(self, mode: bool = True) -> "Stage1V2TeacherPolicy":
        """Keep EMA target encoders frozen and in evaluation mode."""
        super().train(mode)
        for encoder in self.target_image_encoders.values():
            encoder.eval()
        return self

    def _state_token(
        self,
        observations: dict[str, Tensor],
        action_history: Tensor,
        action_history_valid_mask: Tensor,
        presence: Tensor,
    ) -> tuple[Tensor, Tensor]:
        tokens, latest = self._encode_sensor_tokens(observations, presence)
        if tuple(action_history.shape[1:]) != (self.action_history_steps, 7):
            raise ValueError("action_history must have shape [B,8,7]")
        history = torch.where(
            action_history_valid_mask.unsqueeze(-1).bool(), action_history, torch.zeros_like(action_history),
        )
        _, (hidden, _) = self.action_history(history)
        state_input = torch.cat((latest, hidden[-1]), dim=-1)
        z = self.state_tokenizer(state_input)
        attended, _ = self.cross_attention(z.unsqueeze(1), tokens, tokens)
        conditioning = self.condition_projection(z + attended[:, 0])
        return z, conditioning

    def _normalize_action_history(self, action_history: Tensor) -> Tensor:
        return self.normalizer["action"].normalize(action_history)

    def prepare_denoising_sample(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DenoisingSample:
        actions = batch["action"]
        loss_mask = batch["action_loss_mask"].to(device=actions.device, dtype=actions.dtype)
        if actions.shape[1:] != (self.horizon, 7) or loss_mask.shape != actions.shape:
            raise ValueError("action and action_loss_mask must have shape [B,16,7]")
        trajectory = self.normalizer["action"].normalize(actions)
        trajectory = torch.where(loss_mask.bool(), trajectory, torch.zeros_like(trajectory))
        if noise is None:
            noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype, generator=generator)
        if timesteps is None:
            timesteps = torch.randint(
                self.noise_scheduler.config.num_train_timesteps,
                (actions.shape[0],),
                device=actions.device,
                generator=generator,
            )
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else trajectory
        return DenoisingSample(
            noisy_action=self.noise_scheduler.add_noise(trajectory, noise, timesteps),
            noise=noise,
            timesteps=timesteps,
            target=target,
            loss_mask=loss_mask,
        )

    def _future_targets(self, future_obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        vision_targets = []
        tactile_targets = []
        with torch.no_grad():
            for key in IMAGE_KEYS[:3]:
                images = future_obs[key]
                features = self.target_image_encoders[key](images.flatten(0, 1)).reshape(
                    images.shape[0], FUTURE_STEPS, self.image_feature_dim,
                )
                vision_targets.append(features)
            for key in IMAGE_KEYS[3:]:
                images = future_obs[key]
                features = self.target_image_encoders[key](images.flatten(0, 1)).reshape(
                    images.shape[0], FUTURE_STEPS, self.image_feature_dim,
                )
                tactile_targets.append(features)
        return torch.stack(vision_targets, dim=2), torch.stack(tactile_targets, dim=2)

    def _future_predictions(self, z: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        action_features = self.future_action_projection(actions[:, 1:1 + FUTURE_STEPS])
        state_features = z.unsqueeze(1).expand(-1, FUTURE_STEPS, -1)
        hidden = self.future_projection(torch.cat((state_features, action_features), dim=-1))
        vision = self.future_vision_head(hidden).reshape(
            actions.shape[0], FUTURE_STEPS, 3, self.image_feature_dim,
        )
        tactile = self.future_tactile_head(hidden).reshape(
            actions.shape[0], FUTURE_STEPS, 2, self.image_feature_dim,
        )
        return vision, tactile

    def compute_loss_with_metrics(
        self,
        batch: dict[str, Any],
        condition_ids: Tensor | None = None,
        denoising_sample: DenoisingSample | None = None,
    ) -> dict[str, Tensor]:
        batch_size = batch["action"].shape[0]
        if condition_ids is None:
            condition_ids = self.sample_condition_ids(batch_size)
        presence = self.condition_presence[condition_ids.to(self.device)]
        observations = self.normalizer.normalize(batch["obs"])
        action_history = self._normalize_action_history(batch["action_history"])
        action_history = action_history * batch["action_dimension_mask"].unsqueeze(1)
        z, conditioning = self._state_token(
            observations,
            action_history,
            batch["action_history_valid_mask"].to(self.device),
            presence,
        )
        sample = denoising_sample or self.prepare_denoising_sample(batch)
        prediction = self.model(sample.noisy_action, sample.timesteps, global_cond=conditioning)
        action_error = (prediction - sample.target).square() * sample.loss_mask
        action_loss = action_error.sum(dim=(1, 2)) / sample.loss_mask.sum(dim=(1, 2)).clamp_min(1.0)

        future_obs = {
            key: self.normalizer[key].normalize(batch["future_obs"][key].to(self.device))
            for key in IMAGE_KEYS
        }
        future_actions = self.normalizer["action"].normalize(batch["action"].to(self.device))
        future_actions = future_actions * batch["action_dimension_mask"].unsqueeze(1)
        predicted_vision, predicted_tactile = self._future_predictions(z, future_actions)
        target_vision, target_tactile = self._future_targets(future_obs)
        valid = batch["future_valid_mask"].to(self.device, dtype=predicted_vision.dtype)
        vision_error = (predicted_vision - target_vision).square().mean(dim=(2, 3))
        tactile_error = (predicted_tactile - target_tactile).square().mean(dim=(2, 3))
        vision_loss = (vision_error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        tactile_loss = (tactile_error * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return {
            "action_loss": action_loss.mean(),
            "future_vision_loss": vision_loss.mean(),
            "future_tactile_loss": tactile_loss.mean(),
            "vision_latent_variance": target_vision.var(dim=(0, 1), unbiased=False).mean(),
            "tactile_latent_variance": target_tactile.var(dim=(0, 1), unbiased=False).mean(),
            "z": z,
        }

    def compute_loss(
        self,
        batch: dict[str, Any],
        lambda_vision: float = 1.0,
        lambda_tactile: float = 1.0,
        condition_ids: Tensor | None = None,
    ) -> Tensor:
        """Return action plus separately weighted future latent losses."""
        losses = self.compute_loss_with_metrics(batch, condition_ids=condition_ids)
        return (
            losses["action_loss"]
            + lambda_vision * losses["future_vision_loss"]
            + lambda_tactile * losses["future_tactile_loss"]
        )

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Return the eight executable actions with the t-1 prefix removed."""
        sensor_obs = {key: obs_dict[key] for key in (*IMAGE_KEYS, "state", "task_embedding")}
        if "action_history" not in obs_dict:
            raise ValueError("Supply the last eight executed actions; never substitute predicted future actions")
        action_history = obs_dict["action_history"]
        history_valid = obs_dict.get(
            "action_history_valid_mask",
            torch.ones(action_history.shape[:2], dtype=torch.bool, device=action_history.device),
        )
        observations = self.normalizer.normalize(sensor_obs)
        history = self._normalize_action_history(action_history)
        if "action_dimension_mask" in obs_dict:
            history = history * obs_dict["action_dimension_mask"].unsqueeze(1)
        presence = torch.ones(
            (sensor_obs["state"].shape[0], len(SENSOR_TOKEN_NAMES)),
            dtype=torch.bool,
            device=sensor_obs["state"].device,
        )
        _, conditioning = self._state_token(observations, history, history_valid, presence)
        trajectory = torch.randn(
            sensor_obs["state"].shape[0], self.horizon, self.action_dim,
            device=conditioning.device,
            dtype=conditioning.dtype,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps, device=conditioning.device)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, global_cond=conditioning)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory).prev_sample
        actions = self.normalizer["action"].unnormalize(trajectory)
        return actions[:, 1:1 + self.n_action_steps]

    def parameter_counts(self) -> dict[str, int]:
        """Return trainable and total parameter counts for the handoff report."""
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable": trainable, "total": total}


__all__ = ["DenoisingSample", "Stage1V2TeacherPolicy", "TemporalTokenEncoder"]
