"""Full-sensor DDPM reference policy with paired, per-sample denoising losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from adavip.manifeel.full_sensor_obs_encoder import FullSensorObsEncoder
from adavip.manifeel.full_sensor_schema import CONDITION_NAMES, condition_presence, condition_probabilities
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


@dataclass(frozen=True)
class DenoisingSample:
    """The shared target, noise, timestep and reduction mask for paired evaluation."""

    noisy_action: Tensor
    noise: Tensor
    timesteps: Tensor
    target: Tensor
    loss_mask: Tensor


class FullSensorReferencePolicy(BaseImagePolicy):
    """Train a seven-input reference DP without Stage-1/2 research modules."""

    def __init__(
        self,
        obs_encoder: FullSensorObsEncoder,
        noise_scheduler: Any,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        num_inference_steps: int = 100,
        full_input_probability: float = 0.8,
        mask_weights: list[float] | None = None,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (512, 1024, 2048),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        self._dummy_variable.requires_grad_(False)
        if horizon < n_obs_steps + n_action_steps - 1 or min(n_obs_steps, n_action_steps) <= 0:
            raise ValueError("Prediction horizon must contain history and the executed action chunk")
        if obs_encoder.n_obs_steps != n_obs_steps:
            raise ValueError("Policy and encoder observation histories must agree")
        if horizon % (2 ** (len(down_dims) - 1)):
            raise ValueError("horizon must be divisible by the U-Net downsampling factor")
        if noise_scheduler.config.prediction_type not in ("epsilon", "sample"):
            raise ValueError("Only epsilon and sample prediction are supported")
        if not 1 <= num_inference_steps <= noise_scheduler.config.num_train_timesteps:
            raise ValueError("Invalid number of inference diffusion steps")
        self.obs_encoder = obs_encoder
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_inference_steps = num_inference_steps
        self.action_dim = 7
        self.normalizer = LinearNormalizer()
        self.register_buffer("condition_presence", torch.from_numpy(condition_presence()))
        self.register_buffer("condition_probabilities", torch.as_tensor(
            condition_probabilities(full_input_probability, mask_weights), dtype=torch.float32,
        ))
        self.model = ConditionalUnet1D(
            input_dim=7, global_cond_dim=obs_encoder.token_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=list(down_dims),
            kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale,
        )

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Restore all fitted training statistics as frozen checkpoint parameters."""
        self.normalizer.load_state_dict(normalizer.state_dict(), strict=True)
        self.normalizer.requires_grad_(False)

    def sample_condition_ids(self, batch_size: int, generator: torch.Generator | None = None) -> Tensor:
        """Draw one full/removal condition per sample, shared across all history frames."""
        return torch.multinomial(self.condition_probabilities, batch_size, replacement=True, generator=generator)

    def prepare_denoising_sample(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> DenoisingSample:
        """Create one noisy action shared by every counterfactual condition."""
        actions = batch["action"]
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (self.horizon, 7):
            raise ValueError("action must have shape [B,horizon,7]")
        loss_mask = batch["action_loss_mask"].to(device=actions.device, dtype=actions.dtype)
        if loss_mask.shape != actions.shape or not torch.all((loss_mask == 0) | (loss_mask == 1)):
            raise ValueError("action_loss_mask must be binary with shape [B,horizon,7]")
        if not torch.all(loss_mask.sum(dim=(1, 2)) > 0):
            raise ValueError("Each sample must contain a valid supervised action")
        trajectory = self.normalizer["action"].normalize(actions)
        trajectory = torch.where(loss_mask.bool(), trajectory, torch.zeros_like(trajectory))
        if not torch.isfinite(trajectory).all():
            raise ValueError("Valid normalized actions must be finite")
        if noise is None:
            noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype, generator=generator)
        if (noise.shape != trajectory.shape or noise.device != trajectory.device
                or noise.dtype != trajectory.dtype or not torch.isfinite(noise).all()):
            raise ValueError("noise must be finite and match the action shape/device/dtype")
        if timesteps is None:
            timesteps = torch.randint(
                self.noise_scheduler.config.num_train_timesteps, (actions.shape[0],),
                device=actions.device, generator=generator,
            )
        if timesteps.dtype != torch.long or timesteps.shape != (actions.shape[0],) or timesteps.device != actions.device:
            raise ValueError("timesteps must be int64 [B] on the action device")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.noise_scheduler.config.num_train_timesteps):
            raise ValueError("Diffusion timestep is out of range")
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else trajectory
        return DenoisingSample(
            self.noise_scheduler.add_noise(trajectory, noise, timesteps), noise, timesteps, target, loss_mask,
        )

    def denoising_loss_per_sample(self, sample: DenoisingSample, conditioning: Tensor) -> Tensor:
        """Return [B] MSE, divided by each sample's valid time/dimension count."""
        prediction = self.model(sample.noisy_action, sample.timesteps, global_cond=conditioning)
        squared_error = (prediction - sample.target).square() * sample.loss_mask
        return squared_error.sum(dim=(1, 2)) / sample.loss_mask.sum(dim=(1, 2))

    def compute_loss(
        self,
        batch: dict[str, Any],
        condition_ids: Tensor | None = None,
        denoising_sample: DenoisingSample | None = None,
    ) -> Tensor:
        """Train with configurable dropout; validation defaults to full input."""
        return self.compute_loss_per_sample(batch, condition_ids, denoising_sample).mean()

    def compute_loss_per_sample(
        self,
        batch: dict[str, Any],
        condition_ids: Tensor | None = None,
        denoising_sample: DenoisingSample | None = None,
    ) -> Tensor:
        """Expose unreduced [B] losses for task/condition-aware aggregation."""
        batch_size = batch["action"].shape[0]
        if condition_ids is None:
            condition_ids = self.sample_condition_ids(batch_size) if self.training else torch.zeros(
                batch_size, dtype=torch.long, device=self.device,
            )
        if condition_ids.dtype != torch.long or condition_ids.shape != (batch_size,):
            raise ValueError("condition_ids must have shape [B] and int64 dtype")
        if torch.any(condition_ids < 0) or torch.any(condition_ids >= len(CONDITION_NAMES)):
            raise ValueError("Unknown sensor removal condition")
        presence = self.condition_presence[condition_ids.to(self.device)]
        normalized = self.normalizer.normalize(batch["obs"])
        conditioning = self.obs_encoder(normalized, presence)
        sample = self.prepare_denoising_sample(batch) if denoising_sample is None else denoising_sample
        return self.denoising_loss_per_sample(sample, conditioning)

    @torch.no_grad()
    def paired_denoising_losses(
        self,
        batch: dict[str, Any],
        noise: Tensor | None = None,
        timesteps: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Evaluate full plus nine masks in eval mode using exactly the same noise.

        Columns follow CONDITION_NAMES. Utility is raw signed masked minus full
        loss. This does not generate rollouts or modify the teacher's weights.
        """
        if self.training or self.obs_encoder.training or self.model.training:
            raise RuntimeError("Paired labeling requires policy.eval() and deterministic encoders")
        sample = self.prepare_denoising_sample(batch, noise, timesteps, generator)
        tokens = self.obs_encoder.encode_tokens(self.normalizer.normalize(batch["obs"]))
        losses = []
        for presence in self.condition_presence:
            conditioning = self.obs_encoder.fuse_tokens(tokens, presence.expand(tokens.shape[0], -1))
            losses.append(self.denoising_loss_per_sample(sample, conditioning))
        absolute_losses = torch.stack(losses, dim=1)
        return {"losses": absolute_losses, "utility": absolute_losses[:, 1:] - absolute_losses[:, :1]}

    @torch.no_grad()
    def predict_action(
        self,
        obs_dict: dict[str, Tensor],
        presence: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        """Sample [B,Ta,7] actions; task-specific truncation belongs to the runner."""
        conditioning = self.obs_encoder(self.normalizer.normalize(obs_dict), presence)
        trajectory = torch.randn(
            (conditioning.shape[0], self.horizon, 7), device=conditioning.device,
            dtype=conditioning.dtype, generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, global_cond=conditioning)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory, generator=generator).prev_sample
        actions = self.normalizer["action"].unnormalize(trajectory)
        start = self.n_obs_steps - 1
        return {"action": actions[:, start:start + self.n_action_steps], "action_pred": actions}
