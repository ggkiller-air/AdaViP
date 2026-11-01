"""OpenViTac adapter around the existing UViP ConditionalUnet1D DP."""
from __future__ import annotations

import torch

from adavip.manifeel.reference_diffusion_policy import DenoisingSample, FullSensorReferencePolicy
from adavip.openvitac.schema import ACTION_DIM, CONDITION_NAMES, condition_presence, condition_probabilities
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class OpenViTacPolicy(FullSensorReferencePolicy):
    """Reuse masked loss reduction, normalizer handling and the original DP backbone."""

    def __init__(self, obs_encoder, noise_scheduler, horizon=16, n_obs_steps=2,
                 n_action_steps=8, num_inference_steps=100, full_input_probability=.8,
                 diffusion_step_embed_dim=128, down_dims=(512, 1024, 2048), kernel_size=5,
                 n_groups=8, cond_predict_scale=True):
        # Do not construct the ManiFeel 7D U-Net or condition buffers.
        BaseImagePolicy.__init__(self)
        self._dummy_variable.requires_grad_(False)
        if n_obs_steps != obs_encoder.n_obs_steps or horizon < n_obs_steps + n_action_steps - 1:
            raise ValueError("Inconsistent policy history/horizon")
        if horizon % 2 ** (len(down_dims) - 1):
            raise ValueError("Horizon must match the unchanged U-Net downsampling")
        if noise_scheduler.config.prediction_type not in ("epsilon", "sample"):
            raise ValueError("Unsupported diffusion target")
        if not 1 <= num_inference_steps <= noise_scheduler.config.num_train_timesteps:
            raise ValueError("Invalid inference steps")
        self.obs_encoder, self.noise_scheduler = obs_encoder, noise_scheduler
        self.horizon, self.n_obs_steps, self.n_action_steps = horizon, n_obs_steps, n_action_steps
        self.num_inference_steps, self.action_dim = num_inference_steps, ACTION_DIM
        self.normalizer = LinearNormalizer()
        self.register_buffer("condition_presence", torch.from_numpy(condition_presence()))
        self.register_buffer("condition_probabilities", torch.from_numpy(condition_probabilities(full_input_probability)).float())
        self.model = ConditionalUnet1D(input_dim=ACTION_DIM, global_cond_dim=obs_encoder.token_dim,
                                      diffusion_step_embed_dim=diffusion_step_embed_dim, down_dims=list(down_dims),
                                      kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale)

    def set_normalizer(self, normalizer):
        """Dynamically restored normalizer fields follow the policy device."""
        super().set_normalizer(normalizer)
        self.normalizer.to(self.device)

    def prepare_denoising_sample(self, batch, noise=None, timesteps=None, generator=None):
        """Use valid 8D derived targets and pair every intervention with identical noise."""
        actions = batch["action"]
        if tuple(actions.shape[1:]) != (self.horizon, self.action_dim):
            raise ValueError("Expected [B,H,8] derived action")
        mask = batch["action_loss_mask"].to(actions)
        if mask.shape != actions.shape or not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.sum((1, 2)) > 0):
            raise ValueError("Invalid action validity mask")
        clean = self.normalizer["action"].normalize(actions)
        clean = torch.where(mask.bool(), clean, torch.zeros_like(clean))
        if not torch.isfinite(clean).all():
            raise ValueError("Non-finite action target")
        if noise is None:
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        if noise.shape != clean.shape or noise.device != clean.device or noise.dtype != clean.dtype or not torch.isfinite(noise).all():
            raise ValueError("Invalid paired diffusion noise")
        if timesteps is None:
            timesteps = torch.randint(self.noise_scheduler.config.num_train_timesteps, (len(actions),), device=actions.device, generator=generator)
        if timesteps.dtype != torch.long or timesteps.shape != (len(actions),) or timesteps.device != actions.device:
            raise ValueError("Invalid diffusion timestep shape/device")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.noise_scheduler.config.num_train_timesteps):
            raise ValueError("Diffusion timestep out of range")
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else clean
        return DenoisingSample(self.noise_scheduler.add_noise(clean, noise, timesteps), noise, timesteps, target, mask)

    def compute_loss_per_sample(self, batch, condition_ids=None, denoising_sample=None):
        batch_size = len(batch["action"])
        if condition_ids is None:
            condition_ids = self.sample_condition_ids(batch_size) if self.training else torch.zeros(batch_size, dtype=torch.long, device=self.device)
        if condition_ids.dtype != torch.long or condition_ids.shape != (batch_size,) or torch.any(condition_ids < 0) or torch.any(condition_ids >= len(CONDITION_NAMES)):
            raise ValueError("Unknown OpenViTac intervention")
        presence = self.condition_presence[condition_ids.to(self.device)]
        conditioning = self.obs_encoder(self.normalizer.normalize(batch["obs"]), presence)
        sample = self.prepare_denoising_sample(batch) if denoising_sample is None else denoising_sample
        return self.denoising_loss_per_sample(sample, conditioning)

    @torch.no_grad()
    def predict_action(self, obs_dict, presence=None, generator=None):
        conditioning = self.obs_encoder(self.normalizer.normalize(obs_dict), presence)
        return self.sample_from_condition(conditioning, generator)

    @torch.no_grad()
    def sample_from_condition(self, conditioning, generator=None):
        """Expose the same chunk interface to Stage 0 and Stage 2."""
        trajectory = torch.randn((len(conditioning), self.horizon, self.action_dim), device=self.device,
                                 dtype=self.dtype, generator=generator)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for step in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, step, global_cond=conditioning)
            trajectory = self.noise_scheduler.step(prediction, step, trajectory, generator=generator).prev_sample
        actions = self.normalizer["action"].unnormalize(trajectory)
        start = self.n_obs_steps - 1
        return {"action": actions[:, start:start + self.n_action_steps], "action_pred": actions}
