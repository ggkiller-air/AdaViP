"""Masked action-space policies for ManiFeel multi-task training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from einops import reduce

try:
    from manifeel.policy.fm_diffusion_policy import FMDP
except ImportError:  # pragma: no cover - only exercised without the FM dependency.
    FMDP = None


class MaskedDiffusionUnetImagePolicy(DiffusionUnetImagePolicy):
    """Diffusion policy that ignores padded action dimensions in the loss."""

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = self.obs_encoder(this_nobs)
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = (~condition_mask).type(noisy_trajectory.dtype)
        if "action_loss_mask" in batch:
            batch_action_loss_mask = batch["action_loss_mask"].to(
                device=loss_mask.device,
                dtype=loss_mask.dtype,
            )
            if self.obs_as_global_cond:
                loss_mask = loss_mask * batch_action_loss_mask
            else:
                padded_mask = torch.zeros_like(loss_mask)
                padded_mask[..., : self.action_dim] = batch_action_loss_mask
                loss_mask = loss_mask * padded_mask

        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        pred = self.model(noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask
        loss = reduce(loss, "b ... -> b", "sum")
        denom = reduce(loss_mask, "b ... -> b", "sum").clamp_min(1.0)
        return (loss / denom).mean()


if FMDP is not None:

    class MaskedFMDP(FMDP):
        """Flow Matching policy that ignores padded action dimensions in the loss."""

        def compute_loss(self, batch):
            assert "valid_mask" not in batch
            nobs = self.normalizer.normalize(batch["obs"])
            nactions = self.normalizer["action"].normalize(batch["action"])
            batch_size = nactions.shape[0]
            horizon = nactions.shape[1]

            global_cond = None
            trajectory = nactions
            cond_data = trajectory
            if self.obs_as_global_cond:
                this_nobs = dict_apply(
                    nobs,
                    lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
                )
                nobs_features = self.obs_encoder(this_nobs)
                global_cond = nobs_features.reshape(batch_size, -1)
            else:
                this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs)
                nobs_features = nobs_features.reshape(batch_size, horizon, -1)
                cond_data = torch.cat([nactions, nobs_features], dim=-1)
                trajectory = cond_data.detach()

            x0 = torch.randn(trajectory.shape, device=trajectory.device)
            timestep, xt, ut = self.FM.sample_location_and_conditional_flow(
                x0, trajectory
            )
            vt = self.model(sample=xt, timestep=timestep, global_cond=global_cond)

            loss = F.mse_loss(vt, ut, reduction="none")
            loss_mask = torch.ones_like(loss)
            if "action_loss_mask" in batch:
                action_loss_mask = batch["action_loss_mask"].to(
                    device=loss.device,
                    dtype=loss.dtype,
                )
                if self.obs_as_global_cond:
                    loss_mask = loss_mask * action_loss_mask
                else:
                    action_mask = torch.ones_like(loss_mask)
                    action_mask[..., : self.action_dim] = action_loss_mask
                    loss_mask = loss_mask * action_mask

            loss = loss * loss_mask
            loss = reduce(loss, "b ... -> b", "sum")
            denom = reduce(loss_mask, "b ... -> b", "sum").clamp_min(1.0)
            return (loss / denom).mean()

else:

    class MaskedFMDP:
        """Placeholder that explains the missing Flow Matching dependency."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MaskedFMDP requires manifeel.policy.fm_diffusion_policy and torchcfm."
            )
