"""Optional same-process multi-device losses with one global batch and RNG stream."""

from __future__ import annotations

from dataclasses import fields

import torch
from torch import nn

from adavip.manifeel.reference_diffusion_policy import DenoisingSample


class Stage2LossModule(nn.Module):
    """Expose only per-example losses so gathering never averages device means."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict, stage: str, sample_tensors: tuple | None):
        sample = None if sample_tensors is None else DenoisingSample(*sample_tensors)
        result = self.policy.training_losses(batch, stage, sample)
        utility = nn.functional.smooth_l1_loss(result["u_hat"], result["utility_target"],
                                               reduction="none", beta=1.0)
        return result["action_loss_per_sample"], utility


class MultiDeviceLosses:
    """Keep the bare policy, optimizer and EMA on device0; split the unchanged global batch across devices."""

    def __init__(self, policy, device_ids=(0, 1)):
        if policy.ablation != "C_teacher_residual":
            raise ValueError("This bounded parallel experiment is residual C only")
        self.policy = policy
        self.parallel = nn.DataParallel(Stage2LossModule(policy), device_ids=list(device_ids),
                                        output_device=device_ids[0] if device_ids else None)

    def __call__(self, batch: dict, stage: str, sample=None) -> dict:
        # Generate the original global noise/timesteps before scattering. Replica
        # forward paths then consume no randomness, including unequal final batches.
        if stage != "A" and sample is None:
            sample = self.policy.prepare_denoising_sample(batch)
        tensors = None if sample is None else tuple(getattr(sample, f.name) for f in fields(sample))
        action, utility = self.parallel(batch, stage, tensors)
        return dict(action_loss=action.mean(), utility_loss=utility.mean(),
                    action_loss_per_sample=action, utility_loss_per_sample=utility.mean(dim=-1))
