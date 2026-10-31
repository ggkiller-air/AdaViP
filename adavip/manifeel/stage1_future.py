"""Training-only per-sensor future heads, target EMA and loss diagnostics."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from adavip.manifeel.full_sensor_schema import CAMERA_KEYS, TACTILE_KEYS
from adavip.manifeel.stage1_future_dataset import EXPERIMENTS


class PixelDecoder(nn.Module):
    """Decode a single z_t into one RGB frame in [0,1], without skip inputs."""

    def __init__(self, token_dim: int, resolution: int = 64) -> None:
        super().__init__()
        if resolution < 8 or resolution & (resolution - 1):
            raise ValueError("Raw resolution must be a power of two >= 8")
        self.projection = nn.Linear(token_dim, 128 * 4 * 4)
        blocks = []
        channels = 128
        for _ in range(int(math.log2(resolution)) - 2):
            output = max(16, channels // 2)
            blocks.extend((nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                           nn.Conv2d(channels, output, 3, padding=1),
                           nn.GroupNorm(math.gcd(8, output), output), nn.SiLU()))
            channels = output
        self.decoder = nn.Sequential(*blocks, nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, z_t: Tensor) -> Tensor:
        """Return one independently decoded image for each sample."""
        return self.decoder(self.projection(z_t).reshape(-1, 128, 4, 4))


class Stage1FutureHeads(nn.Module):
    """Keep auxiliary parameters outside the unchanged inference policy."""

    def __init__(self, obs_encoder: nn.Module, experiment: str = "A", raw_resolution: int = 64,
                 momentum: float = 0.996, head_seed: int = 17042) -> None:
        super().__init__()
        if experiment not in EXPERIMENTS or not 0 <= momentum < 1:
            raise ValueError("Invalid experiment or target EMA momentum")
        self.target_space, self.sensor_keys = EXPERIMENTS[experiment]
        self.momentum, self.raw_resolution = momentum, raw_resolution
        self.heads, self.target_encoders = nn.ModuleDict(), nn.ModuleDict()
        # Do not perturb either the common policy initialization or training RNG.
        with torch.random.fork_rng(devices=[]):
            for key in self.sensor_keys:
                torch.manual_seed(head_seed + (*CAMERA_KEYS, *TACTILE_KEYS).index(key))
                if self.target_space == "raw":
                    self.heads[key] = PixelDecoder(obs_encoder.token_dim, raw_resolution)
                else:
                    self.heads[key] = nn.Sequential(
                        nn.Linear(obs_encoder.token_dim, 512), nn.SiLU(),
                        nn.Linear(512, obs_encoder.image_feature_dim))
                    self.target_encoders[key] = copy.deepcopy(obs_encoder.image_encoders[key])
        self.target_encoders.requires_grad_(False)
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.target_encoders.eval()

    def train(self, mode: bool = True) -> "Stage1FutureHeads":
        """Keep target normalization and stochastic layers in evaluation mode."""
        super().train(mode)
        self.target_encoders.eval()
        return self

    @torch.no_grad()
    def update_targets(self, obs_encoder: nn.Module) -> None:
        """EMA parameters and floating buffers; copy integer counters after an update."""
        if self.target_space != "latent":
            return
        for key, target in self.target_encoders.items():
            online = obs_encoder.image_encoders[key]
            online_parameters = dict(online.named_parameters())
            for name, value in target.named_parameters():
                value.mul_(self.momentum).add_(online_parameters[name], alpha=1 - self.momentum)
            online_buffers = dict(online.named_buffers())
            for name, value in target.named_buffers():
                source = online_buffers[name]
                if value.is_floating_point():
                    value.mul_(self.momentum).add_(source, alpha=1 - self.momentum)
                else:
                    value.copy_(source)
        self.ema_updates.add_(1)
        self.target_encoders.eval()

    def forward(self, z_t: Tensor, batch: dict, normalizer: nn.Module) -> dict:
        """Predict each valid future independently, reducing only at the loss level."""
        valid = batch["future_valid"].bool()
        zero = z_t.sum() * 0
        losses = {}
        for key in self.sensor_keys:
            if not valid.any():
                # Keep heads in the graph without evaluating invalid target images.
                losses[key] = self.heads[key](z_t).sum() * 0
                continue
            prediction = self.heads[key](z_t[valid])
            target_image = batch["future_obs"][key][valid]
            if self.target_space == "raw":
                if prediction.shape != target_image.shape:
                    raise ValueError(f"Raw target/prediction shape mismatch for {key}")
                losses[key] = F.mse_loss(prediction, target_image)
            else:
                with torch.no_grad():
                    target = self.target_encoders[key](normalizer[key].normalize(target_image))
                if prediction.shape != target.shape:
                    raise ValueError(f"Latent target/prediction shape mismatch for {key}")
                losses[key] = (1 - F.cosine_similarity(prediction, target, dim=-1, eps=1e-8)).mean()
        vision = torch.stack([losses[key] for key in CAMERA_KEYS if key in losses]).mean() if any(
            key in losses for key in CAMERA_KEYS) else zero
        tactile = torch.stack([losses[key] for key in TACTILE_KEYS if key in losses]).mean() if any(
            key in losses for key in TACTILE_KEYS) else zero
        future = 0.5 * (vision + tactile) if any(key in losses for key in CAMERA_KEYS) else tactile
        return {"future": future, "vision": vision, "tactile": tactile,
                "sensors": losses, "valid_count": int(valid.sum()), "sample_count": len(valid)}


def stage1_losses(policy: nn.Module, auxiliary: Stage1FutureHeads, batch: dict,
                  denoising_sample: object) -> dict:
    """Use the identical L1 action path and branch from its pre-language 256D state."""
    observations = policy.normalizer.normalize(batch["obs"])
    encoder = policy.obs_encoder
    z_t = encoder.fuse_state_tokens(encoder.encode_sensor_tokens(observations))
    condition = encoder.condition_state(z_t, encoder.language_embedding(observations))
    action = policy.denoising_loss_per_sample(denoising_sample, condition)
    return {"action_per_sample": action, "action": action.mean(), "z_t": z_t,
            **auxiliary(z_t, batch, policy.normalizer)}


def gradient_diagnostics(losses: dict, policy: nn.Module, weight: float = 1.0,
                         include_resnets: bool = False) -> dict:
    """Measure action/future gradients on shared fusion weights without .grad mutation."""
    parameters = list(policy.obs_encoder.fusion.parameters())
    action = torch.autograd.grad(losses["action"], parameters, retain_graph=True)
    future = torch.autograd.grad(losses["future"], parameters, retain_graph=True)
    action_vector = torch.cat([g.flatten() for g in action])
    future_vector = torch.cat([g.flatten() for g in future])
    action_norm, future_norm = action_vector.norm(), future_vector.norm()
    result = {"action_fusion_grad_norm": float(action_norm),
              "future_fusion_grad_norm": float(future_norm),
              "weighted_future_action_grad_ratio": float(weight * future_norm / action_norm.clamp_min(1e-20)),
              "fusion_gradient_cosine": float(F.cosine_similarity(
                  action_vector, future_vector, dim=0, eps=1e-20))}
    if include_resnets:
        selected = [next(module.parameters()) for module in policy.obs_encoder.image_encoders.values()]
        gradients = torch.autograd.grad(losses["future"], selected, retain_graph=True)
        result["future_resnet_first_weight_grad_norms"] = {
            key: float(g.norm()) for key, g in zip(policy.obs_encoder.image_encoders, gradients)}
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise FloatingPointError("Nonfinite gradient diagnostic")
    return result
