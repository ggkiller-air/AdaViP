"""Paired input-mask diagnostics shared by L0, L1 and L1-dropout checkpoints."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from adavip.manifeel.full_sensor_schema import CONDITION_NAMES
from adavip.manifeel.l1_dropout import fuse_masked_l1_tokens
from adavip.manifeel.language_conditioned_obs_encoder import LanguageConditionedObsEncoder


@torch.no_grad()
def mask_conditionings(policy: object, observations: dict) -> Tensor:
    """Encode observations once and return all ten [condition,batch,d] conditions."""
    encoder = policy.obs_encoder
    normalized = policy.normalizer.normalize(observations)
    conditions = []
    if isinstance(encoder, LanguageConditionedObsEncoder):
        if encoder.variant != "l1":
            raise ValueError("This diagnostic supports L0, L1 and L1-dropout")
        tokens = encoder.encode_sensor_tokens(normalized)
        language = encoder.language_embedding(normalized)
        for flags in policy.condition_presence:
            presence = flags.expand(len(tokens), -1)
            state = fuse_masked_l1_tokens(encoder, tokens, presence[:, :6])
            masked_language = torch.where(presence[:, 6:7], language, torch.zeros_like(language))
            conditions.append(encoder.condition_state(state, masked_language))
    else:
        tokens = encoder.encode_tokens(normalized)
        for flags in policy.condition_presence:
            conditions.append(encoder.fuse_tokens(tokens, flags.expand(len(tokens), -1)))
    result = torch.stack(conditions)
    # Check the adapted full-input path against the checkpoint's native encoder.
    torch.testing.assert_close(result[0], encoder(normalized), rtol=0, atol=0)
    return result


def fixed_noise(indices: list[int], repeat: int, horizon: int, diffusion_steps: int,
                seed: int = 2042) -> tuple[Tensor, Tensor]:
    """Pair noise/timesteps across models, masks, batch sizes and continuation jobs."""
    noise, timesteps = [], []
    for index in indices:
        generator = torch.Generator().manual_seed(seed + index * 1009 + repeat * 1000000007)
        noise.append(torch.randn(horizon, 7, generator=generator))
        timesteps.append(torch.randint(diffusion_steps, (), generator=generator))
    return torch.stack(noise), torch.stack(timesteps)


@torch.no_grad()
def paired_mask_losses(policy: object, batch: dict, indices: list[int], repeats: int = 4,
                       seed: int = 2042) -> tuple[np.ndarray, np.ndarray]:
    """Return [B,K,10] losses; every mask reuses the same noisy action and timestep."""
    conditions = mask_conditionings(policy, batch["obs"])
    losses, timesteps = [], []
    for repeat in range(repeats):
        noise, steps = fixed_noise(indices, repeat, policy.horizon,
                                   policy.noise_scheduler.config.num_train_timesteps, seed)
        sample = policy.prepare_denoising_sample(batch, noise=noise.to(policy.device),
                                                 timesteps=steps.to(policy.device))
        values = torch.stack([policy.denoising_loss_per_sample(sample, condition) for condition in conditions], dim=-1)
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite paired denoising loss")
        losses.append(values.cpu().numpy())
        timesteps.append(steps.numpy())
    return np.stack(losses, axis=1), np.stack(timesteps, axis=1)


def summarize_losses(losses: np.ndarray) -> dict:
    """Preserve signed effects and report repeat consistency instead of clipping utility."""
    if losses.ndim != 3 or losses.shape[2] != len(CONDITION_NAMES) or not np.isfinite(losses).all():
        raise ValueError("Expected finite [samples,repeats,10] paired losses")
    mean = losses.astype(np.float64).mean(axis=1)
    delta = mean[:, 1:] - mean[:, :1]
    full = float(mean[:, 0].mean())
    result = {"samples": len(losses), "mean_loss": mean.mean(axis=0).tolist(), "mean_delta": delta.mean(axis=0).tolist(),
              "relative_mean_delta": (delta.mean(axis=0) / max(full, 1e-12)).tolist(),
              "median_delta": np.median(delta, axis=0).tolist(),
              "delta_p05": np.quantile(delta, 0.05, axis=0).tolist(),
              "delta_p95": np.quantile(delta, 0.95, axis=0).tolist(),
              "positive_fraction": (delta > 0).mean(axis=0).tolist(),
              "zero_fraction": (delta == 0).mean(axis=0).tolist()}
    if losses.shape[1] > 1:
        repeat_delta = losses[:, :, 1:] - losses[:, :, :1]
        consistent = np.all(repeat_delta > 0, axis=1) | np.all(repeat_delta < 0, axis=1)
        result["all_repeat_sign_agreement_fraction"] = consistent.mean(axis=0).tolist()
        result["mean_repeat_delta_std"] = repeat_delta.std(axis=1).mean(axis=0).tolist()
    return result
