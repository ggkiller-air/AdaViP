"""Paired L1 K64 utility scoring with the established window and draw protocol."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from adavip.manifeel.utility_label_store import window_draws


@torch.inference_mode()
def mask_conditionings(policy: object, observations: dict) -> Tensor:
    """Use yesterday's L1 zero-token/flag/late-language intervention without new parameters."""
    encoder = policy.obs_encoder
    if encoder.variant != 'l1':
        raise ValueError('Only the original L1 late-language encoder is supported')
    normalized = policy.normalizer.normalize(observations)
    tokens = encoder.encode_sensor_tokens(normalized)
    language = encoder.language_embedding(normalized)
    result = []
    for flags in policy.condition_presence:
        presence = flags.expand(len(tokens), -1)
        masked = torch.where(presence[:, :6, None], tokens, torch.zeros_like(tokens))
        state = encoder.fusion(torch.cat((masked.flatten(1), presence[:, :6].to(tokens.dtype)), dim=-1))
        masked_language = torch.where(presence[:, 6:7], language, torch.zeros_like(language))
        result.append(encoder.condition_state(state, masked_language))
    conditions = torch.stack(result)
    torch.testing.assert_close(conditions[0], encoder(normalized), rtol=0, atol=0)
    return conditions


class ReferenceK64Draws:
    """Reuse the legacy K64 pilot overlap and deterministic per-window streams."""

    def __init__(self, root: object) -> None:
        from pathlib import Path
        import json

        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.reuse = np.load(self.root / "metadata/pilot_reuse_rows.npy", mmap_mode="r")
        self.pilot_steps = np.load(self.root / "metadata/pilot_timesteps.npy", mmap_mode="r")
        self.pilot_noise = np.load(self.root / "metadata/pilot_noise.npy", mmap_mode="r")

    def get(self, window_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return timesteps/noise independent of minibatch and continuation boundaries."""
        timesteps, noise = [], []
        for window_id in window_ids:
            window_id = int(window_id)
            if not 0 <= window_id < len(self.reuse):
                raise IndexError(window_id)
            reused = int(self.reuse[window_id])
            if reused >= 0:
                timesteps.append(np.asarray(self.pilot_steps[reused]))
                noise.append(np.asarray(self.pilot_noise[reused]))
            else:
                steps, seed = window_draws(window_id, self.manifest["draws"]["timestep_seed"],
                                          self.manifest["draws"]["noise_seed"])
                timesteps.append(steps)
                noise.append(torch.randn((64, 16, 7), generator=torch.Generator().manual_seed(seed)).numpy())
        return np.stack(timesteps), np.stack(noise)


@torch.inference_mode()
def score_conditionings(policy: object, batch: dict, conditions: Tensor, noise: np.ndarray,
                       timesteps: np.ndarray, denoise_batch_size: int = 256) -> np.ndarray:
    """Batch noise draws while pairing the same action/noise/timestep across all masks."""
    count, draws = timesteps.shape
    if (noise.shape != (count, draws, policy.horizon, 7)
            or conditions.shape[:2] != (10, count) or len(batch["action"]) != count
            or denoise_batch_size < 1):
        raise ValueError("Invalid paired utility tensor shapes or denoising batch size")
    device = batch["action"].device
    flat_noise = torch.as_tensor(noise.reshape(-1, policy.horizon, 7), device=device)
    flat_steps = torch.as_tensor(timesteps.reshape(-1), device=device)
    values = torch.empty((count * draws, 10), dtype=torch.float32, device=device)
    for start in range(0, count * draws, denoise_batch_size):
        stop = min(start + denoise_batch_size, count * draws)
        windows = torch.arange(start, stop, device=device) // draws
        expanded = {name: batch[name][windows] for name in ("action", "action_loss_mask")}
        sample = policy.prepare_denoising_sample(expanded, noise=flat_noise[start:stop],
                                                 timesteps=flat_steps[start:stop])
        for condition in range(10):
            values[start:stop, condition] = policy.denoising_loss_per_sample(
                sample, conditions[condition, windows])
    if not torch.isfinite(values).all() or (values < 0).any():
        raise FloatingPointError("Invalid paired K64 losses")
    return values.reshape(count, draws, 10).cpu().numpy()


@torch.inference_mode()
def paired_k64_losses(policy: object, batch: dict, noise: np.ndarray, timesteps: np.ndarray,
                      denoise_batch_size: int = 256) -> np.ndarray:
    """Cache sensor encodings once and predict all ten paired L1 mask conditions."""
    return score_conditionings(policy, batch, mask_conditionings(policy, batch["obs"]),
                              noise, timesteps, denoise_batch_size)


@torch.inference_mode()
def reference_paired_losses(policy: object, batch: dict, conditions: Tensor,
                           noise: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    """Evaluate one draw at a time as an independent correctness reference."""
    result = []
    device = batch["action"].device
    for draw in range(timesteps.shape[1]):
        sample = policy.prepare_denoising_sample(
            batch, noise=torch.as_tensor(noise[:, draw].copy(), device=device),
            timesteps=torch.as_tensor(timesteps[:, draw].copy(), device=device))
        result.append(torch.stack([policy.denoising_loss_per_sample(sample, condition)
                                   for condition in conditions], dim=-1))
    return torch.stack(result, dim=1).cpu().numpy()
