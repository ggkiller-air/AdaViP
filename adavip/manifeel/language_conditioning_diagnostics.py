"""Fixed training probes and task-pair gradients, isolated from optimizer state."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import default_collate

from diffusion_policy.common.pytorch_util import dict_apply


def select_probe_indices(dataset: object, per_task: int, seed: int) -> list[list[int]]:
    """Select complete windows across episodes and trajectory phases once."""
    candidates = [[] for _ in dataset.task_specs]
    previous = 0
    for record, cumulative in zip(dataset.records, dataset.cumulative_lengths):
        task, _, start, end = record
        first, last = dataset.n_obs_steps - 1, end - start - dataset.horizon + dataset.n_obs_steps - 1
        candidates[task].extend(range(previous + first, previous + last + 1))
        previous = cumulative
    rng = np.random.default_rng(seed)
    return [rng.choice(rows, size=min(per_task, len(rows)), replace=False).tolist() for rows in candidates]


def gradient_matrix(gradients: list[torch.Tensor]) -> dict:
    """Exclude zero-norm pairs and count negative unordered task pairs."""
    values = torch.stack(gradients).double().cpu()
    norms = values.norm(dim=1)
    valid = norms > 1e-12
    matrix = (values @ values.T) / (norms[:, None] * norms[None, :]).clamp_min(1e-24)
    matrix = matrix.clamp(-1, 1)
    pairs = torch.triu(valid[:, None] & valid[None, :], diagonal=1)
    return {"cosine": [[float(matrix[i, j]) if valid[i] and valid[j] else None
                        for j in range(len(values))] for i in range(len(values))],
            "gradient_norms": norms.tolist(), "valid_pairs": int(pairs.sum()),
            "negative_pairs": int(((matrix < 0) & pairs).sum()),
            "negative_pair_fraction": float(((matrix < 0) & pairs).sum() / pairs.sum()) if pairs.any() else None}


def measure_probes(policy: object, dataset: object, indices: list[list[int]],
                   gradient: bool, noise_repeats: int = 4, gradient_samples: int = 8,
                   batch_size: int = 8, seed: int = 2042) -> dict:
    """Measure raw weights using fixed windows/noise/timesteps without training side effects."""
    was_training = policy.training
    encoder = policy.obs_encoder
    parameters = list(encoder.fusion.parameters())
    gradients, task_losses = [], []
    try:
        policy.eval()
        for task_index, rows in enumerate(indices):
            if not rows:
                raise ValueError("Every task needs at least one complete training probe")
            total_loss, count = 0.0, 0
            accumulated = torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64)
            gradient_count = 0
            for offset in range(0, len(rows), batch_size):
                selected = rows[offset:offset + batch_size]
                batch = dict_apply(default_collate([dataset[row] for row in selected]),
                                   lambda tensor: tensor.to(policy.device))
                with torch.no_grad():
                    normalized = policy.normalizer.normalize(batch["obs"])
                    tokens = encoder.encode_sensor_tokens(normalized)
                    language = encoder.language_embedding(normalized)
                for repeat in range(noise_repeats):
                    generator = torch.Generator(device=policy.device).manual_seed(
                        seed + task_index * 100000 + offset * 100 + repeat)
                    sample = policy.prepare_denoising_sample(batch, generator=generator)
                    with torch.no_grad():
                        condition = encoder.condition_state(encoder.fuse_state_tokens(tokens), language)
                        losses = policy.denoising_loss_per_sample(sample, condition)
                        total_loss += float(losses.double().sum())
                        count += len(losses)
                    if gradient and offset < gradient_samples:
                        # Each graph is released immediately; CNN features stay detached.
                        size = min(len(selected), gradient_samples - offset)
                        with torch.enable_grad():
                            condition = encoder.condition_state(encoder.fuse_state_tokens(tokens), language)
                            loss = policy.denoising_loss_per_sample(sample, condition)[:size].mean()
                            values = torch.autograd.grad(loss, parameters)
                        accumulated += torch.cat([value.detach().flatten().cpu().double() for value in values]) * size
                        gradient_count += size
            task_losses.append(total_loss / count)
            if gradient:
                gradients.append(accumulated / gradient_count)
        result = {"weights": "raw", "per_task_action_loss": task_losses,
                  "macro_action_loss": float(np.mean(task_losses)),
                  "noise_repeats": noise_repeats, "samples_per_task": list(map(len, indices))}
        if gradient:
            result.update(gradient_matrix(gradients))
            result["gradient_scope"] = "obs_encoder.fusion; unclipped action gradients"
        return result
    finally:
        policy.train(was_training)
