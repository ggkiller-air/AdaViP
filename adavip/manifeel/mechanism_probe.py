"""CPU-only checks for modulation collapse and matched checkpoint protocols."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as functional


def modulation_variance(values: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    """Separate within-environment variation from between-environment differences."""
    if values.ndim != 3 or values.shape[:2] != valid.shape or valid.dtype != np.bool_:
        raise ValueError("expected [time,environment,channel] values and boolean validity")
    if not np.isfinite(values).all():
        raise ValueError("modulation values must be finite")
    means, variances, squares, saturations, derivatives = [], [], [], [], []
    for environment in np.flatnonzero(valid.any(axis=0)):
        selected = values[valid[:, environment], environment].astype(np.float64)
        means.append(selected.mean(axis=0))
        variances.append(selected.var(axis=0))
        squares.append(np.square(selected).mean(axis=0))
        saturations.append((np.abs(selected) >= 0.95).mean())
        derivatives.append((1 - np.square(selected)).mean())
    if not means:
        raise ValueError("no valid environments")
    within = float(np.mean(variances))
    between = float(np.asarray(means).var(axis=0).mean())
    energy = float(np.mean(squares))
    return {"environments": len(means), "within_environment_variance": within,
            "between_environment_variance": between, "total_equal_environment_variance": within + between,
            "within_fraction_of_variance": within / max(within + between, 1e-12),
            "within_rms_over_total_rms": float(np.sqrt(within / max(energy, 1e-12))),
            "saturated_fraction_abs_ge_095": float(np.mean(saturations)),
            "mean_tanh_local_derivative": float(np.mean(derivatives))}


def parameter_distance(reference: dict[str, torch.Tensor], changed: dict[str, torch.Tensor],
                       prefix: str) -> dict[str, object]:
    """Compare shared trained parameters without treating distance as performance attribution."""
    keys = [key for key in reference if key.startswith(prefix) and key in changed
            and reference[key].shape == changed[key].shape and reference[key].is_floating_point()]
    energy = difference = 0.0
    identical = True
    for key in keys:
        original, other = reference[key].double(), changed[key].double()
        energy += float(original.square().sum())
        difference += float((original - other).square().sum())
        identical = identical and torch.equal(reference[key], changed[key])
    return {"shared_tensors": len(keys), "identical": identical if keys else None,
            "relative_l2_distance": float(np.sqrt(difference / max(energy, 1e-12))) if keys else None}


def phase_matched_donors(phases: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Choose another environment at the same time and phase without future observations."""
    if not phases or any(mask.shape != phases[0].shape or mask.dtype != np.bool_ for mask in phases):
        raise ValueError("phase masks must have identical boolean shapes")
    if np.any(np.stack(phases).sum(axis=0) > 1):
        raise ValueError("phase masks must be disjoint")
    donors = np.broadcast_to(np.arange(phases[0].shape[1]), phases[0].shape).copy()
    paired = np.zeros_like(phases[0])
    for mask in phases:
        for step in range(len(mask)):
            environments = np.flatnonzero(mask[step])
            if len(environments) >= 2:
                donors[step, environments] = np.roll(environments, 1)
                paired[step, environments] = True
    return donors, paired


def fusion_attention_logits(module: torch.nn.MultiheadAttention, vision: torch.Tensor,
                            tactile: torch.Tensor) -> torch.Tensor:
    """Replay actual projected query/key scores before two-token attention softmax."""
    if module.training or not module.batch_first or not module._qkv_same_embed_dim:
        raise ValueError("requires evaluation-mode batch-first same-width attention")
    if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError("additional attention tokens are unsupported")
    if vision.shape != tactile.shape or vision.ndim != 2 or vision.shape[-1] != module.embed_dim:
        raise ValueError("modality features must share [batch,embed_dim] layout")
    query_weight, key_weight, _ = module.in_proj_weight.chunk(3, dim=0)
    if module.in_proj_bias is None:
        query_bias = key_bias = None
    else:
        query_bias, key_bias, _ = module.in_proj_bias.chunk(3, dim=0)
    tokens = torch.stack((vision, tactile), dim=1)
    head_width = module.embed_dim // module.num_heads
    query = functional.linear(vision, query_weight, query_bias).reshape(-1, module.num_heads, head_width)
    keys = functional.linear(tokens, key_weight, key_bias).reshape(-1, 2, module.num_heads, head_width)
    return torch.einsum("bhd,bkhd->bhk", query, keys) / np.sqrt(head_width)


def fixed_fusion_output(module: torch.nn.MultiheadAttention, vision: torch.Tensor,
                        tactile: torch.Tensor, tactile_weights: torch.Tensor) -> torch.Tensor:
    """Apply actual value/output projections with fixed per-head modality probabilities."""
    if tactile_weights.shape != (module.num_heads,) or not torch.isfinite(tactile_weights).all():
        raise ValueError("one finite tactile probability per head is required")
    if torch.any((tactile_weights < 0) | (tactile_weights > 1)):
        raise ValueError("tactile probabilities must be in [0,1]")
    if vision.shape != tactile.shape or vision.ndim != 2 or vision.shape[-1] != module.embed_dim:
        raise ValueError("modality features must share [batch,embed_dim] layout")
    if not module._qkv_same_embed_dim or module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError("only two-token same-width attention is supported")
    value_weight = module.in_proj_weight.chunk(3, dim=0)[2]
    value_bias = module.in_proj_bias.chunk(3, dim=0)[2] if module.in_proj_bias is not None else None
    tokens = torch.stack((vision, tactile), dim=1)
    projected = functional.linear(tokens, value_weight, value_bias).reshape(
        -1, 2, module.num_heads, module.embed_dim // module.num_heads)
    weights = tactile_weights.to(projected).reshape(1, module.num_heads, 1)
    mixed = ((1 - weights) * projected[:, 0] + weights * projected[:, 1]).flatten(start_dim=1)
    return module.out_proj(mixed)


def paired_success_difference(original: np.ndarray, changed: np.ndarray) -> dict[str, object]:
    """Bootstrap paired environments within fixed evaluation seeds, never frames."""
    if original.ndim != 2 or original.shape != changed.shape or original.dtype != np.bool_ or changed.dtype != np.bool_:
        raise ValueError("expected paired boolean outcomes [evaluation_seed,environment]")
    if not original.size:
        raise ValueError("paired outcomes must be nonempty")
    gained = int((changed & ~original).sum())
    lost = int((original & ~changed).sum())
    discordant = gained + lost
    probability = min(1.0, 2 * sum(math.comb(discordant, index) for index in range(min(gained, lost) + 1)) / 2**discordant) if discordant else 1.0
    difference = changed.astype(float) - original.astype(float)
    rng = np.random.default_rng(0)
    bootstraps = []
    for values in difference:
        indices = rng.integers(0, len(values), size=(20000, len(values)))
        bootstraps.append(values[indices].mean(axis=1))
    interval = np.quantile(np.mean(bootstraps, axis=0), [0.025, 0.975])
    return {"gained_successes": gained, "lost_successes": lost, "difference_pp": 100 * float(difference.mean()),
            "paired_environment_bootstrap_95_interval_pp": (100 * interval).tolist(),
            "exact_paired_sign_test_p_unadjusted": probability,
            "scope": "Exploratory; conditional on these evaluation seeds and this training checkpoint, not training-seed uncertainty."}
