"""Weighted, independent-replicate precision diagnostics for uniform diffusion utility."""

from __future__ import annotations

from itertools import combinations

import numpy as np


STRATA = ((0, 12), (12, 25), (25, 50), (50, 75), (75, 100))
BASE_COUNTS = (4, 3, 3, 3, 3)
STRATUM_WEIGHTS = np.array([0.12, 0.13, 0.25, 0.25, 0.25], dtype=np.float64)


def make_timesteps(count: int, seed: int) -> np.ndarray:
    """Generate eight independent K16 blocks: four per independent K64 replica."""
    if count <= 0:
        raise ValueError("Window count must be positive")
    rng = np.random.default_rng(seed)
    return np.concatenate([
        np.concatenate([rng.integers(low, high, size=(count, samples), dtype=np.int64)
                        for (low, high), samples in zip(STRATA, BASE_COUNTS)], axis=1)
        for _ in range(8)
    ], axis=1)


def weighted_estimates(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint weighted K estimates and unbiased stratified sampling variances."""
    values = np.asarray(values, dtype=np.float64)
    if count not in (16, 32, 64) or values.ndim != 3 or values.shape[1] != 128:
        raise ValueError("Expected finite [windows,128,dimensions] values and K16/32/64")
    if not np.isfinite(values).all() or not values.shape[0] or not values.shape[2]:
        raise ValueError("Values must be nonempty and finite")
    blocks = values.reshape(len(values), 8, 16, values.shape[-1])
    block_count = count // 16
    means, variances = [], []
    for start in range(0, 8, block_count):
        estimate = np.zeros((len(values), values.shape[-1]), dtype=np.float64)
        variance = np.zeros_like(estimate)
        offset = 0
        for samples, weight in zip(BASE_COUNTS, STRATUM_WEIGHTS):
            selected = blocks[:, start:start + block_count, offset:offset + samples].reshape(
                len(values), block_count * samples, values.shape[-1])
            estimate += weight * selected.mean(axis=1)
            variance += weight ** 2 * selected.var(axis=1, ddof=1) / selected.shape[1]
            offset += samples
        means.append(estimate)
        variances.append(variance)
    return np.stack(means, axis=1), np.stack(variances, axis=1)


def precision_metrics(estimates: np.ndarray, variances: np.ndarray | None,
                      windows: list[dict], delta: float = 1e-4) -> dict:
    """Compare independent replicates, preserving near-zero and temporal noise distinctions."""
    estimates = np.asarray(estimates, dtype=np.float64)
    if estimates.ndim != 2 or estimates.shape[1] < 2 or len(windows) != len(estimates):
        raise ValueError("Need at least two independent estimates per window")
    if not np.isfinite(estimates).all():
        raise ValueError("Estimates must be finite")
    repeat_count = estimates.shape[1]
    empirical = estimates.var(axis=1, ddof=1)
    analytic = empirical if variances is None else np.asarray(variances, dtype=np.float64).mean(axis=1)
    if analytic.shape != (len(estimates),) or not np.isfinite(analytic).all() or np.any(analytic < 0):
        raise ValueError("Invalid per-window sampling variance")
    mean = estimates.mean(axis=1)
    noise = float(analytic.mean())
    empirical_noise = float(empirical.mean())
    signal_energy = max(0.0, float(np.square(mean).mean()) - noise / repeat_count)
    signal_variance = max(0.0, float(mean.var()) - noise / repeat_count)
    ratio = float(np.sqrt(noise / signal_energy)) if signal_energy > 0 else None
    centered_ratio = float(np.sqrt(noise / signal_variance)) if signal_variance > 0 else None
    pair_rmse, raw_flip, material_flip, correlations = [], [], [], []
    for first, second in combinations(range(repeat_count), 2):
        left, right = estimates[:, first], estimates[:, second]
        opposite = left * right < 0
        pair_rmse.append(float(np.sqrt(np.square(left - right).mean())))
        raw_flip.append(float(opposite.mean()))
        material_flip.append(float((opposite & (np.abs(left) > delta) & (np.abs(right) > delta)).mean()))
        if min(left.std(), right.std()) > 0:
            correlations.append(float(np.corrcoef(left, right)[0, 1]))
    selected_variance = np.broadcast_to(empirical[:, None], estimates.shape) if variances is None else variances
    error = 2 * np.sqrt(selected_variance)
    near = np.abs(estimates) + error <= delta
    nonzero = np.abs(estimates) - error > delta
    resolved = float((near | nonzero).mean())
    temporal_variances, temporal_noise = [], []
    trajectories = dict.fromkeys((window["task_id"], window["episode_index"]) for window in windows)
    for task, episode in trajectories:
        indices = [index for index, window in enumerate(windows)
                   if window["task_id"] == task and window["episode_index"] == episode]
        if len(indices) > 1:
            temporal_variances.append(float(mean[indices].var(ddof=1)))
            temporal_noise.append(float(analytic[indices].mean()) / repeat_count)
    observed_temporal = float(np.mean(temporal_variances)) if temporal_variances else None
    mean_temporal_noise = float(np.mean(temporal_noise)) if temporal_noise else None
    return {
        "replicates": repeat_count, "windows": len(estimates),
        "mean": float(mean.mean()), "primary_std": float(estimates[:, 0].std()),
        "negative_fraction": float((estimates < 0).mean()),
        "near_zero_1e5_fraction": float((np.abs(estimates) <= 1e-5).mean()),
        "near_zero_1e4_fraction": float((np.abs(estimates) <= delta).mean()),
        "near_zero_1e3_fraction": float((np.abs(estimates) <= 1e-3).mean()),
        "primary_near_zero_1e4_fraction": float((np.abs(estimates[:, 0]) <= delta).mean()),
        "estimated_noise_mse": noise, "empirical_replicate_noise_mse": empirical_noise,
        "analytic_to_empirical_variance_ratio": noise / empirical_noise if empirical_noise > 0 else None,
        "noise_to_signal_rms": ratio, "noise_to_centered_signal_sd": centered_ratio,
        "debiased_signal_second_moment": signal_energy, "debiased_between_window_variance": signal_variance,
        "pair_rmse_mean": float(np.mean(pair_rmse)), "pair_rmse_max": max(pair_rmse),
        "raw_sign_flip_mean": float(np.mean(raw_flip)), "raw_sign_flip_max": max(raw_flip),
        "material_sign_flip_mean": float(np.mean(material_flip)), "material_sign_flip_max": max(material_flip),
        "pair_pearson_mean": float(np.mean(correlations)) if correlations else None,
        "resolved_near_zero_fraction": float(near.mean()), "resolved_nonzero_fraction": float(nonzero.mean()),
        "unresolved_fraction": 1 - resolved,
        "observed_within_trajectory_variance_of_repeat_mean": observed_temporal,
        "estimated_mc_variance_of_repeat_mean": mean_temporal_noise,
        "excess_within_trajectory_variance": max(0.0, observed_temporal - mean_temporal_noise)
        if observed_temporal is not None else None,
        "heuristic_precision_gate": bool(ratio is not None and ratio <= 0.25 and resolved >= 0.9
                                         and max(material_flip) <= 0.05),
    }


def independent_reference_errors(estimates: np.ndarray, reference: np.ndarray,
                                 reference_variance: np.ndarray) -> dict:
    """Use only the opposite K64 replica as a noisy reference, never a shared-sample target."""
    if estimates.ndim != 2 or reference.shape != (len(estimates), 2) or estimates.shape[1] not in (2, 4, 8):
        raise ValueError("Expected balanced estimates from two independent K64 replicas")
    repeats_per_half = estimates.shape[1] // 2
    differences, reference_noise = [], []
    for repeat in range(estimates.shape[1]):
        opposite = 1 - repeat // repeats_per_half
        differences.append(estimates[:, repeat] - reference[:, opposite])
        reference_noise.append(reference_variance[:, opposite])
    difference = np.stack(differences, axis=1)
    observed_mse = float(np.square(difference).mean())
    reference_mse = float(np.mean(reference_noise))
    return {"opposite_k64_reference_rmse": float(np.sqrt(observed_mse)),
            "opposite_k64_reference_mean_difference": float(difference.mean()),
            "estimated_reference_noise_mse": reference_mse,
            "estimated_mse_after_reference_noise_subtraction": max(0.0, observed_mse - reference_mse),
            "reference_noise_subtraction_was_negative": observed_mse < reference_mse}
