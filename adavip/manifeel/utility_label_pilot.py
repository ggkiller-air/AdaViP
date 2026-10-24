"""Weighted pilot labels and trajectory-safe regression diagnostics, without Student training."""

from __future__ import annotations

import numpy as np

from adavip.manifeel.utility_k64_precision import BASE_COUNTS, STRATA, STRATUM_WEIGHTS


def validate_pilot_windows(windows: list[dict]) -> None:
    """Require the authorized nine-task, ten-trajectory, 32-anchor pilot without duplicates."""
    if len(windows) != 2880 or len({window["task_id"] for window in windows}) != 9:
        raise ValueError("Only the fixed2880-window nine-task pilot is authorized")
    identities = {(window["task_id"], window["episode_index"], window["timestep"]) for window in windows}
    if len(identities) != 2880:
        raise ValueError("Duplicate pilot source anchors")
    for task in dict.fromkeys(window["task_id"] for window in windows):
        selected = [window for window in windows if window["task_id"] == task]
        episodes = {window["episode_index"] for window in selected}
        if len(episodes) != 10 or len(selected) != 320:
            raise ValueError("Expected ten trajectories per task")
        for episode in episodes:
            trajectory = sorted((window for window in selected if window["episode_index"] == episode),
                                key=lambda window: window["point"])
            if [window["point"] for window in trajectory] != list(range(32)):
                raise ValueError("Expected32 distinct ordered points per trajectory")
            for window in trajectory:
                if not 1 <= window["timestep"] <= window["episode_length"] - 15:
                    raise ValueError("Pilot anchor does not have a complete horizon16/history2")


def aggregate_losses(losses: np.ndarray, timesteps: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce raw64 draws by stratum length, saving signed utility and analytic variance."""
    losses = np.asarray(losses, dtype=np.float64)
    if losses.ndim != 3 or losses.shape[1:] != (64, 10) or timesteps.shape != losses.shape[:2]:
        raise ValueError("Expected [windows,64,10] losses and matching timesteps")
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("Raw denoising losses must be finite and nonnegative")
    blocks = timesteps.reshape(len(losses), 4, 16)
    offset = 0
    for (low, high), samples in zip(STRATA, BASE_COUNTS):
        selected = blocks[:, :, offset:offset + samples]
        if not ((selected >= low) & (selected < high)).all():
            raise ValueError("Draw order does not match the frozen stratifiedK64 protocol")
        offset += samples
    draw_utility = losses[:, :, 1:] - losses[:, :, :1]
    means, variances = [], []
    for start in (0, 32):
        total = np.zeros((len(losses), 10), dtype=np.float64)
        variance = np.zeros((len(losses), 9), dtype=np.float64)
        for (low, high), samples, weight in zip(STRATA, BASE_COUNTS, STRATUM_WEIGHTS):
            mask = (timesteps[:, start:start + 32] >= low) & (timesteps[:, start:start + 32] < high)
            selected = losses[:, start:start + 32][mask].reshape(len(losses), samples * 2, 10)
            differences = selected[:, :, 1:] - selected[:, :, :1]
            total += weight * selected.mean(axis=1)
            variance += weight ** 2 * differences.var(axis=1, ddof=1) / (samples * 2)
        means.append(total)
        variances.append(variance)
    weighted = np.zeros((len(losses), 10), dtype=np.float64)
    variance = np.zeros((len(losses), 9), dtype=np.float64)
    stratum_means, stratum_variances = [], []
    for (low, high), samples, weight in zip(STRATA, BASE_COUNTS, STRATUM_WEIGHTS):
        mask = (timesteps >= low) & (timesteps < high)
        selected = losses[mask].reshape(len(losses), samples * 4, 10)
        differences = selected[:, :, 1:] - selected[:, :, :1]
        mean = selected.mean(axis=1)
        component = weight ** 2 * differences.var(axis=1, ddof=1) / (samples * 4)
        weighted += weight * mean
        variance += component
        stratum_means.append(mean)
        stratum_variances.append(component)
    utility = weighted[:, 1:] - weighted[:, :1]
    halves = np.stack(means, axis=1)
    np.testing.assert_allclose(utility, (halves[:, :, 1:] - halves[:, :, :1]).mean(axis=1), atol=1e-14)
    return {"utility": utility, "weighted_losses": weighted, "utility_variance": variance,
            "k32_utility": halves[:, :, 1:] - halves[:, :, :1],
            "k32_variance": np.stack(variances, axis=1),
            "stratum_loss_means": np.stack(stratum_means, axis=1),
            "stratum_utility_variance_contributions": np.stack(stratum_variances, axis=1)}


def trajectory_split(windows: list[dict], seed: int = 20261066) -> np.ndarray:
    """Reserve two complete trajectories per task for downstream diagnostic validation only."""
    validate_pilot_windows(windows)
    split = np.zeros(len(windows), dtype=np.int8)
    for task_index, task in enumerate(dict.fromkeys(window["task_id"] for window in windows)):
        episodes = sorted({window["episode_index"] for window in windows if window["task_id"] == task})
        rng = np.random.default_rng(np.random.SeedSequence([seed, task_index]))
        held_out = set(rng.choice(episodes, size=2, replace=False).tolist())
        for index, window in enumerate(windows):
            if window["task_id"] == task and window["episode_index"] in held_out:
                split[index] = 1
    return split


def fit_label_normalization(utility: np.ndarray, split: np.ndarray,
                            scale_floor: float = 1e-6) -> dict:
    """Fit pooled per-dimension label statistics on diagnostic-train trajectories only."""
    values = np.asarray(utility, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9 or split.shape != (len(values),):
        raise ValueError("Expected nine-dimensional labels and a matching split")
    if not np.isfinite(values).all() or not set(np.unique(split)).issubset({0, 1}) or scale_floor <= 0:
        raise ValueError("Invalid label normalization inputs")
    if not np.any(split == 0) or not np.any(split == 1):
        raise ValueError("Need both diagnostic-train and diagnostic-validation trajectories")
    train = values[split == 0]
    standard_deviation = train.std(axis=0)
    return {"method": "pooled per-dimension z-score; not per-task; inverse transform preserves signed raw units",
            "mean": train.mean(axis=0).tolist(), "std_before_floor": standard_deviation.tolist(),
            "scale": np.maximum(standard_deviation, scale_floor).tolist(), "scale_floor": scale_floor,
            "train_windows": len(train), "validation_windows": int((split == 1).sum()),
            "scope": "pilot diagnostic candidate only; refit on actual Stage1 training trajectories, never validation/test",
            "teacher_normalizer_modified": False}


def correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    """Return a descriptive correlation only when both curves vary."""
    return float(np.corrcoef(first, second)[0, 1]) if min(first.std(), second.std()) > 0 else None


def distribution_metrics(values: np.ndarray, variance: np.ndarray, halves: np.ndarray) -> dict:
    """Describe signed magnitude, noise, outliers and two independentK32 halves."""
    values, variance, halves = (np.asarray(array, dtype=np.float64) for array in (values, variance, halves))
    if values.ndim != 1 or variance.shape != values.shape or halves.shape != (len(values), 2):
        raise ValueError("Invalid scalar-dimension diagnostic inputs")
    mean, standard_deviation = float(values.mean()), float(values.std())
    noise = float(variance.mean())
    energy = max(0.0, float(np.square(values).mean()) - noise)
    centered_variance = max(0.0, float(values.var()) - noise)
    error = 2 * np.sqrt(variance)
    residual_energy = np.square(values - mean)
    count = max(1, int(np.ceil(len(values) * 0.01)))
    opposite = halves[:, 0] * halves[:, 1] < 0
    return {"windows": len(values), "mean": mean, "std": standard_deviation,
            **{f"q{int(quantile*100):02d}": float(np.quantile(values, quantile))
               for quantile in (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1)},
            "negative_fraction": float((values < 0).mean()), "exact_zero_fraction": float((values == 0).mean()),
            **{f"near_zero_{threshold:g}_fraction": float((np.abs(values) <= threshold).mean())
               for threshold in (1e-5, 1e-4, 1e-3)},
            "analytic_k64_noise_mse": noise,
            "noise_to_signal_rms": float(np.sqrt(noise / energy)) if energy > 0 else None,
            "noise_to_centered_signal_sd": float(np.sqrt(noise / centered_variance)) if centered_variance > 0 else None,
            "resolved_near_zero_fraction": float((np.abs(values) + error <= 1e-4).mean()),
            "resolved_nonzero_fraction": float((np.abs(values) - error > 1e-4).mean()),
            "unresolved_fraction": float(((np.abs(values) + error > 1e-4) & (np.abs(values) - error <= 1e-4)).mean()),
            "independent_k32_pearson": correlation(halves[:, 0], halves[:, 1]),
            "independent_k32_rmse": float(np.sqrt(np.square(halves[:, 0] - halves[:, 1]).mean())),
            "independent_k32_raw_sign_flip_fraction": float(opposite.mean()),
            "independent_k32_material_sign_flip_fraction": float(
                (opposite & (np.abs(halves[:, 0]) > 1e-4) & (np.abs(halves[:, 1]) > 1e-4)).mean()),
            "top1pct_centered_squared_energy_fraction": float(np.sort(residual_energy)[-count:].sum() / residual_energy.sum())
            if residual_energy.sum() > 0 else 0.0,
            "beyond_3sd_fraction": float((np.abs(values - mean) > 3 * standard_deviation).mean())
            if standard_deviation > 0 else 0.0}


def regression_metrics(utility: np.ndarray, variance: np.ndarray, split: np.ndarray,
                       normalization: dict, dimensions: list[str]) -> list[dict]:
    """Compare constant-baseline loss/gradient sensitivity, not learned Student performance."""
    mean, scale = np.array(normalization["mean"]), np.array(normalization["scale"])
    rows = []
    for name, mask in (("train", split == 0), ("validation", split == 1)):
        residual = utility[mask] - mean
        squared = np.square(residual)
        normalized = residual / scale
        huber = np.where(np.abs(normalized) <= 1, 0.5 * np.square(normalized), np.abs(normalized) - 0.5)
        raw_mse = squared.mean(axis=0)
        for dimension_index, dimension in enumerate(dimensions):
            squared_dimension = np.square(normalized[:, dimension_index])
            count = max(1, int(np.ceil(mask.sum() * 0.01)))
            rows.append({"split": name, "dimension": dimension,
                         "raw_constant_mean_mse": float(raw_mse[dimension_index]),
                         "raw_squared_gradient_energy_share": float(raw_mse[dimension_index] / raw_mse.sum())
                         if raw_mse.sum() > 0 else 0.0,
                         "normalized_constant_mean_mse": float(squared_dimension.mean()),
                         "normalized_huber_delta1_constant_mean_loss": float(huber[:, dimension_index].mean()),
                         "huber_delta1_linear_fraction": float((np.abs(normalized[:, dimension_index]) > 1).mean()),
                         "huber_delta2_linear_fraction": float((np.abs(normalized[:, dimension_index]) > 2).mean()),
                         "normalized_noise_mse": float((variance[mask, dimension_index] / scale[dimension_index] ** 2).mean()),
                         "normalized_mse_top1pct_energy_fraction": float(np.sort(squared_dimension)[-count:].sum() / squared_dimension.sum())
                         if squared_dimension.sum() > 0 else 0.0,
                         "interpretation": "fixed train-mean prediction sensitivity only; no model fitted; MSE and Huber values not directly comparable"})
    return rows
