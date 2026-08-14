"""Success-rate aggregation helpers for ManiFeel evaluation."""

from __future__ import annotations

import numpy as np


def aggregate_success_rate(infos: dict) -> float:
    """Aggregate ManiFeel success info into a rollout success rate.

    Prefer info["success"] from each env because it preserves per-env success.
    MultiStepWrapper.get_infos() stores that as (num_envs, time, ...) for
    rollouts, so SR is the fraction of envs that succeed at least once.
    info["successes"] is a scalar task-side mean and is only a compatibility
    fallback when no per-env vector was recorded.
    """
    if "success" in infos:
        raw_success = np.asarray(infos["success"], dtype=np.float32)
        if raw_success.ndim == 0:
            if not np.isfinite(raw_success):
                return float("nan")
            return float(raw_success > 0.5)

        success = np.nan_to_num(raw_success, nan=0.0, posinf=0.0, neginf=0.0)
        if raw_success.ndim == 1:
            per_env_success = success > 0.5
        else:
            per_env_success = np.any(success.reshape(success.shape[0], -1) > 0.5, axis=1)
        return float(np.mean(per_env_success > 0.5))

    if "successes" in infos:
        raw_successes = np.asarray(infos["successes"], dtype=np.float32)
        if raw_successes.ndim == 0:
            if not np.isfinite(raw_successes):
                return float("nan")
            return float(raw_successes)
        successes = raw_successes[np.isfinite(raw_successes)]
        if successes.size == 0:
            return float("nan")
        return float(successes[-1])

    return float("nan")
