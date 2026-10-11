"""Physical-contact alignment and environment-level summaries for Power Plug."""

from __future__ import annotations

import numpy as np


def contact_phase_masks(rewards: np.ndarray, force: np.ndarray, initial_force: np.ndarray,
                         chunk_steps: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """Align policy inputs to first plug/socket contact and exclude post-success samples."""
    if rewards.ndim != 2 or force.shape != (*rewards.shape, 3):
        raise ValueError("expected rewards [step, environment] and pairwise force [step, environment, 3]")
    if initial_force.shape != (rewards.shape[1], 3) or threshold <= 0:
        raise ValueError("initial force shape or positive contact threshold is invalid")
    if not np.isfinite(force).all() or not np.isfinite(initial_force).all():
        raise ValueError("contact forces must be finite")
    if not np.isin(rewards, (0, 1)).all():
        raise ValueError("Power Plug benchmark rewards must be binary reset/success flags")
    if chunk_steps.ndim != 1 or np.any(chunk_steps < 0) or np.any(chunk_steps > len(rewards)):
        raise ValueError("policy input steps are outside the recorded simulator history")
    force_norm = np.concatenate((np.linalg.norm(initial_force, axis=-1)[None],
                                 np.linalg.norm(force, axis=-1)), axis=0)
    contacted = force_norm > threshold
    successful = np.concatenate((np.zeros((1, rewards.shape[1]), dtype=bool), rewards > 0.5), axis=0)
    contacted_ever = np.maximum.accumulate(contacted, axis=0)
    succeeded_ever = np.maximum.accumulate(successful, axis=0)
    first_contact = np.where(contacted.any(axis=0), contacted.argmax(axis=0), -1)
    first_success = np.where(successful.any(axis=0), successful.argmax(axis=0), -1)
    success_at_input = succeeded_ever[chunk_steps]
    contact_at_input = contacted_ever[chunk_steps]
    return {"pre_contact": ~contact_at_input & ~success_at_input,
            "post_contact_pre_success": contact_at_input & ~success_at_input,
            "post_success": success_at_input,
            "first_contact_step": first_contact, "first_success_step": first_success,
            "force_norm": force_norm, "ever_success": succeeded_ever[-1]}


def environment_level_summary(values: np.ndarray, selected: np.ndarray) -> dict[str, object]:
    """Weight environments equally and bootstrap environments, never individual frames."""
    if values.shape[:2] != selected.shape or values.ndim != 3 or selected.dtype != np.bool_:
        raise ValueError("expected values [chunk, environment, metric] and a boolean selection mask")
    counts = selected.sum(axis=0)
    eligible = counts > 0
    means = np.full((selected.shape[1], values.shape[2]), np.nan)
    for environment in np.flatnonzero(eligible):
        means[environment] = values[selected[:, environment], environment].mean(axis=0)
    result: dict[str, object] = {"environments": int(eligible.sum()), "chunks": int(selected.sum()),
                                "chunks_by_environment": counts.tolist()}
    if not eligible.any():
        result.update({"mean": None, "bootstrap_95_interval": None})
        return result
    valid = means[eligible]
    rng = np.random.default_rng(0)
    bootstrap = valid[rng.integers(0, len(valid), size=(2000, len(valid)))].mean(axis=1)
    result.update({"mean": valid.mean(axis=0).tolist(),
                   "bootstrap_95_interval": np.quantile(bootstrap, (0.025, 0.975), axis=0).tolist(),
                   "environment_indices": np.flatnonzero(eligible).tolist(),
                   "environment_means": valid.tolist()})
    return result
