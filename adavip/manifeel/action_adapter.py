"""Action adapters for the ManiFeel multi-task protocol."""

from __future__ import annotations

from typing import Any

import numpy as np


MULTITASK_ACTION_DIM = 7


def action_loss_mask(action_dim: int, horizon: int, target_dim: int = MULTITASK_ACTION_DIM) -> np.ndarray:
    """Return a horizon x target_dim loss mask for a task action dimension."""
    if action_dim <= 0 or action_dim > target_dim:
        raise ValueError(f"action_dim must be in [1, {target_dim}], got {action_dim}")
    mask = np.zeros((horizon, target_dim), dtype=np.float32)
    mask[:, :action_dim] = 1.0
    return mask


def pad_action(action: np.ndarray, target_dim: int = MULTITASK_ACTION_DIM) -> np.ndarray:
    """Pad a T x D action array to the unified multi-task action dimension."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim < 1:
        raise ValueError(f"action must have at least one dimension, got {action.shape}")
    action_dim = action.shape[-1]
    if action_dim > target_dim:
        raise ValueError(f"action dim {action_dim} exceeds target dim {target_dim}")
    if action_dim == target_dim:
        return action.astype(np.float32, copy=False)
    padded = np.zeros(action.shape[:-1] + (target_dim,), dtype=np.float32)
    padded[..., :action_dim] = action
    return padded


def truncate_action_for_task(action: Any, action_dim: int) -> Any:
    """Truncate a unified action tensor/array to the simulator action dimension."""
    if action_dim <= 0 or action_dim > MULTITASK_ACTION_DIM:
        raise ValueError(f"action_dim must be in [1, {MULTITASK_ACTION_DIM}], got {action_dim}")
    return action[..., :action_dim]
