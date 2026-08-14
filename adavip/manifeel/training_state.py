"""Helpers for interpreting ManiFeel checkpoint training state."""

from __future__ import annotations

import math


def next_training_state(
    epoch: int,
    global_step: int,
    checkpoint_loaded: bool,
) -> tuple[int, int]:
    """Return the first epoch and step not yet completed by a checkpoint."""
    if not checkpoint_loaded:
        return epoch, global_step
    return epoch + 1, global_step + 1


def remaining_epochs(total_epochs: int, next_epoch: int) -> int:
    """Return epochs left when ``total_epochs`` is the overall run target."""
    return max(0, total_epochs - next_epoch)


def validation_improved(val_loss: float, best_val_loss: float) -> bool:
    """Return whether a finite validation loss is a new minimum."""
    return math.isfinite(val_loss) and val_loss < best_val_loss
