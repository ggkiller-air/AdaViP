"""Contracts for the redesigned ManiFeel Stage 1 Teacher."""

from __future__ import annotations

import numpy as np

from adavip.manifeel.full_sensor_schema import (
    ACTION_CODEC,
    CAMERA_KEYS,
    IMAGE_KEYS,
    TACTILE_KEYS,
    make_shape_meta,
)


SCHEMA_VERSION = "adavip.manifeel.stage1_v2"
OBS_HISTORY_STEPS = 6
ACTION_HORIZON = 16
ACTION_EXECUTION_STEPS = 8
FUTURE_STEPS = 4
ACTION_HISTORY_STEPS = 8
TOKEN_DIM = 256
IMAGE_FEATURE_DIM = 512
LANGUAGE_DIM = 512
SENSOR_TOKEN_NAMES = IMAGE_KEYS + ("proprio", "language")
MASK_NAMES = (
    "full",
    "front",
    "side",
    "wrist",
    "left_tactile",
    "right_tactile",
    "proprio",
    "all_vision",
    "all_tactile",
)


def condition_presence() -> np.ndarray:
    """Return the nine Stage 1 conditions with language always present."""
    presence = np.ones((len(MASK_NAMES), len(SENSOR_TOKEN_NAMES)), dtype=bool)
    for condition_index in range(1, 6):
        presence[condition_index, condition_index - 1] = False
    presence[6, 5] = False
    presence[7, :3] = False
    presence[8, 3:5] = False
    return presence


def condition_probabilities(full_probability: float = 0.8) -> np.ndarray:
    """Return 80% full input and equal probability for eight masks."""
    if not np.isfinite(full_probability) or not 0.0 <= full_probability <= 1.0:
        raise ValueError("full_probability must be in [0, 1]")
    return np.r_[full_probability, np.full(8, (1.0 - full_probability) / 8.0)]


def validate_stage1_shape_meta(shape_meta: dict) -> None:
    """Validate the seven inputs used by the v2 teacher."""
    expected = set(IMAGE_KEYS) | {"state", "task_embedding"}
    if set(shape_meta.get("obs", {})) != expected:
        raise ValueError(f"Stage 1 v2 requires observations {sorted(expected)}")
    if tuple(shape_meta["obs"]["state"]["shape"]) != (7,):
        raise ValueError("state must be the 7D EEF pose [position, quaternion xyzw]")
    if tuple(shape_meta["obs"]["task_embedding"]["shape"]) != (LANGUAGE_DIM,):
        raise ValueError("task_embedding must be the fixed CLIP 512D embedding")
    if tuple(shape_meta.get("action", {}).get("shape", ())) != (7,):
        raise ValueError("Stage 1 v2 uses the padded shared 7D action space")


__all__ = [
    "ACTION_CODEC",
    "ACTION_EXECUTION_STEPS",
    "ACTION_HISTORY_STEPS",
    "ACTION_HORIZON",
    "CAMERA_KEYS",
    "FUTURE_STEPS",
    "IMAGE_FEATURE_DIM",
    "IMAGE_KEYS",
    "LANGUAGE_DIM",
    "MASK_NAMES",
    "OBS_HISTORY_STEPS",
    "SCHEMA_VERSION",
    "SENSOR_TOKEN_NAMES",
    "TACTILE_KEYS",
    "TOKEN_DIM",
    "condition_presence",
    "condition_probabilities",
    "make_shape_meta",
    "validate_stage1_shape_meta",
]
