"""Versioned sensor and intervention contracts for the Stage-0 reference DP."""

from __future__ import annotations

import numpy as np


SCHEMA_VERSION = "adavip.full_sensor.v1"
CAMERA_KEYS = ("front", "side", "wrist")
TACTILE_KEYS = ("left_tactile_camera_taxim", "right_tactile_camera_taxim")
IMAGE_KEYS = CAMERA_KEYS + TACTILE_KEYS
SENSOR_KEYS = IMAGE_KEYS + ("state", "task_embedding")
CONDITION_NAMES = ("full", "V1", "V2", "V3", "T1", "T2", "P", "L", "vision", "tactile")
ACTION_CODEC = {
    "position": "world_delta = command[0:3] * 0.01 metres",
    "rotation": "left-multiplied axis-angle delta = command[3:6] * 0.05 radians",
    "gripper": "command[6] is the finger joint position target in metres; absent for 6D tasks",
    "state": "world end-effector position followed by quaternion xyzw, not joint angles",
}


def make_shape_meta(
    camera_shape: tuple[int, int, int] = (3, 256, 256),
    tactile_shape: tuple[int, int, int] = (3, 320, 240),
    language_dim: int = 512,
) -> dict:
    """Build the canonical three-camera, two-TacRGB observation schema."""
    observations = {
        key: {"shape": list(camera_shape), "type": "rgb"} for key in CAMERA_KEYS
    }
    observations.update(
        {key: {"shape": list(tactile_shape), "type": "rgb"} for key in TACTILE_KEYS}
    )
    observations["state"] = {"shape": [7], "type": "low_dim"}
    observations["task_embedding"] = {"shape": [language_dim], "type": "low_dim"}
    return {"obs": observations, "action": {"shape": [7]}}


def validate_shape_meta(shape_meta: dict) -> None:
    """Reject reordered semantics, absent sensors and unsupported action spaces."""
    if set(shape_meta["obs"]) != set(SENSOR_KEYS):
        raise ValueError(f"Full-sensor observations must be exactly {SENSOR_KEYS}")
    for key in IMAGE_KEYS:
        entry = shape_meta["obs"][key]
        shape = tuple(entry["shape"])
        if entry.get("type") != "rgb" or len(shape) != 3 or shape[0] != 3 or min(shape) <= 0:
            raise ValueError(f"{key} must be RGB with positive [3,H,W] shape")
    if tuple(shape_meta["obs"]["state"]["shape"]) != (7,):
        raise ValueError("state must contain position plus quaternion, shape [7]")
    for key in ("state", "task_embedding"):
        if shape_meta["obs"][key].get("type") != "low_dim":
            raise ValueError(f"{key} must have type low_dim")
    language_shape = tuple(shape_meta["obs"]["task_embedding"]["shape"])
    if len(language_shape) != 1 or language_shape[0] <= 0:
        raise ValueError("task_embedding must have a positive one-dimensional shape")
    if tuple(shape_meta["action"]["shape"]) != (7,):
        raise ValueError("The shared action space must have shape [7]")


def condition_presence() -> np.ndarray:
    """Return [10,7] presence flags: full, seven removals, vision/tactile removals."""
    presence = np.ones((10, 7), dtype=bool)
    for sensor_index in range(7):
        presence[sensor_index + 1, sensor_index] = False
    presence[8, :3] = False
    presence[9, 3:5] = False
    return presence


def condition_probabilities(
    full_input_probability: float = 0.8,
    mask_weights: list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    """Distribute non-full probability across nine configurable removal weights."""
    if not np.isfinite(full_input_probability) or not 0 <= full_input_probability <= 1:
        raise ValueError("full_input_probability must be in [0,1]")
    weights = np.ones(9, dtype=np.float64) if mask_weights is None else np.asarray(mask_weights, dtype=np.float64)
    if weights.shape != (9,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("mask_weights must contain nine finite nonnegative weights")
    if weights.sum() <= 0:
        raise ValueError("At least one mask weight must be positive")
    return np.r_[full_input_probability, (1 - full_input_probability) * weights / weights.sum()]
