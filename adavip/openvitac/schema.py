"""Explicit six-input contract and derived-action provenance for OpenViTac."""
from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "uvip.openvitac.v1"
HF_REVISION = "5fb2757b88826917242adefd9196424ef8cf2d98"
CAMERA_KEYS = ("head", "wrist")
TACTILE_KEYS = ("tactile_left", "tactile_right")
IMAGE_KEYS = CAMERA_KEYS + TACTILE_KEYS
SENSOR_KEYS = IMAGE_KEYS + ("state", "task_embedding")
CONDITION_NAMES = ("full", "V1", "V2", "T1", "T2", "P", "L", "vision", "tactile")
UTILITY_DIM = len(CONDITION_NAMES) - 1
ACTION_DIM = 8
STATE_DIM = 8
ACTION_CODEC = {
    "name": "derived_next_observed_abs_joint_left_finger_v1",
    "source_has_action": False,
    "target_offset_frames": 1,
    "arm": "next observed q0..q6, radians; proxy target, not recorded commands",
    "gripper": "next observed LEFT finger position, metres; NOT gripper_open_fraction",
    "observation_state": "measured q0..q6 (rad), gripper_open_fraction in [0,1]",
    "nominal_hz": 60,
    "physical_simulation_hz": 120,
    "valid_transition": "within episode and source_step_delta == 2",
    "rollout_status": "requires independent controller/timing/contact validation",
}


def make_shape_meta(camera_shape: tuple = (3, 256, 256), tactile_shape: tuple = (3, 240, 320), language_dim: int = 512) -> dict:
    """Use H,W order; tactile images must not inherit ManiFeel's transposed shape."""
    obs = {key: {"shape": list(camera_shape), "type": "rgb"} for key in CAMERA_KEYS}
    obs.update({key: {"shape": list(tactile_shape), "type": "rgb"} for key in TACTILE_KEYS})
    obs.update(state={"shape": [STATE_DIM], "type": "low_dim"},
               task_embedding={"shape": [language_dim], "type": "low_dim"})
    return {"obs": obs, "action": {"shape": [ACTION_DIM]}}


def validate_shape_meta(meta: dict) -> None:
    """Reject silent sensor, action or proprioception substitutions."""
    if set(meta["obs"]) != set(SENSOR_KEYS):
        raise ValueError("OpenViTac requires exactly 2V + 2T + P + L")
    for key in IMAGE_KEYS:
        shape = meta["obs"][key]["shape"]
        if len(shape) != 3 or shape[0] != 3 or min(shape) <= 0 or meta["obs"][key]["type"] != "rgb":
            raise ValueError("Expected positive RGB [3,H,W]")
    if meta["obs"]["state"]["shape"] != [STATE_DIM] or meta["action"]["shape"] != [ACTION_DIM]:
        raise ValueError("Expected measured 8D proprio and derived 8D targets")
    if len(meta["obs"]["task_embedding"]["shape"]) != 1 or meta["obs"]["task_embedding"]["shape"][0] <= 0:
        raise ValueError("Invalid language dimension")


def condition_presence() -> np.ndarray:
    """Rows: full, six single removals, all vision, all tactile; columns V,V,T,T,P,L."""
    result = np.ones((len(CONDITION_NAMES), len(SENSOR_KEYS)), dtype=bool)
    for index in range(len(SENSOR_KEYS)):
        result[index + 1, index] = False
    result[7, :2] = False
    result[8, 2:4] = False
    return result


def condition_probabilities(full_probability: float = 0.8) -> np.ndarray:
    """Share non-full probability across the eight declared interventions."""
    if not np.isfinite(full_probability) or not 0 <= full_probability <= 1:
        raise ValueError("Invalid full-input probability")
    return np.r_[full_probability, np.full(UTILITY_DIM, (1 - full_probability) / UTILITY_DIM)]


def continuous_segments(length: int, gaps: list) -> list:
    """Partition observed rows at every non-nominal source transition."""
    boundaries = [0] + [int(g["after_frame_index"]) + 1 for g in gaps] + [length]
    if boundaries != sorted(set(boundaries)) or boundaries[0] != 0 or boundaries[-1] != length:
        raise ValueError("Invalid source step gap indices")
    return list(zip(boundaries[:-1], boundaries[1:]))


def complete_anchors(length: int, gaps: list, horizon: int = 16, n_obs_steps: int = 2) -> list:
    """Require the additional next-state target row for every action in the chunk."""
    return [(t, start, stop) for start, stop in continuous_segments(length, gaps)
            for t in range(start + n_obs_steps - 1, stop - horizon + n_obs_steps - 1)]
