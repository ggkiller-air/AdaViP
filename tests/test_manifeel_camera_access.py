"""Regression tests for the TacSL camera tensor access lifecycle."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("isaacgym", exc_type=ImportError)
torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACGYM_ENVS_ROOT = REPO_ROOT / "third_party" / "manifeel-isaacgymenvs"
if str(ISAACGYM_ENVS_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAACGYM_ENVS_ROOT))

from isaacgymenvs.tacsl_sensors.tacsl_sensors import CameraSensor


class _CameraSpec(dict):
    """Provide the mapping and attribute access used by the TacSL sensor."""

    def __init__(self, image_type: str) -> None:
        super().__init__(image_type=image_type)
        self.image_size = (2, 3)


class _FakeGym:
    """Record the graphics access calls made by a camera sensor."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def step_graphics(self, sim) -> None:
        self.calls.append("step")

    def render_all_camera_sensors(self, sim) -> None:
        self.calls.append("render")

    def start_access_image_tensors(self, sim) -> None:
        self.calls.append("start")

    def end_access_image_tensors(self, sim) -> None:
        self.calls.append("end")


def _make_sensor(image_type: str = "rgb") -> tuple[CameraSensor, _FakeGym]:
    sensor = CameraSensor()
    gym = _FakeGym()
    sensor.gym = gym
    sensor.sim = object()
    sensor.device = "cpu"
    sensor.num_envs = 1
    sensor.camera_spec_dict = {"front": _CameraSpec(image_type)}
    sensor.camera_tensors_list = [
        {"front": torch.zeros((2, 3, 4), dtype=torch.uint8)}
    ]
    return sensor, gym


def test_camera_tensor_access_is_closed_after_copy() -> None:
    sensor, gym = _make_sensor()

    images = sensor.get_camera_image_tensors_dict()

    assert images["front"].shape == (1, 2, 3, 3)
    assert gym.calls == ["step", "render", "start", "end"]


def test_camera_tensor_access_is_closed_after_error() -> None:
    sensor, gym = _make_sensor(image_type="unsupported")

    with pytest.raises(NotImplementedError):
        sensor.get_camera_image_tensors_dict()

    assert gym.calls == ["step", "render", "start", "end"]
