"""CPU regression coverage for paired action-probe comparison figures."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.manifeel.compare_power_plug_action_probes import compare_probes


def write_probes(root: Path, mismatch: bool = False) -> None:
    """Create small saved probes with optional mismatched spatial inputs."""
    for name in ("baseline", "hyperresnet"):
        action_dir = root / "gpu5_action_probes" / f"{name}_epoch100_demos012"
        spatial_dir = root / "gpu5_action_occlusion" / f"{name}_epoch100_episode1"
        action_dir.mkdir(parents=True)
        spatial_dir.mkdir(parents=True)
        variants = ["repeat", "vision_blur", "tactile_blur"]
        if name == "hyperresnet":
            variants.append("zero_hypernet")
        np.savez(action_dir / "action_sensitivity.npz", episode=np.array([0, 1]),
            episode_step=np.array([2, 3]), observation_indices=np.array([[1, 2], [2, 3]]),
            delayed_observation_indices=np.zeros((2, 2)), seeds=np.array([11]),
            action_scale=np.ones(6), variants=np.asarray(variants),
            rms_all=np.zeros((1, 2, len(variants))))
        inputs = np.zeros((1, 2, 3, 8, 8))
        if mismatch and name == "hyperresnet":
            inputs += 1
        np.savez(spatial_dir / "action_occlusion.npz", inputs=inputs, steps=np.array([3]),
            seeds=np.array([11]), modalities=np.array(["wrist", "tactile"]),
            rms_all=np.full((1, 1, 2, 2, 2), 0.02 if name == "hyperresnet" else 0.01))


def test_paired_action_probe_figures(tmp_path: Path) -> None:
    """Both model maps use the same observed maximum and generate real figures."""
    write_probes(tmp_path)
    output = tmp_path / "report"
    report = compare_probes(tmp_path, output)
    assert report["occlusion_color_max"] == pytest.approx(0.02)
    assert json.loads((output / "summary.json").read_text()) == report
    for name in ("action_sensitivity_comparison.png", "action_occlusion_comparison.png"):
        assert (output / name).stat().st_size > 1000


def test_mismatched_action_probe_inputs_rejected(tmp_path: Path) -> None:
    """Reject different observations rather than presenting them as paired maps."""
    write_probes(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="spatial probes are not paired: inputs"):
        compare_probes(tmp_path, tmp_path / "report")
