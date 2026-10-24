"""CPU regressions for the isolated single-H200 profiling workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def profile_module():
    """Load diagnostics without launching any GPU work."""
    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/profile_stage1_v2.py"
    spec = importlib.util.spec_from_file_location("profile_stage1_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_rows_include_six_observations_four_futures_and_actions(profile_module):
    dataset = SimpleNamespace(task_specs=["task"], _resolve_index=lambda index: (0, 0, 100, 150, index))
    images, actions = profile_module.required_rows(dataset, [0, 3])
    np.testing.assert_array_equal(images[0], np.arange(103, 116))
    np.testing.assert_array_equal(actions[0], np.arange(100, 126))


def test_fixture_rows_reject_episode_boundary(profile_module):
    dataset = SimpleNamespace(task_specs=["task"], _resolve_index=lambda index: (0, 0, 0, 22, 0))
    with pytest.raises(ValueError, match="episode boundary"):
        profile_module.required_rows(dataset, [0])


def test_profile_fast_read_matches_original_and_rejects_missing_rows(profile_module):
    from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
    from adavip.manifeel.full_sensor_dataset import FullSensorManiFeelDataset

    values = np.arange(8 * 3 * 4 * 3, dtype=np.float32).reshape(8, 3, 4, 3) / 300
    array = SourceIndexedArray(values, np.arange(2, 10), 12)
    dataset = SimpleNamespace(shape_meta={"obs": {"front": {"shape": [3, 3, 4]}}})
    for rows in (np.array([3, 4, 5]), np.array([3, 5, 7])):
        expected = FullSensorManiFeelDataset._read_images(dataset, array, rows, "front")
        np.testing.assert_array_equal(profile_module.optimized_read(array, rows), expected)
    with pytest.raises(IndexError):
        profile_module.optimized_read(array, np.array([0]))


def test_profile_counts_tensor_bytes_and_fixed_batches(profile_module):
    batch = {"obs": {"image": torch.zeros(2, 3, dtype=torch.float32)}, "valid": torch.ones(2, dtype=torch.bool)}
    assert profile_module.batch_nbytes(batch) == 26
    batches = iter(profile_module.RepeatedFixtureBatches([2, 5, 8]))
    first = next(batches)
    first[0] = 99
    assert next(batches) == [2, 5, 8]


def test_profile_digest_and_atomic_json(profile_module, tmp_path):
    values = np.arange(8, dtype=np.float32)
    original = profile_module.digest_array(values)
    values[-1] += 1
    assert profile_module.digest_array(values) != original
    destination = tmp_path / "out/report.json"
    profile_module.write_json(destination, {"success": True})
    assert destination.exists()
    assert not destination.with_suffix(".partial").exists()


def test_profile_gradient_comparison_covers_all_elements(profile_module):
    parameter = torch.nn.Parameter(torch.zeros(3))
    reference = [torch.tensor([1.0, 2.0, 3.0])]
    parameter.grad = reference[0] * 2
    result = profile_module.gradient_comparison(reference, [parameter])
    assert result["cosine"] == pytest.approx(1.0)
    assert result["norm_ratio"] == pytest.approx(2.0)
    with pytest.raises(AssertionError, match="connectivity"):
        profile_module.gradient_comparison([None], [parameter])
