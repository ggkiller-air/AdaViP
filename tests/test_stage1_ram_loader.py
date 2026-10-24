"""Bitwise and sampling regressions for opt-in Stage 1 RAM batching."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def cached_dataset():
    """Provide nine compact tasks with distinct actions and two episodes each."""
    from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
    from adavip.manifeel.full_sensor_schema import IMAGE_KEYS
    from adavip.manifeel.stage1_parallel_data import CachedStage1Dataset
    from adavip.manifeel.stage1_v2_schema import make_shape_meta

    dataset = CachedStage1Dataset.__new__(CachedStage1Dataset)
    dataset.shape_meta = make_shape_meta(camera_shape=(3, 4, 5), tactile_shape=(3, 3, 4))
    dataset.task_specs = [SimpleNamespace(task_id=f"task{index}", action_dim=6 + index % 2)
                          for index in range(9)]
    dataset.task_embeddings = {spec.task_id: np.full(512, index, dtype=np.float32)
                               for index, spec in enumerate(dataset.task_specs)}
    dataset.episode_ends = [np.array([30, 60]) for _ in dataset.task_specs]
    dataset.train_episodes = [np.array([0, 1]) for _ in dataset.task_specs]
    dataset.split = "train"
    dataset.groups = []
    generator = np.random.default_rng(42)
    for spec in dataset.task_specs:
        arrays = {}
        for key in IMAGE_KEYS:
            channels, height, width = dataset.shape_meta["obs"][key]["shape"]
            values = generator.random((60, height, width, channels), dtype=np.float32)
            arrays[key] = SourceIndexedArray(values, np.arange(60), 60)
        for key, width in (("state", 7), ("action", spec.action_dim)):
            values = generator.standard_normal((60, width)).astype(np.float32)
            arrays[key] = SourceIndexedArray(values, np.arange(60), 60)
        dataset.groups.append({"data": arrays})
    dataset._build_index()
    return dataset


def assert_same_batch(actual, expected):
    """Compare every tensor, including all identities, masks and padded actions."""
    assert actual.keys() == expected.keys()
    for key in expected:
        if isinstance(expected[key], dict):
            assert_same_batch(actual[key], expected[key])
        else:
            assert actual[key].dtype == expected[key].dtype
            assert torch.equal(actual[key], expected[key]), key
            assert actual[key].is_contiguous()


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("epoch,offset", [(0, 0), (3, 17)])
def test_ram_batches_match_native_loader_bitwise(cached_dataset, workers, epoch, offset):
    from torch.utils.data import DataLoader
    from adavip.manifeel.stage1_ram_loader import make_ram_dataloader
    from adavip.manifeel.stage1_v2_dataset import Stage1V2ManiFeelDataset
    from adavip.manifeel.stage1_v2_training import BalancedTaskSampler

    reference = copy.copy(cached_dataset)
    reference.__class__ = Stage1V2ManiFeelDataset
    options = dict(batch_size=32, num_workers=workers, persistent_workers=workers > 0)
    expected = DataLoader(reference, sampler=BalancedTaskSampler(reference, 42, epoch, offset),
                          generator=torch.Generator().manual_seed(42 + epoch), **options)
    actual = make_ram_dataloader(cached_dataset,
                                sampler=BalancedTaskSampler(cached_dataset, 42, epoch, offset),
                                generator=torch.Generator().manual_seed(42 + epoch), **options)
    expected_count = 0
    actual_count = 0
    for actual_batch, expected_batch in zip(actual, expected):
        assert_same_batch(actual_batch, expected_batch)
        actual_count += len(actual_batch["task_index"])
        expected_count += len(expected_batch["task_index"])
        if workers:
            assert actual_batch["obs"]["front"].is_shared()
    assert actual_count == expected_count == len(BalancedTaskSampler(reference, 42, epoch, offset))


def test_repeated_reordered_and_episode_boundary_rows(cached_dataset):
    from torch.utils.data import default_collate
    from adavip.manifeel.stage1_ram_loader import RamBatchDataset

    indices = [0, 7, 8, 15, 16, 143, 0, -1]
    original = cached_dataset.groups[0]["data"]["front"].values.copy()
    expected = default_collate([cached_dataset[index] for index in indices])
    actual = RamBatchDataset(cached_dataset).__getitems__(indices)
    assert_same_batch(actual, expected)
    actual["obs"]["front"].zero_()
    np.testing.assert_array_equal(cached_dataset.groups[0]["data"]["front"].values, original)
    assert not cached_dataset.groups[0]["data"]["front"].values.flags.writeable


def test_noncache_fallback_and_validation(cached_dataset):
    from torch.utils.data import default_collate
    from adavip.manifeel.stage1_ram_loader import RamBatchDataset, make_ram_dataloader

    cached_dataset.groups[0]["data"]["front"] = cached_dataset.groups[0]["data"]["front"].values
    assert_same_batch(RamBatchDataset(cached_dataset).__getitems__([0, 1]),
                      default_collate([cached_dataset[0], cached_dataset[1]]))
    with pytest.raises(ValueError, match="empty batch"):
        RamBatchDataset(cached_dataset).__getitems__([])
    with pytest.raises(ValueError, match="collate_fn"):
        make_ram_dataloader(cached_dataset, collate_fn=lambda batch: batch)
    with pytest.raises(ValueError, match="automatic batching"):
        make_ram_dataloader(cached_dataset, batch_size=None)


@pytest.mark.parametrize("rows", [[2, 3, 4], [2, 4, 6], [4, 4, 2]])
def test_direct_cache_copy_with_compact_and_repeated_rows(cached_dataset, rows):
    from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray

    values = cached_dataset.groups[0]["data"]["front"].values[:4].copy()
    array = SourceIndexedArray(values, np.array([2, 3, 4, 6]), 10)
    indices = np.array(rows)
    expected = cached_dataset._read_images(array, indices, "front")
    destination = torch.empty(expected.shape)
    assert cached_dataset._image_tensor(array, indices, "front", destination) is destination
    np.testing.assert_array_equal(destination.numpy(), expected)
    with pytest.raises(IndexError, match="requested frame"):
        cached_dataset._image_tensor(array, np.array([0]), "front", destination[:1])


def test_direct_copy_rejects_wrong_cache_shape(cached_dataset):
    from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray

    array = SourceIndexedArray(np.zeros((3, 4, 5, 1), dtype=np.float32), np.arange(3), 3)
    with pytest.raises(ValueError, match="shape/dtype"):
        cached_dataset._image_tensor(array, np.arange(3), "front", torch.empty(3, 3, 4, 5))


@pytest.fixture
def benchmark_module():
    """Import the independent benchmark without allocating a GPU."""
    import importlib.util
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts/manifeel"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("benchmark_stage1_ram_loader", scripts / "benchmark_stage1_ram_loader.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(scripts))


def test_benchmark_replays_sampler_prefix_without_mutating_it(benchmark_module):
    indices = list(range(64))
    batches = iter(benchmark_module.FixtureBatches(indices))
    assert next(batches) == indices[:32]
    assert next(batches) == indices[32:]
    first = next(batches)
    first[0] = -1
    assert indices == list(range(64))
    with pytest.raises(ValueError, match="batch32"):
        benchmark_module.FixtureBatches(list(range(33)))
    with pytest.raises(AssertionError, match="keys/order"):
        benchmark_module.compare_tensors({"changed": torch.ones(1)}, {"original": torch.ones(1)})
    with pytest.raises(AssertionError, match="differs"):
        benchmark_module.compare_tensors({"value": torch.ones(1)}, {"value": torch.zeros(1)})


def test_benchmark_summary_uses_only_stable_gpu_samples(benchmark_module, tmp_path):
    import datetime
    import json

    stamp = datetime.datetime(2026, 10, 24, 12, 0, 0).timestamp()
    telemetry = "timestamp, utilization.gpu [%], memory.used [MiB]\n"
    telemetry += "2026/10/24 12:00:00.000, 80 %, 41000 MiB\n"
    telemetry += "2026/10/24 12:00:10.000, 0 %, 0 MiB\n"
    (tmp_path / "gpu_1.csv").write_text(telemetry)
    for label, rate in (("original_a", 40), ("cached", 60), ("batched", 80), ("original_b", 40)):
        directory = tmp_path / label
        directory.mkdir()
        report = {"started_timestamp": stamp - 1, "finished_timestamp": stamp + 1,
                  "case": "original" if label.startswith("original") else label,
                  "samples_per_second": rate, "first_losses": [{"action_loss": 0.01}],
                  "last_losses": {"action_loss": 0.005}}
        (directory / "report.json").write_text(json.dumps(report))
    result = benchmark_module.summarize_results(tmp_path)
    assert result["cases"]["batched"]["speedup_vs_mean_original"] == 2.0
    assert result["cases"]["batched"]["gpu_util_percent"]["mean"] == 80
    assert result["cases"]["batched"]["telemetry_samples"] == 1
    assert result["baseline_drift_ratio"] == 1.0
    assert max(result["initial_loss_and_grad_max_abs_difference"].values()) == 0
    assert max(result["final_loss_and_grad_max_abs_difference"].values()) == 0
    assert result["speedup_interpretable"]
    report["samples_per_second"] = 80
    (directory / "report.json").write_text(json.dumps(report))
    result = benchmark_module.summarize_results(tmp_path)
    assert not result["speedup_interpretable"]
    assert result["cases"]["batched"]["speedup_vs_mean_original"] is None
    assert result["cases"]["batched"]["ratio_vs_fastest_original"] == 1.0
