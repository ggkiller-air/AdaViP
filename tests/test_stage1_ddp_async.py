"""Experiment-only provenance and summary checks for four-H200 input tuning."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def benchmark():
    pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/benchmark_stage1_ddp_async.py"
    specification = importlib.util.spec_from_file_location("stage1_async_benchmark_test", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_source_scope_records_dataset_change_but_rejects_policy_change(tmp_path, benchmark):
    source = tmp_path / "source"
    production = tmp_path / "production"
    current = source / "adavip/manifeel"
    original = production / "source/adavip/manifeel"
    current.mkdir(parents=True)
    original.mkdir(parents=True)
    names = ["stage1_v2_policy.py", "stage1_v2_dataset.py", "full_sensor_dataset.py", "full_sensor_schema.py",
             "decoded_sensor_cache.py", "task_protocol.py", "task_embeddings.py", "action_adapter.py"]
    for name in names:
        (current / name).write_text("original")
        (original / name).write_text("original")
    (current / "stage1_v2_dataset.py").write_text("lossless-experiment")
    scope = benchmark.verify_source_scope(source, production)
    assert not scope["production_resume"]
    assert not scope["files"]["stage1_v2_dataset.py"]["identical"]
    (current / "stage1_v2_policy.py").write_text("changed-model")
    with pytest.raises(ValueError, match="Non-dataset"):
        benchmark.verify_source_scope(source, production)


def test_summary_compares_full_data_pipelines_and_rejects_numerical_change(tmp_path, benchmark):
    for pipeline, world_size, rate in [("copy_stream", 4, 140.0), ("pinned", 4, 120.0),
                                       ("copy_stream", 1, 40.0)]:
        path = tmp_path / pipeline / f"world{world_size}_global{world_size * 32}" / "report.json"
        path.parent.mkdir(parents=True)
        report = {"pipeline": pipeline, "world_size": world_size, "global_batch": world_size * 32,
                  "steps": 2000, "samples_per_second": rate,
                  "ranks": [{"cpu_threads": 4, "first_three_warmup_metrics": [{"action_loss": 0.1}]}]}
        path.write_text(json.dumps(report))
    result = benchmark.summarize_experiment(tmp_path)
    assert result["four_gpu_initial_metric_max_abs_difference"] == 0
    asynchronous = next(case for case in result["cases"]
                        if case["world_size"] == 4 and case["pipeline"] == "copy_stream")
    assert asynchronous["ratio_vs_matched_copy_stream_single"] == 3.5
    path = tmp_path / "pinned/world4_global128/report.json"
    report = json.loads(path.read_text())
    report["ranks"][0]["first_three_warmup_metrics"][0]["action_loss"] = 0.2
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError, match="losses/gradient"):
        benchmark.summarize_experiment(tmp_path)


def test_normalizer_fit_uses_reference_threads_and_restores_serial_parent(benchmark):
    torch = pytest.importorskip("torch")
    from adavip.manifeel.full_sensor_workspace import normalizer_fingerprint

    normalizer = torch.nn.Linear(2, 2)
    expected_hash = normalizer_fingerprint(normalizer)
    observed = []

    class Dataset:
        def get_normalizer(self, mode):
            observed.append((mode, torch.get_num_threads()))
            return normalizer

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = benchmark.fit_reference_normalizer(Dataset(), "limits", expected_hash)
        assert result is normalizer
        assert observed == [("limits", 4)]
        assert torch.get_num_threads() == 1
        with pytest.raises(AssertionError, match="fingerprint mismatch"):
            benchmark.fit_reference_normalizer(Dataset(), "limits", "wrong-hash")
        assert torch.get_num_threads() == 1

        class BrokenDataset:
            def get_normalizer(self, mode):
                raise RuntimeError("source corruption")

        with pytest.raises(RuntimeError, match="source corruption"):
            benchmark.fit_reference_normalizer(BrokenDataset(), "limits", expected_hash)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)
