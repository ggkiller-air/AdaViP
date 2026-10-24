"""CPU regressions for the isolated 4H200 global-batch128 benchmark."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_balanced_stream_global128_has_no_epoch_tail_drop_and_exact_resume() -> None:
    pytest.importorskip("torch")
    from adavip.manifeel.stage1_benchmark_data import BalancedStreamBatches
    from adavip.manifeel.stage1_v2_training import BalancedTaskSampler

    dataset = SimpleNamespace(task_lengths=[7, 13, 9, 5, 18, 4, 11, 6, 8])
    shards = [list(BalancedStreamBatches(dataset, 42, 32, rank, 4, 12)) for rank in range(4)]
    global_batches = [sum([shards[rank][step] for rank in range(4)], []) for step in range(12)]
    expected = []
    for epoch in range(10):
        expected.extend(BalancedTaskSampler(dataset, 42, epoch))
    assert sum(global_batches, []) == expected[:12 * 128]
    for rank in range(4):
        resumed = list(BalancedStreamBatches(dataset, 42, 32, rank, 4, 5, consumed=7 * 128))
        assert resumed == shards[rank][7:]
    starts = np.cumsum([0] + dataset.task_lengths)
    tasks = np.searchsorted(starts[1:], np.asarray(sum(global_batches, [])), side="right")
    counts = np.bincount(tasks, minlength=9)
    assert counts.sum() == 12 * 128
    assert counts.max() - counts.min() <= 1


def test_nested_state_digest_covers_scalars_optimizer_rng_and_arrays() -> None:
    torch = pytest.importorskip("torch")
    from adavip.manifeel.stage1_ddp_benchmark import state_digest

    value = {"step": torch.tensor(3.0), "model": torch.arange(12).reshape(3, 4),
             "numpy_rng": np.random.default_rng(42).bit_generator.state,
             "array": np.arange(4, dtype=np.uint8), "groups": [{"lr": 1e-4}]}
    assert state_digest(value) == state_digest(value)
    original = state_digest(value)
    value["step"] += 1
    assert state_digest(value) != original


def test_report_filters_utilization_to_timed_window_and_compares_fair_baseline(tmp_path) -> None:
    pytest.importorskip("torch")
    from adavip.manifeel.stage1_ddp_benchmark import build_report, write_json

    for world_size, speed in ((1, 50.0), (4, 160.0)):
        write_json(tmp_path / f"world{world_size}_global{world_size * 32}" / "report.json",
                   {"world_size": world_size, "global_batch": world_size * 32, "steps": 2000,
                    "samples_per_second": speed, "steps_per_second": speed / (world_size * 32),
                    "start_timestamp": 10, "end_timestamp": 20, "passed": True})
    entries = [
        {"timestamp": 5, "gpus": [{"uuid": "gpu", "index": 0, "utilization_percent": 0,
                                    "memory_used_mib": 1}]},
        {"timestamp": 15, "gpus": [{"uuid": "gpu", "index": 0, "utilization_percent": 90,
                                     "memory_used_mib": 40000}]},
        {"timestamp": 25, "gpus": [{"uuid": "gpu", "index": 0, "utilization_percent": 10,
                                     "memory_used_mib": 60000}]},
    ]
    (tmp_path / "gpu_monitor.jsonl").write_text("\n".join(json.dumps(item) for item in entries))
    result = build_report(tmp_path)
    parallel = next(item for item in result["reports"] if item["world_size"] == 4)
    assert parallel["speedup_vs_matched_single_gpu"] == pytest.approx(3.2)
    assert parallel["speedup_vs_production_39_7"] == pytest.approx(160 / 39.7)
    assert parallel["gpu_telemetry"][0]["mean_utilization_percent"] == 90
    assert parallel["gpu_telemetry"][0]["peak_memory_used_mib"] == 40000
    assert (tmp_path / "report.md").exists()


def _ddp4_correctness_worker(rank: int, rendezvous: str) -> None:
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel
    from adavip.manifeel.stage1_ddp_benchmark import assert_replicated

    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        torch.manual_seed(42)
        serial = torch.nn.Linear(3, 2)
        parallel = DistributedDataParallel(torch.nn.Linear(3, 2))
        parallel.module.load_state_dict(serial.state_dict())
        inputs = torch.arange(128 * 3, dtype=torch.float32).reshape(128, 3) / 384
        serial(inputs).square().mean().backward()
        parallel(inputs[rank * 32:(rank + 1) * 32]).square().mean().backward()
        for expected, actual in zip(serial.parameters(), parallel.module.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
        assert assert_replicated([parameter.grad for parameter in parallel.module.parameters()],
                                 4, "CPU four-rank gradients") == 0
    finally:
        distributed.destroy_process_group()


def test_ddp4_global128_gradients_equal_single_global_mean(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    torch.multiprocessing.spawn(_ddp4_correctness_worker,
                                args=((tmp_path / "gloo_rendezvous").as_uri(),), nprocs=4)


@pytest.mark.parametrize("streaming", [False, True])
def test_sequential_ram_loader_preserves_windows_and_rejects_corruption(tmp_path, streaming) -> None:
    torch = pytest.importorskip("torch")
    zarr = pytest.importorskip("zarr")
    from adavip.manifeel.decoded_sensor_cache import build_decoded_cache
    from adavip.manifeel.stage1_benchmark_data import BenchmarkDataset
    from adavip.manifeel.stage1_v2_dataset import Stage1V2ManiFeelDataset
    from adavip.manifeel.stage1_v2_schema import IMAGE_KEYS, make_shape_meta

    store = zarr.open_group(str(tmp_path / "task"), mode="w")
    store.create_group("meta").create_dataset("episode_ends", data=np.asarray([30, 60]))
    arrays = store.create_group("data")
    for key in IMAGE_KEYS:
        arrays.create_dataset(key, data=np.random.default_rng(42).random((60, 4, 4, 3), dtype=np.float32))
    arrays.create_dataset("state", data=np.zeros((60, 7), dtype=np.float32))
    arrays.create_dataset("action", data=np.arange(60 * 7, dtype=np.float32).reshape(60, 7))
    embedding = tmp_path / "embedding.npz"
    np.savez(embedding, task_ids=np.asarray(["task"]), embeddings=np.zeros((1, 512), dtype=np.float32))
    options = {"shape_meta": make_shape_meta((3, 4, 4), (3, 4, 4)),
               "dataset_root": str(tmp_path), "task_embedding_path": str(embedding),
               "task_specs": [{"task_id": "task", "text": "task", "dataset": "task",
                               "isaacgym_cfg_name": "unused.yaml", "action_dim": 7}]}
    original = Stage1V2ManiFeelDataset(**options)
    cache = tmp_path / "cache"
    build_decoded_cache(original, str(cache), workers=1)
    dataset_class = BenchmarkDataset
    extra = {}
    if streaming:
        from adavip.manifeel.stage1_streaming_cache import StreamingBenchmarkDataset, build_block_index

        block_index = tmp_path / "blocks.json"
        build_block_index(cache, block_index, block_bytes=1024)
        dataset_class = StreamingBenchmarkDataset
        extra["block_index_path"] = str(block_index)
    prepared = dataset_class(**options, **extra, decoded_cache_dir=str(cache), decoded_cache_mode="ram")
    if streaming:
        assert not prepared.cache_status()["complete"]
    from adavip.manifeel.full_sensor_workspace import normalizer_fingerprint

    assert normalizer_fingerprint(original.get_normalizer()) == normalizer_fingerprint(prepared.get_normalizer())
    assert original.manifest == {key: value for key, value in prepared.manifest.items()
                                 if key != "decoded_cache_contract"}
    for index in range(len(original)):
        expected, actual = original[index], prepared[index]
        for key in expected["obs"]:
            torch.testing.assert_close(actual["obs"][key], expected["obs"][key], rtol=0, atol=0)
        for key in IMAGE_KEYS:
            torch.testing.assert_close(actual["future_obs"][key], expected["future_obs"][key], rtol=0, atol=0)
        assert int(actual["dataset_index"]) == index
    if streaming:
        prepared.warm_cache(workers=2)
        assert prepared.cache_status()["complete"]
    values = np.load(cache / "task0_front.npy", mmap_mode="r+")
    values[0, 0, 0, 0] = 0.123
    values.flush()
    with pytest.raises(ValueError, match="checksum|identity"):
        dataset_class(**options, **extra, decoded_cache_dir=str(cache), decoded_cache_mode="ram")


def test_checkpoint_probe_replays_optimizer_scheduler_rng_and_protects_file(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf
    from adavip.manifeel import stage1_ddp_benchmark as benchmark

    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(3, 2)
    model.normalizer = torch.nn.Linear(2, 2, bias=False).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    inputs = torch.randn(8, 3)

    def update(batch, audit_ema=False):
        optimizer.zero_grad(set_to_none=True)
        loss = (model.backbone(batch) - torch.randn(8, 2)).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        return {"loss": loss.detach()}, norm

    monkeypatch.setattr(benchmark, "local_rng", lambda device: {"torch": torch.get_rng_state()})
    monkeypatch.setattr(benchmark, "restore_local_rng",
                        lambda state, device: torch.set_rng_state(state["torch"]))
    update(inputs)
    result = benchmark.checkpoint_probe(model, optimizer, scheduler, OmegaConf.create({}), inputs,
                                         update, 0, 1, torch.device("cpu"), tmp_path,
                                         128, 1, {"test": True})
    assert result["passed"]
    assert result["sampler_consumed"] == 128
    with pytest.raises(FileExistsError):
        benchmark.checkpoint_probe(model, optimizer, scheduler, OmegaConf.create({}), inputs,
                                   update, 0, 1, torch.device("cpu"), tmp_path,
                                   128, 1, {"test": True})
