"""CPU/Gloo regressions for global-batch DDP, shared maps and stateful resume."""

from __future__ import annotations

import copy
import json
import os
import pickle
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusion_policy")
from torch import nn
from torch.utils.data import SequentialSampler

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray, build_decoded_cache
from adavip.manifeel.distributed_full_sensor_workspace import (
    DistributedFullSensorWorkspace, GlobalBatchShardSampler, PolicyLossModule, ValidityDataset,
)
from adavip.manifeel.full_sensor_dataset import FullSensorManiFeelDataset
from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace
from test_full_sensor_stage0 import dataset_arguments, workspace_config


@pytest.mark.parametrize("size", [1, 3, 21, 64, 65, 129])
def test_global_shards_preserve_order_tail_and_cursor(size: int) -> None:
    sampler = SequentialSampler(range(size))
    shards = [list(GlobalBatchShardSampler(sampler, 64, rank, 4)) for rank in range(4)]
    recovered = []
    for batch_index in range(len(shards[0])):
        interleaved = []
        for rank in range(4):
            for offset, index in enumerate(shards[rank][batch_index]):
                if index >= 0:
                    interleaved.append((offset * 4 + rank, index))
        recovered.extend(index for _, index in sorted(interleaved))
    assert recovered == list(range(size))
    for rank in range(4):
        assert list(GlobalBatchShardSampler(sampler, 64, rank, 4, start_batch=1)) == shards[rank][1:]


def test_mmap_pickle_reattaches_without_serializing_image_payload(tmp_path: Path) -> None:
    path = tmp_path / "images.npy"
    values = np.arange(300000, dtype=np.float32).reshape(100, 3000)
    np.save(path, values)
    original = SourceIndexedArray(np.load(path, mmap_mode="r"), np.arange(100), 100)
    payload = pickle.dumps(original)
    assert len(payload) < 4096
    restored = pickle.loads(payload)
    assert isinstance(restored.values, np.memmap)
    assert not restored.values.flags.writeable
    np.testing.assert_array_equal(restored[[0, 99]], values[[0, 99]])


class QuadraticPolicy(nn.Module):
    """Tiny policy with an analytically comparable global mean gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def compute_loss_per_sample(self, batch: dict, condition_ids: torch.Tensor) -> torch.Tensor:
        """Return an independent squared error for each sample."""
        return (self.linear(batch["features"]).flatten() - batch["target"]).square()


def _ddp_worker(rank: int, world_size: int, rendezvous: str, cfg: dict, single_checkpoint: str) -> None:
    import torch.distributed as distributed
    from omegaconf import OmegaConf
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    os.environ["LOCAL_RANK"] = str(rank)
    distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                                  world_size=world_size, timeout=timedelta(minutes=5))
    torch.manual_seed(5)
    model = QuadraticPolicy()
    reference = copy.deepcopy(model)
    wrapper = DistributedDataParallel(PolicyLossModule(model))
    for size in (21, 3, 1):
        features = torch.arange(size * 2).float().reshape(size, 2) / 20
        target = torch.arange(size).float() / 10
        local = torch.arange(size)[rank::world_size]
        valid = torch.ones(len(local), dtype=torch.bool)
        if not len(local):
            local, valid = torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.bool)
        batch = {"features": features[local], "target": target[local], "_distributed_valid": valid}
        losses = wrapper(batch, torch.zeros(len(local), dtype=torch.long))
        DistributedFullSensorWorkspace._backward(SimpleNamespace(world_size=world_size), losses, batch)
        reference.compute_loss_per_sample({"features": features, "target": target}, None).mean().backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    del wrapper, model, reference
    cfg = OmegaConf.create(cfg)
    migration_cfg = copy.deepcopy(cfg)
    migration_cfg.output_dir += "_migration"
    migration_cfg.resume = single_checkpoint
    migration_cfg.runtime.stop_after_steps = 1
    migrated = DistributedFullSensorWorkspace(migration_cfg, migration_cfg.output_dir)
    migrated.run()
    assert migrated.global_step == migrated.ema_step == 2
    assert migrated.rng_state["world_size"] == world_size
    del migrated
    continuous = DistributedFullSensorWorkspace(cfg, cfg.output_dir)
    continuous.run()
    interrupted_cfg = copy.deepcopy(cfg)
    interrupted_cfg.output_dir += "_interrupted"
    interrupted_cfg.runtime.stop_after_steps = 1
    interrupted = DistributedFullSensorWorkspace(interrupted_cfg, interrupted_cfg.output_dir)
    interrupted.run()
    assert interrupted.global_step == 1 and interrupted.batch_in_epoch == 1
    resumed_cfg = copy.deepcopy(interrupted_cfg)
    resumed_cfg.runtime.stop_after_steps = 0
    resumed_cfg.resume = str(Path(resumed_cfg.output_dir) / "checkpoints/latest.ckpt")
    resumed = DistributedFullSensorWorkspace(resumed_cfg, resumed_cfg.output_dir)
    initialize_runtime = resumed._initialize_runtime

    def verify_loaded_state(device: torch.device) -> None:
        for model_name in ("model", "ema_model"):
            for name, value in getattr(interrupted, model_name).state_dict().items():
                torch.testing.assert_close(value, getattr(resumed, model_name).state_dict()[name], rtol=0, atol=0)
        for actual, expected in zip(resumed.optimizer.state.values(), interrupted.optimizer.state.values()):
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        initialize_runtime(device)

    resumed._initialize_runtime = verify_loaded_state
    resumed.run()
    assert resumed.global_step == continuous.global_step == resumed.ema_step
    for model_name in ("model", "ema_model"):
        for name, value in getattr(continuous, model_name).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, model_name).state_dict()[name], rtol=0, atol=0)
    for actual, expected in zip(resumed.optimizer.state.values(), continuous.optimizer.state.values()):
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    if rank == 0:
        dataset = FullSensorManiFeelDataset(**{key: value for key, value in cfg.dataset.items() if key != "_target_"})
        serial = FullSensorWorkspace(cfg, str(Path(cfg.output_dir) / "serial_validation"))
        serial.ema_model = resumed.ema_model
        expected = serial.validate(dataset.get_validation_dataset(), torch.device("cpu"))
        records = [json.loads(line) for line in (Path(resumed_cfg.output_dir) / "metrics.jsonl").read_text().splitlines()]
        final = records[-1]
        for key, value in expected.items():
            assert final[key] == pytest.approx(value, rel=1e-6, abs=1e-7)
        assert final["validation/sample_count"] == len(dataset.get_validation_dataset())
        assert final["validation/task_coverage"] == 2
        assert sum(value for key, value in final.items() if key.startswith("train/count/")) == len(dataset)
        progress = [json.loads(line) for line in (Path(cfg.output_dir) / "progress.jsonl").read_text().splitlines()]
        assert sum(record["event"] == "shared_cache_verified" for record in progress) == 1
    distributed.barrier()
    distributed.destroy_process_group()


def test_four_rank_global_gradient_ema_validation_and_resume(dataset_arguments: dict, tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    cfg = workspace_config(dataset_arguments, tmp_path / "continuous")
    dataset = FullSensorManiFeelDataset(**dataset_arguments)
    cache_dir = tmp_path / "cache"
    build_decoded_cache(dataset, str(cache_dir), workers=2, block_frames=2)
    cfg.dataset.decoded_cache_dir = str(cache_dir)
    cfg.dataset.decoded_cache_mode = "mmap"
    cfg.dataloader.batch_size = cfg.val_dataloader.batch_size = 8
    cfg.training.max_train_steps = cfg.training.max_val_steps = None
    cfg.training.log_every_steps = 1
    single_cfg = copy.deepcopy(cfg)
    single_cfg.output_dir += "_single"
    single_cfg.runtime.stop_after_steps = 1
    single = FullSensorWorkspace(single_cfg, single_cfg.output_dir)
    single.run()
    single_checkpoint = str(Path(single_cfg.output_dir) / "checkpoints/latest.ckpt")
    torch.multiprocessing.spawn(_ddp_worker, args=(4, str(tmp_path / "rendezvous"),
                               OmegaConf.to_container(cfg, resolve=True), single_checkpoint), nprocs=4, join=True)


def test_spawn_worker_reattaches_shared_decoded_maps(dataset_arguments: dict, tmp_path: Path) -> None:
    original = FullSensorManiFeelDataset(**dataset_arguments)
    cache_dir = tmp_path / "spawn_cache"
    build_decoded_cache(original, str(cache_dir), workers=2, block_frames=2)
    dataset = FullSensorManiFeelDataset(**dataset_arguments, decoded_cache_dir=str(cache_dir), decoded_cache_mode="mmap")
    loader = torch.utils.data.DataLoader(ValidityDataset(dataset), batch_size=3, num_workers=1,
                                        multiprocessing_context="spawn", timeout=60)
    batch = next(iter(loader))
    expected = torch.utils.data.default_collate([original[index] for index in range(3)])
    for key in batch["obs"]:
        torch.testing.assert_close(batch["obs"][key], expected["obs"][key], rtol=0, atol=0)
    assert batch["_distributed_valid"].all()


def test_epoch_extension_is_explicit_and_lr_continuous(dataset_arguments: dict, tmp_path: Path) -> None:
    cfg = workspace_config(dataset_arguments, tmp_path / "extension")
    cfg.runtime.stop_after_steps = 1
    first = FullSensorWorkspace(cfg, cfg.output_dir)
    first.run()
    extension = copy.deepcopy(cfg)
    extension.resume = str(Path(cfg.output_dir) / "checkpoints/latest.ckpt")
    extension.training.num_epochs = 1000
    with pytest.raises(ValueError, match="Resume rejected"):
        FullSensorWorkspace(extension, str(tmp_path / "rejected")).run()
    extension.runtime.allow_epoch_extension = True
    resumed = FullSensorWorkspace(extension, extension.output_dir)
    resumed.run()
    assert resumed.global_step == 2
    assert resumed.lr_schedule_extension["step"] == 1
    assert resumed.optimizer.param_groups[0]["lr"] == first.optimizer.param_groups[0]["lr"]
    assert resumed._lr_multiplier(2000) == pytest.approx(1.0, rel=1e-5)


@pytest.mark.parametrize("fault", [None, "speed", "epoch_speed", "memory", "sampler", "ema", "validation", "protocol", "rng"])
def test_preflight_assessment_fails_closed(tmp_path: Path, fault: str | None) -> None:
    from omegaconf import OmegaConf
    from scripts.manifeel.assess_full_sensor_ddp import assess

    run, baseline = tmp_path / "ddp", tmp_path / "single"
    run.mkdir()
    baseline.mkdir()
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs/manifeel/train_full_sensor_reference.yaml")
    cfg.dataloader.batch_size = cfg.val_dataloader.batch_size = 64
    OmegaConf.save(cfg, baseline / "config.yaml")
    if fault == "protocol":
        cfg.optimizer.lr *= 2
    OmegaConf.save(cfg, run / "config.yaml")
    identity = {"model": "raw", "ema_model": "ema", "global_step": 2164, "ema_step": 2164, "batch_in_epoch": 0}
    reports = [{"file_identity": "shared", "host": "node", "nbytes": 1000000000,
                "mapped_bytes": {"Pss": 250000000, "Rss": 1000000000}} for rank in range(4)]
    if fault == "memory":
        reports[0]["mapped_bytes"]["Pss"] *= 4
    plan = {"event": "global_sampler_verified", "epoch": 2, "global_samples": 69205,
            "global_batches": 1082, "tail_samples_per_rank": [6, 5, 5, 5], "sample_order_sha256": "plan"}
    if fault == "sampler":
        plan["global_samples"] += 3
    progress = [{"event": "replicas_verified", "phase": "initialized", **identity},
                {"event": "shared_cache_verified", "ranks": reports}, plan,
                {"event": "throughput_preflight", "passed": True, "samples_per_second": 400},
                {"event": "train_step", "world_size": 4, "per_rank_batch_size": 16, "global_batch_size": 64},
                {"event": "replicas_verified", "phase": "epoch_complete", **identity},
                {"event": "replicas_verified", "phase": "initialized", **identity},
                {"event": "shared_cache_verified", "ranks": reports},
                {"event": "resume_rng_verified", "source_world_size": 4},
                {"event": "throughput_preflight", "passed": True, "samples_per_second": 400},
                {"event": "stopped", "reason": "bounded_step_preflight", "global_step": 2284}]
    if fault == "speed":
        progress[-2]["samples_per_second"] = 180
    if fault == "ema":
        progress[6]["ema_step"] -= 1
    if fault == "rng":
        progress[8]["source_world_size"] = 1
    metrics = {"runtime/resumed_batch_cursor": 0, "runtime/epoch_wall_seconds": 180,
               "validation/task_coverage": 9, "validation/sample_count": 2060, "train/count/full": 69205,
               "epoch": 2, "train/seconds": 180}
    for condition in ("full", "V1", "V2", "V3", "T1", "T2", "P", "L", "vision", "tactile"):
        metrics[f"validation/macro/{condition}"] = 0.1
    if fault == "validation":
        metrics["validation/sample_count"] += 1
    (run / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in progress))
    (run / "metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    baseline_epoch = {"epoch": 2, "train/seconds": 190 if fault == "epoch_speed" else 360,
                      "train/count/full": 69205}
    (baseline / "metrics.jsonl").write_text(json.dumps(baseline_epoch) + "\n")
    (baseline / "progress.jsonl").write_text(json.dumps({"event": "train_step", "samples_per_second": 200}) + "\n")
    (run / "checkpoints").mkdir()
    for tag in ("latest", "interrupt"):
        (run / "checkpoints" / f"{tag}.ckpt").touch()
    result = assess(run, baseline)
    assert result["passed"] == (fault is None)
