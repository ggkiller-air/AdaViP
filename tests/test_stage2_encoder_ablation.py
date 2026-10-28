"""Encoder initialization, phase timing, gradient isolation and exact S1--S4 resume."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, WeightedRandomSampler

from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
from adavip.manifeel.stage2_ablation_workspace import (
    Stage2AblationTrainer, Stage2EpochSampler, immutable_fingerprint,
    tensor_fingerprint, validate_gradient_routes,
)
from scripts.manifeel.train_stage2_t2 import make_optimizer_scheduler, set_stage_trainability
from test_stage2_evaluation import small_config
from test_stage2_policy import make_batch, make_policy
from test_full_sensor_stage0 import dataset_arguments


@pytest.fixture
def teacher() -> object:
    torch.manual_seed(2026)
    return make_policy()


def student(teacher: object, strategy: str) -> object:
    """Use a common seed and inherit a distinct, deterministic Teacher."""
    torch.manual_seed(42)
    policy = make_policy(strategy)
    policy.initialize_from_teacher(teacher)
    return policy


def test_common_initialization_independence_and_randomization(teacher: object) -> None:
    common, final_rng = [], []
    for strategy in ("s1", "s2", "s3", "s4"):
        policy = student(teacher, strategy)
        final_rng.append(torch.get_rng_state().clone())
        common.append(tensor_fingerprint({k: v for k, v in policy.state_dict().items()
                                          if not k.startswith("action_encoder.")}))
        for key, value in teacher.obs_encoder.state_dict().items():
            torch.testing.assert_close(value, policy.obs_encoder.state_dict()[key], rtol=0, atol=0)
        if not policy.is_dual_encoder:
            continue
        frozen_storage = {p.data_ptr() for p in policy.obs_encoder.parameters()}
        assert frozen_storage.isdisjoint(p.data_ptr() for p in policy.action_encoder.parameters())
        for name in ("state_projection", "language_projection"):
            for a, b in zip(getattr(policy.obs_encoder, name).parameters(),
                            getattr(policy.action_encoder, name).parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(policy.obs_encoder.sensor_identity,
                                   policy.action_encoder.sensor_identity, rtol=0, atol=0)
        for group in ("image_encoders", "temporal_projections"):
            for key, a in getattr(policy.obs_encoder, group).items():
                b = getattr(policy.action_encoder, group)[key]
                equal = tensor_fingerprint(a.state_dict()) == tensor_fingerprint(b.state_dict())
                assert equal == (strategy == "s3")
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            observations = make_batch()["obs"]
            expected_z = teacher.obs_encoder(teacher.normalizer.normalize(observations))
            actual_z = policy.encode_stage2(observations)["z_t"]
            torch.testing.assert_close(actual_z, expected_z, rtol=0, atol=0)
    assert len(set(common)) == 1
    assert all(torch.equal(final_rng[0], state) for state in final_rng[1:])


@pytest.mark.parametrize("strategy", ["s1", "s2", "s3", "s4"])
def test_phase_timing_lr_and_gradient_isolation(teacher: object, strategy: str) -> None:
    policy = student(teacher, strategy)
    batch = make_batch()
    original_rng = torch.get_rng_state().clone()
    original_weights = tensor_fingerprint(policy.state_dict())
    report = validate_gradient_routes(policy, batch)
    assert set(report) == {"B", "C"}
    assert torch.equal(original_rng, torch.get_rng_state())
    assert original_weights == tensor_fingerprint(policy.state_dict())
    for stage in ("A", "B", "C"):
        set_stage_trainability(policy, stage)
        policy.train()
        encoder = policy.action_encoder if policy.is_dual_encoder else policy.obs_encoder
        expected = stage != "A" if policy.is_dual_encoder else stage == "C"
        assert all(p.requires_grad == expected for p in encoder.temporal_projections.parameters())
        assert encoder.sensor_identity.requires_grad == expected
        assert all(p.requires_grad == (expected and strategy != "s1")
                   for p in encoder.image_encoders.parameters())
        if policy.is_dual_encoder:
            assert not policy.obs_encoder.training
            assert not any(p.requires_grad for p in policy.obs_encoder.parameters())
    set_stage_trainability(policy, "A")
    optimizer, scheduler = make_optimizer_scheduler(policy, 1000)
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert all(id(p) in ids for p in policy.model.parameters())
    if strategy == "s1":
        assert not any(id(p) in ids for p in policy.obs_encoder.image_encoders.parameters())
    if policy.is_dual_encoder:
        assert not any(id(p) in ids for p in policy.obs_encoder.parameters())
    # Inspect the exact schedule functions, without performing fake optimizer steps.
    start = 1000 if policy.is_dual_encoder else 6000
    assert scheduler.lr_lambdas[0](start - 1) == 0
    assert scheduler.lr_lambdas[0](start) > 0
    assert scheduler.lr_lambdas[0](start + 499) > scheduler.lr_lambdas[0](start)
    if strategy == "s2":
        rates = {group["name"]: group["initial_lr"] for group in optimizer.param_groups}
        assert rates == {"encoder": 8e-5, "resnet": 1e-5, "utility": 1e-4, "action": 1e-4}


class ToyDataset(Dataset):
    """Provide deterministic raw/RAM-equivalent windows and the real sampler rule."""

    def __init__(self) -> None:
        self.task_specs = [SimpleNamespace(task_id="task0"), SimpleNamespace(task_id="task1")]
        self.task_indices = torch.tensor([0] * 3 + [1] * 5)
        self.seen = []

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict:
        self.seen.append(index)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10000 + index)
            batch = make_batch(1)
        return {"obs": {k: v[0] for k, v in batch["obs"].items()},
                "action": batch["action"][0], "action_loss_mask": batch["action_loss_mask"][0],
                "utility": batch["utility"][0], "task_index": self.task_indices[index]}

    def make_task_balanced_sampler(self, generator: torch.Generator) -> WeightedRandomSampler:
        counts = torch.bincount(self.task_indices)
        return WeightedRandomSampler(1.0 / counts[self.task_indices], len(self), True, generator=generator)


def trainer(teacher: object, strategy: str, output: Path, stop: int = 0) -> Stage2AblationTrainer:
    policy = student(teacher, strategy)
    cfg = OmegaConf.to_container(small_config(), resolve=True)
    cfg["policy"]["encoder_strategy"] = strategy
    instance = Stage2AblationTrainer(
        policy, ToyDataset(), output, {"variant": "stage2_transformer", "encoder_strategy": strategy},
        cfg, lambda: 1234, device="cpu", batch_size=4, epochs=7, workers=0, stop_after_steps=stop,
    )
    return instance


@pytest.mark.parametrize("strategy", ["s1", "s2", "s3", "s4"])
def test_exact_training_resume_ram_order_and_checkpoint_inference(
    teacher: object, tmp_path: Path, strategy: str,
) -> None:
    full = trainer(teacher, strategy, tmp_path / "full")
    full.run()
    stopped = trainer(teacher, strategy, tmp_path / "resumed", stop=3)
    observed = []

    def load() -> int:
        observed.append(stopped.global_step)
        assert (stopped.output / "checkpoints/latest.ckpt").is_file()
        return 1234

    stopped.cache_loader = load
    stopped.run()
    assert observed == [1]
    resumed = trainer(teacher, strategy, stopped.output)
    resumed.run(stopped.output / "checkpoints/latest.ckpt")
    assert full.global_step == resumed.global_step == 14
    assert full.cumulative_counts == resumed.cumulative_counts
    assert full.dataset.seen == stopped.dataset.seen + resumed.dataset.seen
    for name in ("policy", "ema"):
        for key, value in getattr(full, name).state_dict().items():
            torch.testing.assert_close(value, getattr(resumed, name).state_dict()[key], rtol=0, atol=0)
    for i, state in full.optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()["state"][i][key], rtol=0, atol=0)
    assert full.scheduler.state_dict() == resumed.scheduler.state_dict()
    if strategy != "s2":
        assert immutable_fingerprint(resumed.policy) == resumed.initialization["immutable_sha256"]
    checkpoint = resumed.output / "checkpoints/latest.ckpt"
    restored, _, metadata = load_evaluation_policy(checkpoint)
    assert metadata["encoder_strategy"] == strategy
    batch = make_batch()
    resumed.ema.eval()
    with torch.no_grad():
        a = resumed.ema.predict_action(batch["obs"], generator=torch.Generator().manual_seed(12))
        b = restored.predict_action(batch["obs"], generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(a["action"], b["action"], rtol=0, atol=0)
    assert sorted(p.name for p in (resumed.output / "checkpoints").glob("*.ckpt")) == [
        "epoch1.ckpt", "epoch6.ckpt", "latest.ckpt"]
    epochs = [json.loads(line) for line in (full.output / "metrics.jsonl").read_text().splitlines()
              if json.loads(line)["event"] == "epoch_complete"]
    assert all(sum(row["task_counts"]) == 8 for row in epochs)
    assert epochs[0]["action_loss"] is None
    assert all(row["action_loss"] is not None for row in epochs[1:])
    first = json.loads((full.output / "metrics.jsonl").read_text().splitlines()[1])
    assert first["event"] == "formal_update_saved_before_ram"


def test_frozen_guard_and_sampler_cursor(teacher: object, tmp_path: Path) -> None:
    workspace = trainer(teacher, "s3", tmp_path / "guard")
    workspace.output.mkdir()
    with torch.no_grad():
        next(workspace.policy.obs_encoder.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="Frozen E500"):
        workspace.save()
    dataset = ToyDataset()
    sampler = Stage2EpochSampler(dataset, 4)
    expected = list(dataset.make_task_balanced_sampler(torch.Generator().manual_seed(43)))
    assert list(sampler) == expected
    sampler.cursor = 1
    assert list(sampler) == expected[4:]


def test_parallel_ram_cache_matches_raw_and_rejects_corruption(
    dataset_arguments: dict, tmp_path: Path,
) -> None:
    """Exercise actual Zarr decoding, metadata validation and the parallel RAM loader."""
    from adavip.manifeel.decoded_sensor_cache import build_decoded_cache
    from adavip.manifeel.stage2_data import (
        Stage2SourceDataset, load_complete_ram_cache, validate_complete_ram_cache,
    )
    source = Stage2SourceDataset(**dataset_arguments)
    cache = tmp_path / "cache"
    manifest = build_decoded_cache(source, str(cache), workers=2)
    cfg_path = tmp_path / "cache_config.yaml"
    OmegaConf.save(OmegaConf.create({"dataset": {
        "_target_": "adavip.manifeel.full_sensor_dataset.FullSensorManiFeelDataset",
        **dataset_arguments}}), cfg_path)
    assert validate_complete_ram_cache(source, str(cache), str(cfg_path)) == manifest
    before = [source[i] for i in range(len(source))]
    assert load_complete_ram_cache(source, str(cache), str(cfg_path), workers=2) == manifest["nbytes"]
    for i, expected in enumerate(before):
        actual = source[i]
        for key, value in expected["obs"].items():
            torch.testing.assert_close(value, actual["obs"][key], rtol=0, atol=0)
        torch.testing.assert_close(expected["action"], actual["action"], rtol=0, atol=0)
    assert not source.groups[0]["data"]["front"].values.flags.writeable
    values = np.load(cache / "task0_front.npy", mmap_mode="r+")
    values.flat[0] += 0.1
    values.flush()
    with pytest.raises(ValueError, match="Corrupt decoded cache"):
        load_complete_ram_cache(source, str(cache), str(cfg_path), workers=2)


def test_persistent_cpu_workers_preserve_training(teacher: object, tmp_path: Path) -> None:
    """Pinned-transport worker changes leave the model/noise/sampler sequence intact."""
    direct = trainer(teacher, "s3", tmp_path / "direct")
    direct.run()
    transported = trainer(teacher, "s3", tmp_path / "transported")
    transported.workers = 2
    transported.run()
    assert direct.cumulative_counts == transported.cumulative_counts
    for key, value in direct.policy.state_dict().items():
        torch.testing.assert_close(value, transported.policy.state_dict()[key], rtol=0, atol=0)
