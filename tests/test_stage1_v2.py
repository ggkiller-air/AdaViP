"""CPU-friendly contract tests for the Stage 1 v2 Teacher path."""

from __future__ import annotations

import numpy as np
import pytest

from adavip.manifeel.stage1_v2_schema import (
    MASK_NAMES,
    SENSOR_TOKEN_NAMES,
    condition_presence,
    condition_probabilities,
    make_shape_meta,
)


def test_stage1_v2_masks_keep_language_and_match_eight_dropout_modes() -> None:
    presence = condition_presence()
    assert presence.shape == (9, 7)
    assert tuple(MASK_NAMES) == (
        "full", "front", "side", "wrist", "left_tactile", "right_tactile",
        "proprio", "all_vision", "all_tactile",
    )
    assert tuple(SENSOR_TOKEN_NAMES[-1:]) == ("language",)
    assert presence[:, -1].all()
    np.testing.assert_array_equal(presence[7, :3], [False, False, False])
    np.testing.assert_array_equal(presence[8, 3:5], [False, False])


def test_stage1_v2_condition_probabilities() -> None:
    probabilities = condition_probabilities()
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    np.testing.assert_allclose(probabilities[0], 0.8)
    np.testing.assert_allclose(probabilities[1:], 0.025)


def test_stage1_v2_shape_contract() -> None:
    shape_meta = make_shape_meta(language_dim=512)
    from adavip.manifeel.stage1_v2_schema import validate_stage1_shape_meta

    validate_stage1_shape_meta(shape_meta)


def test_stage1_v2_dataset_uses_t_minus_one_action_prefix(tmp_path) -> None:
    pytest.importorskip("torch")
    zarr = pytest.importorskip("zarr")
    from adavip.manifeel.stage1_v2_dataset import Stage1V2ManiFeelDataset

    shape_meta = make_shape_meta(camera_shape=(3, 4, 4), tactile_shape=(3, 4, 4))
    specs = [
        {
            "task_id": "task_a", "text": "task a", "dataset": "task_a",
            "isaacgym_cfg_name": "a.yaml", "action_dim": 6,
        },
    ]
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    store = zarr.open_group(str(dataset_root / "task_a"), mode="w")
    episode_ends = np.asarray([30, 60], dtype=np.int64)
    store.create_group("meta").create_dataset("episode_ends", data=episode_ends)
    data = store.create_group("data")
    for key in ("front", "side", "wrist", "left_tactile_camera_taxim", "right_tactile_camera_taxim"):
        frames = np.broadcast_to(np.arange(60, dtype=np.float32)[:, None, None, None] / 60,
                                 (60, 4, 4, 3)).copy()
        data.create_dataset(key, data=frames)
    data.create_dataset("state", data=np.zeros((60, 7), dtype=np.float32))
    actions = np.arange(60 * 6, dtype=np.float32).reshape(60, 6)
    data.create_dataset("action", data=actions)
    embedding_path = tmp_path / "embeddings.npz"
    np.savez(embedding_path, task_ids=np.asarray(["task_a"]), embeddings=np.zeros((1, 512), dtype=np.float32))

    dataset = Stage1V2ManiFeelDataset(
        shape_meta=shape_meta,
        dataset_root=str(dataset_root),
        task_embedding_path=str(embedding_path),
        task_specs=specs,
        val_ratio=0.5,
        max_train_episodes=1,
        seed=1,
    )
    sample = dataset[0]
    assert sample["obs"]["state"].shape == (6, 7)
    assert sample["action"].shape == (16, 7)
    assert sample["action_history"].shape == (8, 7)
    assert sample["future_obs"]["front"].shape == (4, 3, 4, 4)
    assert int(sample["source_index"]) == 8
    assert len(dataset) == 16
    np.testing.assert_array_equal(sample["action"][0, :6].numpy(), actions[7])
    np.testing.assert_array_equal(sample["action"][1, :6].numpy(), actions[8])
    np.testing.assert_array_equal(sample["action_history"][-1, :6].numpy(), actions[7])
    np.testing.assert_allclose(sample["obs"]["front"][:, 0, 0, 0].numpy(), np.arange(3, 9) / 60)
    np.testing.assert_allclose(sample["future_obs"]["front"][:, 0, 0, 0].numpy(), np.arange(9, 13) / 60)
    np.testing.assert_array_equal(sample["action"][1:5, :6].numpy(), actions[8:12])
    for index in range(len(dataset)):
        sample = dataset[index]
        assert sample["action_valid_mask"].all()
        assert sample["future_valid_mask"].all()
        assert sample["action_history_valid_mask"].all()


def test_balanced_sampler_exhausts_before_reshuffling_across_epochs() -> None:
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from adavip.manifeel.stage1_v2_training import BalancedTaskSampler

    dataset = SimpleNamespace(task_lengths=[5, 8])
    stream = [index for epoch in range(3) for index in BalancedTaskSampler(dataset, 42, epoch)]
    assert len(stream) == 48
    for length, offset in ((5, 0), (8, 5)):
        task_stream = [index - offset for index in stream if offset <= index < offset + length]
        for start in range(0, len(task_stream) - length + 1, length):
            assert sorted(task_stream[start:start + length]) == list(range(length))
    full = list(BalancedTaskSampler(dataset, 42, epoch=1))
    assert list(BalancedTaskSampler(dataset, 42, epoch=1, offset=7)) == full[7:]


@pytest.fixture
def tiny_teacher(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from adavip.manifeel import stage1_v2_policy
    from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

    def backbone(*args, **kwargs):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 1), torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(), torch.nn.Linear(8, 512),
        )

    monkeypatch.setattr(stage1_v2_policy, "get_resnet", backbone)
    torch.manual_seed(42)
    teacher = stage1_v2_policy.Stage1V2TeacherPolicy(
        make_shape_meta(camera_shape=(3, 4, 4), tactile_shape=(3, 4, 4)),
        DDPMScheduler(num_train_timesteps=100),
        down_dims=(16, 32), n_groups=8, diffusion_step_embed_dim=16,
    )
    normalizer = LinearNormalizer()
    for key in (*teacher.shape_meta["obs"], "action"):
        normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    teacher.set_normalizer(normalizer)
    return teacher


def synthetic_batch(teacher, batch_size=2):
    import torch

    observations = {
        key: torch.randn(batch_size, 6, *entry["shape"])
        for key, entry in teacher.shape_meta["obs"].items()
    }
    observations["task_embedding"] = observations["task_embedding"][:, :1].expand(-1, 6, -1).clone()
    return {
        "obs": observations,
        "future_obs": {key: value[:, :4].clone() for key, value in observations.items() if value.ndim == 5},
        "action": torch.randn(batch_size, 16, 7),
        "action_loss_mask": torch.ones(batch_size, 16, 7),
        "action_history": torch.randn(batch_size, 8, 7),
        "action_history_valid_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        "action_dimension_mask": torch.ones(batch_size, 7),
        "future_valid_mask": torch.ones(batch_size, 4, dtype=torch.bool),
    }


def test_teacher_shapes_gradients_ema_and_normalizer_restore(tiny_teacher) -> None:
    import torch

    teacher = tiny_teacher
    teacher.train()
    batch = synthetic_batch(teacher)
    assert all(not encoder.training for encoder in teacher.target_image_encoders.values())
    losses = teacher.compute_loss_with_metrics(batch, torch.zeros(2, dtype=torch.long))
    assert losses["z"].shape == (2, 256)
    sum(losses[name] for name in ("action_loss", "future_vision_loss", "future_tactile_loss")).backward()
    for name in ("image_encoders", "image_temporal", "proprio_frame", "proprio_temporal",
                 "language_projection", "action_history", "state_tokenizer", "cross_attention",
                 "model", "future_vision_head", "future_tactile_head"):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in getattr(teacher, name).parameters()), name
    assert all(parameter.grad is None for parameter in teacher.target_image_encoders.parameters())
    state = teacher.state_dict()
    teacher.normalizer.params_dict.clear()
    teacher.load_state_dict(state, strict=True)
    assert teacher.normalizer["action"].normalize(batch["action"]).shape == (2, 16, 7)
    with torch.no_grad():
        parameter = next(teacher.image_encoders.parameters())
        target = next(teacher.target_image_encoders.parameters())
        previous = target.clone()
        parameter.add_(1)
        teacher.update_target_encoder()
        torch.testing.assert_close(target, previous * teacher.future_ema_decay
                                   + parameter * (1 - teacher.future_ema_decay))


def test_teacher_all_masks_block_both_paths(tiny_teacher) -> None:
    import torch

    teacher = tiny_teacher.eval()
    batch = synthetic_batch(teacher, 1)
    with torch.no_grad():
        for condition in range(1, 9):
            presence = teacher.condition_presence[condition:condition + 1]
            original = teacher._state_token(batch["obs"], batch["action_history"],
                                            batch["action_history_valid_mask"], presence)
            changed = {key: value.clone() for key, value in batch["obs"].items()}
            for key, is_present in zip(teacher.shape_meta["obs"], presence[0]):
                if not is_present:
                    changed[key].normal_(mean=20, std=5)
            modified = teacher._state_token(changed, batch["action_history"],
                                            batch["action_history_valid_mask"], presence)
            for left, right in zip(original, modified):
                torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_teacher_future_inputs_do_not_change_action_conditioning(tiny_teacher) -> None:
    import torch

    teacher = tiny_teacher.eval()
    batch = synthetic_batch(teacher)
    sample = teacher.prepare_denoising_sample(batch)
    conditions = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        first = teacher.compute_loss_with_metrics(batch, conditions, sample)
        batch["action"][:, 1:5].normal_(mean=30, std=5)
        for value in batch["future_obs"].values():
            value.normal_()
        second = teacher.compute_loss_with_metrics(batch, conditions, sample)
    torch.testing.assert_close(first["z"], second["z"], rtol=0, atol=0)
    torch.testing.assert_close(first["action_loss"], second["action_loss"], rtol=0, atol=0)
    assert first["future_vision_loss"] != second["future_vision_loss"]


def test_teacher_diffusion_and_executed_action_history_required(tiny_teacher) -> None:
    import torch

    teacher = tiny_teacher.eval()
    batch = synthetic_batch(teacher, 1)
    with pytest.raises(ValueError, match="executed actions"):
        teacher.predict_action(batch["obs"])
    teacher.num_inference_steps = 2
    result = teacher.predict_action({**batch["obs"], "action_history": batch["action_history"]})
    assert result.shape == (1, 8, 7)
    assert torch.isfinite(result).all()


def test_full_checkpoint_exact_resume_and_early_milestone_protection(tiny_teacher, tmp_path) -> None:
    import torch
    from omegaconf import OmegaConf
    from adavip.manifeel.stage1_v2_training import Stage1V2Workspace, restore_rng

    workspace = Stage1V2Workspace.__new__(Stage1V2Workspace)
    workspace._output_dir = str(tmp_path)
    workspace.cfg = OmegaConf.create({"training": {"seed": 42}})
    workspace.model = tiny_teacher.train()
    workspace.optimizer = torch.optim.AdamW(
        [parameter for parameter in tiny_teacher.parameters() if parameter.requires_grad], lr=1e-4,
    )
    workspace.scheduler = torch.optim.lr_scheduler.StepLR(workspace.optimizer, step_size=5)
    workspace.epoch, workspace.global_step, workspace.sample_cursor = 0, 1, 2
    workspace.epoch_sums, workspace.task_counts = {"samples": 2}, [2]
    workspace.lambda_vision, workspace.lambda_tactile = 0.3, 0.2
    workspace.provenance = {"unit_test": True}
    batch = synthetic_batch(tiny_teacher)

    def update():
        workspace.optimizer.zero_grad(set_to_none=True)
        loss = workspace.model.compute_loss(batch, 0.3, 0.2)
        loss.backward()
        workspace.optimizer.step()
        workspace.model.update_target_encoder()
        workspace.scheduler.step()
        return float(loss.detach())

    update()
    checkpoint = workspace.save("latest")
    expected_loss = update()
    expected = {name: value.clone() for name, value in tiny_teacher.state_dict().items()}
    rng = workspace.restore(str(checkpoint), torch.device("cpu"))
    restore_rng(rng)
    assert update() == expected_loss
    for name, value in tiny_teacher.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    assert workspace.sample_cursor == 2
    assert workspace.task_counts == [2]
    milestone = workspace.save("epoch1")
    original_size = milestone.stat().st_size
    with pytest.raises(FileExistsError, match="Preserved milestone"):
        workspace.save("epoch1")
    assert milestone.stat().st_size == original_size
