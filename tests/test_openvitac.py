"""Regression coverage of the new benchmark's semantic and gradient boundaries."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from diffusers import DDPMScheduler

from adavip.openvitac.dataset import OpenViTacDataset, resize_rgb
from adavip.openvitac.encoder import OpenViTacObsEncoder
from adavip.openvitac.policy import OpenViTacPolicy
from adavip.openvitac.schema import (ACTION_CODEC, CONDITION_NAMES, IMAGE_KEYS, SCHEMA_VERSION,
                                    SENSOR_KEYS, complete_anchors, condition_presence, make_shape_meta)
from adavip.openvitac.stages import (Stage1Heads, Stage2Policy, UtilityLabelStore,
                                    paired_utility_labels, utility_contract, Stage1Dataset,
                                    Stage2Dataset, fit_utility_rms)
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def small_policy(prediction_type="epsilon"):
    meta = make_shape_meta((3, 8, 8), (3, 8, 8), 4)
    encoder = OpenViTacObsEncoder(meta, token_dim=8, image_feature_dim=8,
                                 rgb_model=nn.Sequential(nn.Conv2d(3, 8, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()))
    policy = OpenViTacPolicy(encoder, DDPMScheduler(num_train_timesteps=10, prediction_type=prediction_type),
                             horizon=8, n_action_steps=2, num_inference_steps=2,
                             diffusion_step_embed_dim=8, down_dims=(8, 16), kernel_size=3, n_groups=4)
    normalizer = LinearNormalizer()
    for key in (*SENSOR_KEYS, "action"):
        normalizer[key] = get_image_range_normalizer() if key in IMAGE_KEYS else SingleFieldLinearNormalizer.create_identity()
    policy.set_normalizer(normalizer)
    return policy


def batch():
    result = {"obs": {key: torch.rand(2, 2, 3, 8, 8) for key in IMAGE_KEYS},
              "action": torch.rand(2, 8, 8), "action_loss_mask": torch.ones(2, 8, 8)}
    result["obs"]["state"] = torch.rand(2, 2, 8)
    result["obs"]["task_embedding"] = torch.rand(2, 1, 4).expand(-1, 2, -1).clone()
    return result


def test_masks_and_next_state_boundary_contract():
    presence = condition_presence()
    assert presence.shape == (9, 6)
    np.testing.assert_array_equal(presence[7], [0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(presence[8], [1, 1, 0, 0, 1, 1])
    anchors = complete_anchors(40, [{"after_frame_index": 19, "source_step_delta": 8}], horizon=8, n_obs_steps=2)
    assert all(not (start <= 19 < start + 8) for t, a, b in anchors for start in [t - 1])
    assert all(t - 1 >= a and t - 1 + 8 < b for t, a, b in anchors)
    assert anchors[-1][0] == 32


@pytest.mark.parametrize("mask", range(1, 9))
def test_removed_input_cannot_change_condition(mask):
    policy, sample = small_policy().eval(), batch()
    obs = policy.normalizer.normalize(sample["obs"])
    flags = torch.from_numpy(condition_presence()[mask]).expand(2, -1)
    changed = {key: value.clone() for key, value in obs.items()}
    for i, key in enumerate(SENSOR_KEYS):
        if not flags[0, i]:
            changed[key] += 20
    torch.testing.assert_close(policy.obs_encoder(obs, flags), policy.obs_encoder(changed, flags), rtol=0, atol=0)


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample"])
def test_dp_full_forward_backward_and_chunk(prediction_type):
    policy, sample = small_policy(prediction_type), batch()
    loss = policy.compute_loss(sample, condition_ids=torch.zeros(2, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss)
    for module in [*policy.obs_encoder.image_encoders.values(), policy.obs_encoder.state_projection,
                   policy.obs_encoder.language_projection, policy.model]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    policy.eval()
    assert policy.predict_action(sample["obs"])["action"].shape == (2, 2, 8)


def test_invalid_actions_and_conditions_are_rejected():
    policy, sample = small_policy(), batch()
    with pytest.raises(ValueError):
        policy.compute_loss(sample, condition_ids=torch.tensor([9, 9]))
    sample["action"] = sample["action"][..., :7]
    with pytest.raises(ValueError):
        policy.compute_loss(sample)


def test_paired_labels_share_draws_and_keep_negative_values(tmp_path):
    policy, sample = small_policy().eval(), batch()
    sample["window_id"] = torch.tensor([1, 2])
    noise, timesteps = torch.randn(2, 8, 8), torch.tensor([1, 4])
    paired = policy.paired_denoising_losses(sample, noise=noise, timesteps=timesteps)
    draw = policy.prepare_denoising_sample(sample, noise, timesteps)
    tokens = policy.obs_encoder.encode_tokens(policy.normalizer.normalize(sample["obs"]))
    for i, flags in enumerate(policy.condition_presence):
        expected = policy.denoising_loss_per_sample(draw, policy.obs_encoder.fuse_tokens(tokens, flags.expand(2, -1)))
        torch.testing.assert_close(paired["losses"][:, i], expected)
    labels = paired_utility_labels(policy, sample, draws=2)
    labels["weighted_losses"] = torch.arange(9, 0, -1).float().expand(2, -1)
    labels["utility"] = labels["weighted_losses"][:, 1:] - labels["weighted_losses"][:, :1]
    contract = utility_contract({"action_codec": ACTION_CODEC}, "teacher", "normalizer")
    path = tmp_path / "labels.npz"
    UtilityLabelStore.save(path, contract, np.array([1, 2]), labels)
    values = UtilityLabelStore.load(path, contract)
    assert (values["utility"] < 0).all()
    with pytest.raises(ValueError):
        UtilityLabelStore.load(path, {**contract, "teacher_sha256": "other"})


def test_stage1_future_and_utility_gradients():
    heads = Stage1Heads(token_dim=8, future_keys=IMAGE_KEYS, resolution=8)
    z_t = torch.rand(2, 8, requires_grad=True)
    targets = {k: torch.rand(2, 3, 8, 8) for k in IMAGE_KEYS}
    result = heads(z_t, torch.randn(2, 8), targets, torch.tensor([True, False]))
    (result["utility_loss"] + result["future_loss"]).backward()
    assert z_t.grad.abs().sum() > 0 and len(heads.future_heads) == 4
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()) for head in heads.future_heads.values())


def test_stage2_identity_initialization_and_gradient_isolation():
    teacher = small_policy().eval()
    stage2 = Stage2Policy(small_policy())
    old_dp = copy.deepcopy(stage2.policy.model.state_dict())
    stage2.initialize_from_teacher(teacher)
    for name, value in stage2.policy.model.state_dict().items():
        torch.testing.assert_close(value, old_dp[name], rtol=0, atol=0)
    sample = batch()
    encoded = stage2.encode(sample["obs"])
    torch.testing.assert_close(encoded["conditioning"], teacher.obs_encoder(teacher.normalizer.normalize(sample["obs"])), rtol=0, atol=0)
    sample["utility"] = torch.randn(2, 8)
    stage2.configure_phase("C")
    result = stage2.training_losses(sample)
    result["action_loss"].backward()
    assert all(p.grad is None for p in stage2.utility_predictor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage2.film_hypernets.parameters())
    stage2.zero_grad(set_to_none=True)
    result = stage2.training_losses(sample)
    result["utility_loss"].backward()
    assert all(p.grad is None for p in stage2.policy.obs_encoder.parameters())
    assert any(p.grad is not None for p in stage2.utility_predictor.parameters())
    assert len(stage2.film_hypernets) == 4


def test_aspect_and_channels_preserved():
    image = np.zeros((3, 6, 3), dtype=np.uint8)
    image[..., 0] = 255
    resized = resize_rgb(image, 8, 8)
    assert resized.shape == (8, 8, 3) and resized[..., 2].max() == 0
    assert (resized[:2] == 0).all() and (resized[2:6, :, 0] == 255).all()


@pytest.fixture
def converted_fixture(tmp_path):
    directory = tmp_path / "task"
    directory.mkdir()
    n = 60
    for key, value in {"state": np.arange(n * 8, dtype=np.float32).reshape(n, 8),
                       "action": np.arange(n * 8, dtype=np.float32).reshape(n, 8) / 100,
                       "action_valid": np.ones(n, dtype=bool), "language_id": np.repeat([0, 1, 0], 20),
                       "timestamp": np.tile(np.arange(20) / 60, 3), "source_time": np.tile(np.arange(20) / 60, 3)}.items():
        np.save(directory / (key + ".npy"), value)
    episodes = [{"episode_index": i, "start": i * 20, "stop": (i + 1) * 20,
                 "language_id": i % 2, "split": "validation" if i == 1 else "train", "videos": {}}
                for i in range(3)]
    (directory / "episodes.json").write_text(json.dumps(episodes))
    windows = [(i, i * 20 + t, i * 20, (i + 1) * 20) for i in range(3) for t, a, b in complete_anchors(20, [], 8, 2)]
    np.save(directory / "windows.npy", np.array(windows))
    np.savez(tmp_path / "language.npz", embeddings=np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32), texts=np.array(["cube", "hexagon"]))
    manifest = {"schema": SCHEMA_VERSION, "action_codec": ACTION_CODEC, "raw_root": str(tmp_path),
                "horizon": 8, "n_obs_steps": 2, "tasks": [{"task_id": "task", "directory": "task"}], "files": {}}
    content = json.dumps(manifest)
    (tmp_path / "manifest.json").write_text(content)
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"status": "complete", "manifest_sha256": hashlib.sha256(content.encode()).hexdigest()}))
    return tmp_path


def test_dataset_episode_language_and_train_only_normalization(converted_fixture):
    meta = make_shape_meta((3, 8, 8), (3, 8, 8), 4)
    train = OpenViTacDataset(converted_fixture, shape_meta=meta, n_action_steps=2)
    validation = OpenViTacDataset(converted_fixture, split="validation", shape_meta=meta, n_action_steps=2)
    assert set(train.index[:, 1]) == {0, 2} and set(validation.index[:, 1]) == {1}
    for key, tensor in train.get_normalizer().state_dict().items():
        torch.testing.assert_close(tensor, validation.get_normalizer().state_dict()[key])
    fake = {k: np.zeros((20, 8, 8, 3), dtype=np.uint8) for k in IMAGE_KEYS}
    train._decode_episode = lambda *args: fake
    validation._decode_episode = lambda *args: fake
    assert train[0]["obs"]["task_embedding"][0].tolist() == [1, 0, 0, 0]
    assert validation[0]["obs"]["task_embedding"][0].tolist() == [0, 1, 0, 0]
    assert not train.get_future(0, offsets=(100,))["valid_mask"].any()
    sampler = train.make_task_balanced_sampler()
    assert len(sampler) == len(train)


def test_checkpoint_roundtrip_recreates_normalizer():
    original = small_policy().eval()
    state = copy.deepcopy(original.state_dict())
    restored = small_policy().eval()
    restored.load_state_dict(state, strict=True)
    sample = batch()
    noise, steps = torch.randn(2, 8, 8), torch.tensor([1, 4])
    torch.testing.assert_close(original.paired_denoising_losses(sample, noise, steps)["losses"],
                               restored.paired_denoising_losses(sample, noise, steps)["losses"], rtol=0, atol=0)


def test_label_join_preserves_identity_and_keeps_future_outside_observations(converted_fixture):
    source = OpenViTacDataset(converted_fixture, shape_meta=make_shape_meta((3, 8, 8), (3, 8, 8), 4), n_action_steps=2)
    source._decode_episode = lambda *args: {k: np.zeros((20, 8, 8, 3), dtype=np.uint8) for k in IMAGE_KEYS}
    contract = utility_contract(source.manifest, "teacher", "normalizer")
    ids = source.index[:, 0] * 10000000 + source.index[:, 2]
    losses = torch.rand(len(ids), 9)
    labels = {"utility": losses[:, 1:] - losses[:, :1], "utility_variance": torch.zeros(len(ids), 8), "weighted_losses": losses}
    path = converted_fixture / "joined_labels.npz"
    UtilityLabelStore.save(path, contract, ids, labels)
    stage1 = Stage1Dataset(source, path, contract)
    sample = stage1[0]
    assert set(sample["obs"]) == set(SENSOR_KEYS)
    assert set(sample["future_obs"]) == {"tactile_left", "tactile_right"}
    assert sample["utility"].shape == (8,) and sample["future_valid"]
    assert "future_obs" not in Stage2Dataset(source, path, contract)[0]


def test_signed_rms_statistics_exclude_validation():
    values = np.array([[-1] * 8, [1] * 8, [10000] * 8])
    np.testing.assert_array_equal(fit_utility_rms(values, [0, 1]), np.ones(8))


def test_workspace_bounded_checkpoint_resume_and_nine_condition_metrics(converted_fixture, monkeypatch):
    from pathlib import Path
    from omegaconf import OmegaConf
    from adavip.openvitac.workspace import OpenViTacWorkspace
    configs = Path(__file__).parents[1] / "configs/openvitac"
    cfg = OmegaConf.merge(OmegaConf.load(configs / "stage0.yaml"), OmegaConf.load(configs / "training.yaml"))
    meta = make_shape_meta((3, 8, 8), (3, 8, 8), 4)
    cfg.dataset.root = str(converted_fixture)
    cfg.dataset.shape_meta = meta
    cfg.dataset.n_action_steps = 2
    cfg.policy.obs_encoder.shape_meta = meta
    cfg.policy.obs_encoder.token_dim = 8
    cfg.policy.obs_encoder.image_feature_dim = 8
    cfg.policy.obs_encoder.rgb_model = {"_target_": "torch.nn.Sequential", "_args_": [
        {"_target_": "torch.nn.Conv2d", "in_channels": 3, "out_channels": 8, "kernel_size": 1},
        {"_target_": "torch.nn.AdaptiveAvgPool2d", "output_size": 1}, {"_target_": "torch.nn.Flatten"}]}
    cfg.policy.horizon, cfg.policy.n_action_steps = 8, 2
    cfg.policy.down_dims, cfg.policy.diffusion_step_embed_dim, cfg.policy.kernel_size = [8, 16], 8, 3
    cfg.policy.n_groups = 4
    cfg.output_dir = str(converted_fixture / "run")
    cfg.training.device, cfg.training.num_epochs = "cpu", 2
    cfg.runtime.stop_after_steps = 1
    cfg.dataloader.batch_size, cfg.dataloader.num_workers = 2, 0
    cfg.val_dataloader.batch_size, cfg.val_dataloader.num_workers = 2, 0
    monkeypatch.setattr(OpenViTacDataset, "_decode_episode", lambda *args: {k: np.zeros((20, 8, 8, 3), dtype=np.uint8) for k in IMAGE_KEYS})
    workspace = OpenViTacWorkspace(cfg)
    workspace.run()
    assert workspace.global_step == 1 and workspace.epoch == 0
    cfg.resume = str(Path(cfg.output_dir) / "checkpoints/latest.ckpt")
    cfg.runtime.stop_after_steps = 1
    restored = OpenViTacWorkspace(cfg)
    restored.run()
    assert restored.global_step == 2 and restored.epoch == 0
    restored._stop_requested = None
    metrics = restored.validate(restored._make_dataset().get_validation_dataset(), torch.device("cpu"))
    assert metrics["validation/task_coverage"] == 1
    assert len([key for key in metrics if key.startswith("validation/macro/")]) == 9
