"""Shape, stop-gradient, identity FiLM, loss, and resume contracts for Stage 2."""

from __future__ import annotations

import io

import torch
from torch import nn

from adavip.manifeel.full_sensor_obs_encoder import FullSensorObsEncoder
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, make_shape_meta
from adavip.manifeel.stage2_policy import Stage2Policy
from scripts.manifeel.train_stage2_t2 import (
    atomic_checkpoint, make_optimizer_scheduler, set_stage_trainability, stage_for_epoch,
)
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class TinyImageEncoder(nn.Module):
    """Emit eight image features for a deterministic small policy test."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 8))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of small RGB images."""
        return self.network(images)


def make_policy() -> Stage2Policy:
    """Build a small U-Net while retaining the seven real 256D token contract."""
    shape_meta = make_shape_meta((3, 8, 8), (3, 8, 8), language_dim=8)
    encoder = FullSensorObsEncoder(shape_meta, image_feature_dim=8,
                                   rgb_model=TinyImageEncoder(), use_group_norm=False)
    scheduler = DDPMScheduler(num_train_timesteps=8, prediction_type="epsilon")
    policy = Stage2Policy(encoder, scheduler, horizon=16, n_obs_steps=2,
                          n_action_steps=8, num_inference_steps=8,
                          down_dims=(16, 32), diffusion_step_embed_dim=16,
                          n_groups=8)
    normalizer = LinearNormalizer()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["state"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["task_embedding"] = SingleFieldLinearNormalizer.create_identity()
    for key in IMAGE_KEYS:
        normalizer[key] = get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    return policy


def make_batch(batch_size: int = 2) -> dict:
    """Create valid T=2 observations and masked 7D actions."""
    observations = {key: torch.rand(batch_size, 2, 3, 8, 8) for key in IMAGE_KEYS}
    observations["state"] = torch.rand(batch_size, 2, 7)
    language = torch.rand(batch_size, 1, 8)
    observations["task_embedding"] = language.expand(-1, 2, -1).clone()
    return {"obs": observations, "action": torch.randn(batch_size, 16, 7),
            "action_loss_mask": torch.ones(batch_size, 16, 7),
            "utility": torch.randn(batch_size, 9)}


def test_stage2_shapes_film_identity_and_losses() -> None:
    """One token per sensor, direct conditioning, and finite two-task losses."""
    torch.manual_seed(7)
    policy = make_policy()
    batch = make_batch()
    encoded = policy.encode_stage2(batch["obs"])
    assert encoded["tokens"].shape == (2, 7, 256)
    assert encoded["z_t"].shape == (2, 256)
    assert encoded["u_hat"].shape == (2, 9)
    assert encoded["conditioning"].shape == (2, 256)
    for index, hypernet in enumerate(policy.film_hypernets):
        context = torch.randn(2, 128)
        torch.testing.assert_close(hypernet(encoded["tokens"][:, index], context),
                                   encoded["tokens"][:, index])
    losses = policy.training_losses(batch, "B")
    assert torch.isfinite(losses["action_loss"])
    assert torch.isfinite(losses["utility_loss"])


def test_stage2_gradient_isolation() -> None:
    """Utility only trains its head; Action reaches encoder but not utility head."""
    torch.manual_seed(8)
    policy = make_policy()
    result = policy.training_losses(make_batch(), "C")
    result["utility_loss"].backward(retain_graph=True)
    assert policy.utility_predictor[0].weight.grad is not None
    assert policy.obs_encoder.fusion[0].weight.grad is None
    assert next(policy.model.parameters()).grad is None
    policy.zero_grad(set_to_none=True)
    result["action_loss"].backward()
    assert policy.utility_predictor[0].weight.grad is None
    assert policy.obs_encoder.fusion[0].weight.grad is not None
    assert torch.count_nonzero(policy.obs_encoder.fusion[0].weight.grad) > 0


def test_stage2_state_round_trip() -> None:
    """Model checkpoint preserves RMS, normalizer, FiLM, and diffusion weights."""
    original = make_policy()
    original.utility_rms.copy_(torch.arange(1, 10))
    stream = io.BytesIO()
    torch.save(original.state_dict(), stream)
    stream.seek(0)
    restored = make_policy()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_stage2_stage_gates_and_optimizer_checkpoint(tmp_path) -> None:
    """A/B/C trainability and optimizer/scheduler resume preserve exact state."""
    policy = make_policy()
    assert [stage_for_epoch(epoch) for epoch in (1, 2, 6, 7, 600)] == ["A", "B", "B", "C", "C"]
    set_stage_trainability(policy, "A")
    assert not any(p.requires_grad for p in policy.obs_encoder.parameters())
    assert not any(p.requires_grad for p in policy.model.parameters())
    assert all(p.requires_grad for p in policy.utility_predictor.parameters())
    set_stage_trainability(policy, "B")
    assert not any(p.requires_grad for p in policy.obs_encoder.parameters())
    assert all(p.requires_grad for p in policy.model.parameters())
    set_stage_trainability(policy, "C")
    assert all(p.requires_grad for p in policy.obs_encoder.parameters())
    optimizer, scheduler = make_optimizer_scheduler(policy, 2)
    optimizer.zero_grad(set_to_none=True)
    policy.training_losses(make_batch(), "C")["action_loss"].backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "latest.ckpt"
    atomic_checkpoint(path, {"model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                             "scheduler": scheduler.state_dict(), "utility_rms": policy.utility_rms})
    checkpoint = torch.load(path, weights_only=False)
    restored = make_policy()
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer, restored_scheduler = make_optimizer_scheduler(restored, 2)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restored_scheduler.load_state_dict(checkpoint["scheduler"])
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    for old, new in zip(policy.parameters(), restored.parameters()):
        torch.testing.assert_close(old, new)
    torch.testing.assert_close(checkpoint["utility_rms"], policy.utility_rms)


def test_preflight_weights_can_reset_before_formal_training() -> None:
    """Combined H200 allocation starts its formal optimizer from a clean seed."""
    torch.manual_seed(11)
    policy = make_policy()
    baseline = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    optimizer, scheduler = make_optimizer_scheduler(policy, 2)
    policy.training_losses(make_batch(), "B")["action_loss"].backward()
    optimizer.step()
    scheduler.step()
    assert any(not torch.equal(value.cpu(), baseline[name]) for name, value in policy.state_dict().items())
    policy.load_state_dict(baseline, strict=True)
    fresh_optimizer, fresh_scheduler = make_optimizer_scheduler(policy, 2)
    assert not fresh_optimizer.state
    assert fresh_scheduler.last_epoch == 0
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value.cpu(), baseline[name])
