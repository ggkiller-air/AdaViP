"""Teacher MLP fusion copying, presence flags, gradients, and utility probes."""

from __future__ import annotations

import copy
import io

import numpy as np
import torch

from adavip.manifeel.stage2_mlp_policy import Stage2MlpPolicy
from scripts.manifeel.eval_stage2_mlp_utility import per_dimension_correlation
from scripts.manifeel.train_stage2_mlp_t2 import (
    restore_policy_state, set_stage_trainability, utility_diagnostics,
)
from test_stage2_policy import make_batch, make_policy


def make_mlp_policy() -> tuple[Stage2MlpPolicy, object]:
    """Build one small MLP control using the existing Stage 2 test Teacher."""
    teacher = make_policy()
    policy = Stage2MlpPolicy(
        copy.deepcopy(teacher.obs_encoder), copy.deepcopy(teacher.noise_scheduler),
        horizon=16, n_obs_steps=2, n_action_steps=8, num_inference_steps=8,
        down_dims=(16, 32), diffusion_step_embed_dim=16, n_groups=8,
    )
    policy.initialize_from_teacher(teacher)
    return policy, teacher


def test_teacher_fusion_is_copied_twice_without_sharing() -> None:
    """z_t and final MLP start identically but own separate trainable parameters."""
    policy, teacher = make_mlp_policy()
    for name, parameter in teacher.obs_encoder.fusion.state_dict().items():
        torch.testing.assert_close(policy.obs_encoder.fusion.state_dict()[name], parameter)
        torch.testing.assert_close(policy.final_fusion.state_dict()[name], parameter)
    assert policy.obs_encoder.fusion[0].weight.data_ptr() != policy.final_fusion[0].weight.data_ptr()
    assert not hasattr(policy, "cross_attention")
    assert not hasattr(policy, "query_projection")
    tokens = policy.obs_encoder.encode_tokens(policy.normalizer.normalize(make_batch()["obs"]))
    captured = []
    handle = policy.final_fusion[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()),
    )
    presence = torch.tensor([[True, False, True, True, False, True, True]] * 2)
    encoded = policy.condition_from_tokens(tokens, presence)
    handle.remove()
    torch.testing.assert_close(captured[0][:, -7:], presence.float())
    flattened = captured[0][:, :-7].reshape(2, 7, 256)
    torch.testing.assert_close(flattened[:, 5:], tokens[:, 5:])
    torch.testing.assert_close(encoded["z_t"], policy.obs_encoder.fuse_tokens(tokens, presence))
    torch.testing.assert_close(encoded["conditioning"], encoded["z_t"])
    assert encoded["conditioning"].shape == (2, 256)


def test_mlp_action_and_utility_gradient_isolation() -> None:
    """Only Action reaches both MLP copies; neither Action nor Utility crosses stops."""
    policy, _ = make_mlp_policy()
    result = policy.training_losses(make_batch(), "C")
    result["utility_loss"].backward(retain_graph=True)
    assert policy.utility_predictor[0].weight.grad is not None
    assert policy.obs_encoder.fusion[0].weight.grad is None
    assert policy.final_fusion[0].weight.grad is None
    policy.zero_grad(set_to_none=True)
    result["action_loss"].backward()
    assert policy.utility_predictor[0].weight.grad is None
    # Identity FiLM initially blocks the z_t-to-FiLM path; it opens once
    # the zero-initialized HyperNet output layer receives an update.
    assert torch.count_nonzero(policy.obs_encoder.fusion[0].weight.grad) == 0
    assert torch.count_nonzero(policy.final_fusion[0].weight.grad) > 0
    policy.zero_grad(set_to_none=True)
    with torch.no_grad():
        policy.film_hypernets[0].network[-1].weight.fill_(0.01)
    policy.training_losses(make_batch(), "C")["action_loss"].backward()
    assert torch.count_nonzero(policy.obs_encoder.fusion[0].weight.grad) > 0
    set_stage_trainability(policy, "B")
    assert not any(parameter.requires_grad for parameter in policy.obs_encoder.parameters())
    assert all(parameter.requires_grad for parameter in policy.final_fusion.parameters())


def test_mlp_diagnostics_and_checkpoint_round_trip() -> None:
    """The three utility conditions share one loss target and weights restore."""
    policy, _ = make_mlp_policy()
    batch = make_batch()
    result = policy.training_losses(batch, "B")
    diagnostics = utility_diagnostics(policy, batch, result)
    assert set(diagnostics["counterfactual_action_loss"]) == {"predicted", "zero", "shuffled"}
    assert all(torch.isfinite(torch.tensor(value)) for value in diagnostics["counterfactual_action_loss"].values())
    assert diagnostics["film_delta_gamma_abs_mean"] == 0
    assert diagnostics["film_beta_abs_mean"] == 0
    policy.eval()
    paired = policy.paired_denoising_losses(batch)
    assert paired["utility"].shape == (2, 9)
    torch.testing.assert_close(paired["utility"], paired["losses"][:, 1:] - paired["losses"][:, :1])
    stream = io.BytesIO()
    torch.save(policy.state_dict(), stream)
    stream.seek(0)
    restored, _ = make_mlp_policy()
    restore_policy_state(restored, torch.load(stream, weights_only=True), torch.device("cpu"))
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_offline_correlation_handles_constant_utility() -> None:
    """The offline report marks a constant GT dimension as undefined."""
    values = per_dimension_correlation(
        np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]]),
        np.array([[2.0, 1.0], [4.0, 1.0], [6.0, 1.0]]),
    )
    assert values == [1.0, None]
