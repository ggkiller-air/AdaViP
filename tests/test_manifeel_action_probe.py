"""Deterministic CPU checks for action intervention pairing and episode histories."""

import numpy as np
import pytest
import torch
from torch import nn

from adavip.manifeel.action_probe import (
    normalized_action_change,
    observation_windows,
    override_hypernet,
    override_hypernet_context,
    patch_blur_observations,
    perturb_observations,
    seeded_prediction,
    seeded_condition_prediction,
)


def test_modulation_override_selects_whole_environment_histories() -> None:
    """The two history frames of only the selected environment receive the intervention."""
    encoder = nn.Module()
    encoder.hypernet = nn.Identity()
    values = torch.ones(6, 4)
    select = lambda: np.array([True, False, True])
    with override_hypernet(encoder, torch.zeros(1, 4), select, n_obs_steps=2):
        actual = encoder.hypernet(values)
    torch.testing.assert_close(actual, torch.tensor([[0.] * 4] * 2 + [[1.] * 4] * 2 + [[0.] * 4] * 2))
    torch.testing.assert_close(encoder.hypernet(values), values)


def test_context_override_changes_only_one_slice_and_selected_histories() -> None:
    """HyperNet context can change without mutating its original raw feature/state tensor."""
    encoder = nn.Module()
    encoder.hypernet = nn.Sequential(nn.Linear(11, 11, bias=False))
    with torch.no_grad():
        encoder.hypernet[0].weight.copy_(torch.eye(11))
    context = torch.ones(4, 11)
    select = lambda: np.array([False, True])
    with override_hypernet_context(encoder, torch.zeros(1, 11), "tactile", select, n_obs_steps=2):
        actual = encoder.hypernet(context)
    expected = context.clone()
    expected[2:, 2:4] = 0
    torch.testing.assert_close(actual, expected)
    assert torch.all(context == 1)
    torch.testing.assert_close(encoder.hypernet(context), context)


def test_observation_histories_never_cross_episode_boundaries() -> None:
    """Initial padding and lagging remain inside their original episode."""
    windows, episodes, steps = observation_windows(np.array([3, 8]), [0, 1], 3, 2)
    np.testing.assert_array_equal(windows, [[0, 0], [0, 1], [1, 2], [3, 3], [4, 5], [6, 7]])
    delayed, _, _ = observation_windows(np.array([3, 8]), [0, 1], 3, 2, lag=2)
    np.testing.assert_array_equal(delayed, [[0, 0], [0, 0], [0, 0], [3, 3], [3, 3], [4, 5]])


def test_modality_perturbations_preserve_other_inputs() -> None:
    """Blur and delay must not mutate the input or silently change state."""
    observation = {"wrist": torch.rand(2, 2, 3, 32, 32), "state": torch.ones(2, 2, 7)}
    original = observation["wrist"].clone()
    delayed = {"wrist": torch.zeros_like(original), "state": torch.zeros(2, 2, 7)}
    blurred = perturb_observations(observation, delayed, "wrist", "blur")
    assert blurred["state"] is observation["state"]
    assert not torch.equal(blurred["wrist"], original)
    assert torch.equal(observation["wrist"], original)
    assert perturb_observations(observation, delayed, "wrist", "delay")["wrist"] is delayed["wrist"]


def test_paired_noise_cancels_sampling_variation_and_restores_rng() -> None:
    """Identical diffusion randomness must not be mistaken for modality sensitivity."""
    class StubPolicy:
        """Small stochastic policy with one known observation-dependent term."""

        def predict_action(self, observations: dict) -> dict:
            """Return a noisy action with a deterministic observation offset."""
            return {"action": torch.randn(2, 3, 6) + observations["state"].mean()}

    observation = {"state": torch.zeros(2, 2, 7)}
    rng_state = torch.random.get_rng_state()
    original = seeded_prediction(StubPolicy(), observation, 17)
    repeated = seeded_prediction(StubPolicy(), observation, 17)
    changed = seeded_prediction(StubPolicy(), {"state": torch.ones(2, 2, 7)}, 17)
    assert torch.equal(original, repeated)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    torch.testing.assert_close(changed - original, torch.ones_like(original))


def test_modulation_override_is_temporary_and_keeps_current_input() -> None:
    """Only generated modulation changes, and hooks are removed even on failure."""
    encoder = nn.Module()
    encoder.hypernet = nn.Linear(3, 4)
    observations = torch.ones(2, 3)
    original = encoder.hypernet(observations)
    with override_hypernet(encoder, torch.zeros(1, 4)):
        assert torch.equal(encoder.hypernet(observations), torch.zeros(2, 4))
    assert torch.equal(encoder.hypernet(observations), original)
    with pytest.raises(ValueError):
        with override_hypernet(encoder, torch.zeros(3, 4)):
            encoder.hypernet(observations)
    assert not encoder.hypernet._forward_hooks


def test_action_change_uses_saved_scale_not_offset_or_physical_unit_mixing() -> None:
    """Translation and rotation are normalized before aggregating their RMS."""
    original = torch.zeros(2, 3, 6)
    changed = torch.ones_like(original)
    scale = torch.tensor([2, 2, 2, 4, 4, 4])
    result = normalized_action_change(original, changed, scale)
    torch.testing.assert_close(result["translation"], torch.full((2,), 2.0))
    torch.testing.assert_close(result["rotation"], torch.full((2,), 4.0))


def test_spatial_patch_probe_changes_only_selected_regions() -> None:
    """Both history frames change locally while the other modality stays untouched."""
    observations = {"wrist": torch.rand(1, 2, 3, 32, 32), "tactile": torch.rand(1, 2, 3, 32, 32)}
    changed = patch_blur_observations(observations, "wrist", [(0, 16, 0, 16), (16, 32, 16, 32)])
    assert changed["wrist"].shape == (2, 2, 3, 32, 32)
    torch.testing.assert_close(changed["tactile"][1], observations["tactile"][0])
    torch.testing.assert_close(changed["wrist"][0, ..., 16:, :], observations["wrist"][0, ..., 16:, :])
    assert not torch.equal(changed["wrist"][0, ..., :16, :16], observations["wrist"][0, ..., :16, :16])


def test_cached_condition_sampler_matches_original_global_condition_path() -> None:
    """Cached conditions retain the saved sampling, normalization and action slicing."""
    class IdentityNormalizer:
        """Minimal action normalizer with a nonzero inverse offset."""

        def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
            """Apply a known inverse affine transform."""
            return value / 2 + 3

    class StubConditionPolicy:
        """Tiny stochastic global-condition sampler with the upstream interface."""

        obs_as_global_cond = True
        n_obs_steps = 2
        n_action_steps = 2
        horizon = 4
        action_dim = 6
        kwargs = {}
        normalizer = {"action": IdentityNormalizer()}

        def conditional_sample(self, data, mask, local_cond, global_cond):
            """Generate condition-dependent stochastic normalized actions."""
            return torch.randn_like(data) + global_cond.mean(dim=-1)[:, None, None]

    policy = StubConditionPolicy()
    encoded = torch.ones(2, 2, 4)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        sample = policy.conditional_sample(torch.zeros(2, 4, 6), None, None, encoded.reshape(2, -1))
        expected = policy.normalizer["action"].unnormalize(sample)[:, 1:3]
    actual = seeded_condition_prediction(policy, encoded, 17)
    assert torch.equal(actual, expected)
