"""CPU checks for causal ordering of target contact and environment-level pooling."""

import numpy as np
import pytest

from adavip.manifeel.contact_probe import contact_phase_masks, environment_level_summary


def test_contact_stages_use_policy_input_time_and_exclude_success() -> None:
    """A force at simulator step 2 cannot label the policy input at step 1 as contacted."""
    rewards = np.zeros((4, 2))
    rewards[2, 0] = 1
    force = np.zeros((4, 2, 3))
    force[1, 0, 0] = 1
    initial_force = np.zeros((2, 3))
    initial_force[1, 0] = 1
    phases = contact_phase_masks(rewards, force, initial_force, np.array([0, 1, 2, 3]), 0.01)
    np.testing.assert_array_equal(phases["first_contact_step"], [2, 0])
    np.testing.assert_array_equal(phases["first_success_step"], [3, -1])
    np.testing.assert_array_equal(phases["pre_contact"][:, 0], [True, True, False, False])
    np.testing.assert_array_equal(phases["post_contact_pre_success"][:, 0], [False, False, True, False])
    assert phases["post_contact_pre_success"][:, 1].all()
    assert not (phases["pre_contact"] & phases["post_success"]).any()


def test_phase_summary_weights_environments_not_frames() -> None:
    """Different numbers of phase frames cannot overweight a single environment."""
    values = np.array([[[1.0], [3.0]], [[1.0], [99.0]]])
    selected = np.array([[True, True], [True, False]])
    summary = environment_level_summary(values, selected)
    assert summary["mean"] == [2.0]
    assert summary["environments"] == 2
    assert summary["chunks"] == 3
    assert environment_level_summary(values, np.zeros_like(selected))["mean"] is None


def test_phase_analysis_rejects_nonbenchmark_rewards() -> None:
    """General shaped reward must not be silently reported as benchmark success."""
    with pytest.raises(ValueError, match="binary"):
        contact_phase_masks(np.full((2, 1), 0.2), np.zeros((2, 1, 3)),
                             np.zeros((1, 3)), np.array([0]), 0.01)
