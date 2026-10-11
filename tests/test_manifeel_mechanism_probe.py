"""Deterministic CPU tests for modulation variance and parameter comparison."""

import numpy as np
import pytest
import torch

from adavip.manifeel.mechanism_probe import (
    fixed_fusion_output, fusion_attention_logits, modulation_variance, paired_success_difference,
    parameter_distance, phase_matched_donors,
)
from scripts.manifeel.analyze_power_plug_failure_geometry import geometry_summary
from scripts.manifeel.summarize_mechanism_matrix import verify_modulation


def test_phase_modulation_verification_ignores_dynamic_unselected_histories() -> None:
    """Only selected environments are frozen, including all their observation histories."""
    values = np.ones((2, 2, 2, 2, 3))
    selected = np.array([[True, False], [False, True]])
    values[~selected] = np.arange(24).reshape(2, 2, 2, 3)
    assert verify_modulation(values, values, selected, "mean")["selected_environment_chunks"] == 2
    changed = values.copy()
    changed[1, 1, 1, 0, 0] = 2
    with pytest.raises(ValueError, match="varied"):
        verify_modulation(changed, values, selected, "mean")
    with pytest.raises(ValueError, match="never applied"):
        verify_modulation(values, values, np.zeros_like(selected), "zero")
    values[selected] = 0
    verify_modulation(values, values, selected, "zero")


def test_static_environment_offsets_are_not_temporal_adaptation() -> None:
    """Constant but different environments have zero within-environment variation."""
    values = np.array([[[0.0], [1.0]], [[0.0], [1.0]]])
    report = modulation_variance(values, np.ones((2, 2), dtype=bool))
    assert report["within_environment_variance"] == 0
    assert report["between_environment_variance"] == pytest.approx(0.25)
    assert report["saturated_fraction_abs_ge_095"] == 0.5


def test_modulation_summary_weights_environments_equally() -> None:
    """Different trajectory lengths cannot create frame-weighted temporal variance."""
    values = np.array([[[0.0], [1.0]], [[1.0], [0.0]]])
    report = modulation_variance(values, np.array([[True, True], [True, False]]))
    assert report["within_environment_variance"] == pytest.approx(0.125)
    assert report["between_environment_variance"] == pytest.approx(0.0625)
    with pytest.raises(ValueError, match="no valid"):
        modulation_variance(values, np.zeros((2, 2), dtype=bool))


def test_parameter_distance_reports_only_matching_group() -> None:
    """Distance excludes incompatible tensors and does not infer task utility."""
    report = parameter_distance({"encoder.weight": torch.ones(2), "other": torch.ones(3)},
                                {"encoder.weight": torch.zeros(2), "other": torch.zeros(4)}, "encoder.")
    assert report == {"shared_tensors": 1, "identical": False, "relative_l2_distance": 1.0}
    assert parameter_distance({}, {}, "encoder.")["relative_l2_distance"] is None


def test_context_donors_are_distinct_and_share_time_and_phase() -> None:
    """Singleton phases are excluded rather than paired with future or successful inputs."""
    before = np.array([[True, True, False], [True, False, False]])
    after = np.array([[False, False, True], [False, True, True]])
    donors, paired = phase_matched_donors([before, after])
    np.testing.assert_array_equal(paired, [[True, True, False], [False, True, True]])
    np.testing.assert_array_equal(donors[0, :2], [1, 0])
    np.testing.assert_array_equal(donors[1, 1:], [2, 1])
    with pytest.raises(ValueError, match="disjoint"):
        phase_matched_donors([before, before])


def test_failure_geometry_excludes_post_success_motion() -> None:
    """Post-success proximity and movement cannot change the reported pre-success minimum."""
    errors = np.array([[0.02, 0.04], [0.01, 0.03], [0.0, 0.02]])
    plug = np.zeros((3, 2, 1, 3))
    socket = np.zeros_like(plug)
    socket[:, :, 0, 0] = errors
    rollout = {"reward_by_step": np.array([[0, 0], [1, 0], [0, 0]]),
               "keypoints_plug": plug, "keypoints_socket": socket,
               "plug_socket_force": np.ones((3, 2, 3)), "initial_plug_socket_force": np.zeros((2, 3))}
    report = geometry_summary(rollout, 0.025)
    assert report["minimum_pre_success_keypoint_error_m"] == [0.01, 0.02]
    assert report["categories"] == ["success", "contacted_near_goal_failure"]


@pytest.mark.parametrize("bias", [False, True])
def test_attention_logit_replay_matches_real_per_head_weights(bias: bool) -> None:
    """Projection biases and the per-head scaling match PyTorch's actual attention."""
    torch.manual_seed(17)
    attention = torch.nn.MultiheadAttention(8, 2, bias=bias, batch_first=True).eval()
    vision, tactile = torch.rand(3, 8), torch.rand(3, 8)
    tokens = torch.stack((vision, tactile), dim=1)
    with torch.inference_mode():
        _, weights = attention(vision[:, None], tokens, tokens, average_attn_weights=False)
        replay = fusion_attention_logits(attention, vision, tactile).softmax(dim=-1)
    torch.testing.assert_close(replay, weights[:, :, 0], rtol=1e-6, atol=1e-7)


def test_fixed_attention_retains_value_and_output_projections() -> None:
    """Zero query/key scores give exact uniform attention and an equivalent affine map."""
    torch.manual_seed(19)
    attention = torch.nn.MultiheadAttention(8, 2, batch_first=True).eval()
    with torch.no_grad():
        attention.in_proj_weight[:16].zero_()
        attention.in_proj_bias[:16].zero_()
    vision, tactile = torch.rand(3, 8), torch.rand(3, 8)
    tokens = torch.stack((vision, tactile), dim=1)
    with torch.inference_mode():
        expected, _ = attention(vision[:, None], tokens, tokens)
        actual = fixed_fusion_output(attention, vision, tactile, torch.full((2,), 0.5))
    torch.testing.assert_close(actual, expected[:, 0], rtol=1e-6, atol=1e-7)


def test_paired_success_summary_retains_gains_losses_and_seed_structure() -> None:
    """Paired changes distinguish replacing successes from a genuine net gain."""
    original = np.array([[True, False, False], [True, False, False]])
    changed = np.array([[False, True, False], [True, True, False]])
    report = paired_success_difference(original, changed)
    assert report["gained_successes"] == 2
    assert report["lost_successes"] == 1
    assert report["difference_pp"] == pytest.approx(100 / 6)
    assert report["exact_paired_sign_test_p_unadjusted"] == 1
    equal = paired_success_difference(original, original)
    assert equal["paired_environment_bootstrap_95_interval_pp"] == [0, 0]
    assert equal["exact_paired_sign_test_p_unadjusted"] == 1
