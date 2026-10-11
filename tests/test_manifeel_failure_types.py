"""CPU checks for descriptive modulation failure grouping and outcome direction."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.manifeel.analyze_modulation_failure_types import paired_groups, refined_geometry


def make_rollout() -> dict[str, np.ndarray]:
    """Build one environment for every failure type plus a success."""
    errors = np.asarray([[0.04, 0.05, 0.015, 0.03, 0.03],
                         [0.03, 0.04, 0.04, 0.015, 0.005],
                         [0.03, 0.04, 0.04, 0.015, 0.2]])
    plug = np.zeros((3, 5, 1, 3))
    socket = np.zeros_like(plug)
    socket[:, :, 0, 0] = errors
    force = np.ones((3, 5, 3))
    force[:, 0] = 0
    reward = np.zeros((3, 5))
    reward[1, 4] = 1
    return {"reward_by_step": reward, "keypoints_plug": plug, "keypoints_socket": socket,
            "plug_socket_force": force, "initial_plug_socket_force": np.zeros((5, 3))}


def test_refined_categories_cover_distinct_failure_geometry() -> None:
    """Near-then-departed is separate from unfinished but still near."""
    report = refined_geometry(make_rollout(), 0.01)
    assert report["categories"] == ["no_contact_failure", "contacted_far_failure",
                                     "near_then_departed_failure", "near_goal_unfinished_failure", "success"]
    assert sum(report["counts"].values()) == 5
    assert report["last_pre_success_keypoint_error_m"][-1] == pytest.approx(0.005)


def test_initial_contact_prevents_no_contact_label() -> None:
    """Executed force may be zero even if contact was present at reset."""
    rollout = make_rollout()
    rollout["initial_plug_socket_force"][0, 0] = 0.01
    assert refined_geometry(rollout, 0.01)["categories"][0] == "contacted_far_failure"


def test_group_direction_uses_failed_counterfactual() -> None:
    """Normal-only rescue must describe the zero run's failure, not the success."""
    normal = {"success": [True, False, True, False],
              "categories": ["success", "near_then_departed_failure", "success", "contacted_far_failure"]}
    zero = {"success": [False, True, True, False],
            "categories": ["no_contact_failure", "success", "success", "near_goal_unfinished_failure"]}
    report = paired_groups(normal, zero)
    assert report["groups"]["normal_only_success"]["failure_type_counts"]["no_contact_failure"] == 1
    assert report["groups"]["zero_only_success"]["failure_type_counts"]["near_then_departed_failure"] == 1
    assert report["both_success"] == report["both_failure"] == 1
    assert np.asarray(report["normal_rows_zero_columns"]).sum() == 4


@pytest.mark.parametrize("threshold,multiplier", [(float("nan"), 2.0), (0.01, 1.0), (0.0, 2.0)])
def test_invalid_thresholds_are_rejected(threshold: float, multiplier: float) -> None:
    """Invalid thresholds cannot silently change class membership."""
    with pytest.raises(ValueError):
        refined_geometry(make_rollout(), threshold, multiplier)


def test_inconsistent_success_labels_are_rejected() -> None:
    """Success/category disagreement must fail before counting a rescue."""
    with pytest.raises(ValueError, match="categories disagree"):
        paired_groups({"success": [True], "categories": ["contacted_far_failure"]},
                      {"success": [True], "categories": ["success"]})
