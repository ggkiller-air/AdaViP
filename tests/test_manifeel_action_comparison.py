from __future__ import annotations

import json
import sys

import numpy as np

from scripts.manifeel.compare_action_trajectories import comparison, main


def test_action_trajectory_comparison_reports_differences() -> None:
    left = np.array([[0.0, 1.0]], dtype=np.float32)
    right = np.array([[0.0, 3.0]], dtype=np.float32)

    result = comparison(left, right)

    assert result["shape_match"] is True
    assert result["exactly_equal"] is False
    assert result["mean_absolute_difference"] == 1.0
    assert result["root_mean_square_difference"] == np.sqrt(2.0)
    assert result["max_absolute_difference"] == 2.0


def test_action_trajectory_comparison_handles_shape_mismatch() -> None:
    result = comparison(np.zeros((1, 2)), np.zeros((2, 1)))

    assert result == {
        "left_shape": [1, 2],
        "right_shape": [2, 1],
        "shape_match": False,
    }


def test_single_mode_trajectory_summary(tmp_path, monkeypatch) -> None:
    trajectory_dir = tmp_path / "correct" / "power_plug_insertion"
    trajectory_dir.mkdir(parents=True)
    np.savez_compressed(
        trajectory_dir / "action_trajectory.npz",
        task_embedding=np.ones(4, dtype=np.float32),
        embedding_source_task_id=np.asarray("power_plug_insertion"),
        action_chunks=np.ones((2, 1, 8, 6), dtype=np.float32),
        executed_actions=np.ones((1, 16, 6), dtype=np.float32),
        action_predictions=np.ones((2, 1, 16, 7), dtype=np.float32),
    )
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_action_trajectories.py",
            "--run-root",
            str(tmp_path),
            "--modes",
            "correct",
            "--output",
            str(output),
        ],
    )

    main()

    result = json.loads(output.read_text())
    assert result["modes"]["correct"]["embedding_source_task_id"] == (
        "power_plug_insertion"
    )
    assert result["pairs"] == {}
