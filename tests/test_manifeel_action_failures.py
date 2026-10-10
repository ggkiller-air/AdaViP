"""Unit coverage for success-stratified action trace summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def test_summarize_run_separates_outcomes(tmp_path: Path) -> None:
    """A transient success is retained and action statistics remain finite."""
    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/summarize_action_failures.py"
    spec = importlib.util.spec_from_file_location("summarize_action_failures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actions = np.ones((4, 2, 7))
    actions[:, 1] *= 2
    successes = np.zeros((4, 2), dtype=bool)
    successes[1, 0] = True
    np.savez_compressed(tmp_path / "rollout_trace.npz", executed_actions=actions,
                        success_by_step=successes)
    np.savez_compressed(tmp_path / "policy_chunks.npz", action_chunks=np.ones((1, 2, 3, 7)))
    result = module.summarize_run(tmp_path)
    assert result["success"]["environment_indices"] == [0]
    assert result["success"]["first_success_step_by_env"] == [1]
    assert result["failure"]["environment_indices"] == [1]
    assert result["failure"]["translation_norm_mean"] > result["success"]["translation_norm_mean"]
    assert result["prediction"]["horizon"] == 3
