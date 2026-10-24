"""Read-only plotting contracts for live Stage1 v2 logs."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def plotting():
    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/plot_stage1_v2_training.py"
    specification = importlib.util.spec_from_file_location("stage1_training_plot_test", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def metric_record(event="train", step=20):
    return {"event": event, "global_step": step, "epoch": 1, "timestamp": step,
            "loss": 0.7, "action_loss": 0.2, "future_vision_loss": 0.6,
            "future_tactile_loss": 0.4, "vision_latent_variance": 0.1,
            "tactile_latent_variance": 0.2}


def test_snapshot_deduplicates_resume_ignores_partial_and_reconstructs_epoch_total(tmp_path, plotting):
    path = tmp_path / "metrics.jsonl"
    epoch = metric_record("epoch_end", 40)
    del epoch["loss"]
    records = [{"event": "workspace_ready", "lambda_vision": 0.5, "lambda_tactile": 0.5},
               metric_record(), metric_record(), epoch]
    path.write_text("".join(json.dumps(record) + "\n" for record in records) + '{"unfinished":')
    train, epochs, weights, snapshot = plotting.read_snapshot(path)
    assert len(train) == 1
    assert epochs[0]["loss"] == pytest.approx(0.7)
    assert weights == (0.5, 0.5)
    assert snapshot.endswith(b"\n")
    assert plotting.trailing_mean([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]


def test_snapshot_rejects_changed_loss_formula_and_nonfinite_values(tmp_path, plotting):
    path = tmp_path / "metrics.jsonl"
    ready = {"event": "workspace_ready", "lambda_vision": 0.5, "lambda_tactile": 0.5}
    changed = metric_record()
    changed["loss"] = 0.8
    path.write_text(json.dumps(ready) + "\n" + json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="weighted objective"):
        plotting.read_snapshot(path)
    changed["loss"] = float("nan")
    path.write_text(json.dumps(ready) + "\n" + json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="nonfinite"):
        plotting.read_snapshot(path)
