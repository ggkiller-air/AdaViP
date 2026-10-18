"""CPU-only regression coverage for plotting live Stage0 metrics."""

import json
from pathlib import Path

import pytest

from scripts.manifeel.plot_full_sensor_training import load_epoch_records, plot_training


def make_record(epoch: int, validation_loss: float) -> dict:
    """Build a minimal completed-epoch record."""
    return {"epoch": epoch, "global_step": epoch * 10, "train/loss": 0.1 / epoch,
            "validation/macro/full": validation_loss, "validation/micro/full": validation_loss / 2}


def test_epoch_records_ignore_partial_tail_and_deduplicate(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    records = [make_record(2, 0.3), make_record(1, 0.2), make_record(2, 0.4)]
    path.write_text("".join(json.dumps(record) + "\n" for record in records) + '{"epoch":3')
    loaded = load_epoch_records(path)
    assert [record["epoch"] for record in loaded] == [1, 2]
    assert loaded[-1]["validation/macro/full"] == 0.4


def test_epoch_records_reject_corrupt_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("broken\n")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_epoch_records(path)


def test_plot_writes_png_and_selects_best_validation_not_latest(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(json.dumps(make_record(epoch, loss)) + "\n"
                            for epoch, loss in [(1, 0.1), (2, 0.05), (3, 0.2)]))
    output = tmp_path / "plots" / "loss.png"
    summary = plot_training(path, output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert summary["best"]["epoch"] == 2
    assert summary["latest"]["epoch"] == 3
    assert json.loads(output.with_suffix(".json").read_text()) == summary
