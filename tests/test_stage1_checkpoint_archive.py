"""CPU regressions for non-invasive live-training checkpoint retention."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def archive_module():
    """Load the standalone CPU watcher without importing the training stack."""
    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/archive_stage1_v2_checkpoints.py"
    spec = importlib.util.spec_from_file_location("stage1_checkpoint_archive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_checkpoint(run_dir: Path, epoch: int, cursor: int = 0) -> Path:
    """Emulate the trainer's atomic latest replacement and complete state."""
    directory = run_dir / "checkpoints"
    directory.mkdir(exist_ok=True)
    payload = {
        "cfg": {}, "model": {"weights": torch.arange(4)}, "normalizer": {},
        "optimizer": {"state": {}}, "scheduler": {}, "rng": {},
        "sampler": {"epoch": epoch, "consumed": cursor}, "provenance": {"source": "frozen"},
        "epoch": epoch, "global_step": epoch * 3758 + cursor // 32, "sample_cursor": cursor,
    }
    temporary = directory / "latest.partial"
    torch.save(payload, temporary)
    latest = directory / "latest.ckpt"
    temporary.replace(latest)
    return latest


def test_archive_preserves_exact_file_across_latest_replacement(archive_module, tmp_path):
    latest = write_checkpoint(tmp_path, 30)
    original = latest.read_bytes()
    result = archive_module.archive_latest(tmp_path)
    milestone = latest.parent / "epoch30.ckpt"
    assert result["event"] == "preserved"
    assert milestone.stat().st_ino == latest.stat().st_ino
    write_checkpoint(tmp_path, 30, cursor=32000)
    assert milestone.read_bytes() == original
    assert not list((tmp_path / "checkpoint_archive").glob(".capture*"))


@pytest.mark.parametrize("epoch,cursor", [(29, 0), (30, 32), (0, 0)])
def test_archive_rejects_partial_or_non_periodic_epochs(archive_module, tmp_path, epoch, cursor):
    write_checkpoint(tmp_path, epoch, cursor)
    assert archive_module.archive_latest(tmp_path)["event"] == "observed"
    assert not list((tmp_path / "checkpoints").glob("epoch*.ckpt"))


@pytest.mark.parametrize("epoch", [10, 20, 50, 100, 200, 300, 400])
def test_archive_never_races_native_training_milestones(archive_module, tmp_path, epoch):
    write_checkpoint(tmp_path, epoch)
    assert archive_module.archive_latest(tmp_path)["event"] == "native_milestone"
    assert not (tmp_path / "checkpoints" / f"epoch{epoch}.ckpt").exists()


def test_archive_does_not_overwrite_existing_milestone(archive_module, tmp_path):
    write_checkpoint(tmp_path, 30)
    archive_module.archive_latest(tmp_path)
    milestone = tmp_path / "checkpoints" / "epoch30.ckpt"
    original = milestone.read_bytes()
    assert archive_module.archive_latest(tmp_path)["event"] == "already_preserved"
    write_checkpoint(tmp_path, 30, cursor=32)
    conflict = tmp_path / "checkpoints" / "epoch40.ckpt"
    conflict.write_bytes((tmp_path / "checkpoints" / "latest.ckpt").read_bytes())
    write_checkpoint(tmp_path, 40)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        archive_module.archive_latest(tmp_path)
    assert milestone.read_bytes() == original
    assert archive_module.checkpoint_metadata(conflict)["sample_cursor"] == 32


def test_archive_pins_latest_before_inspection(archive_module, tmp_path, monkeypatch):
    latest = write_checkpoint(tmp_path, 30)
    original = latest.read_bytes()
    inspect = archive_module.checkpoint_metadata

    def replace_before_read(path):
        write_checkpoint(tmp_path, 30, cursor=32000)
        return inspect(path)

    monkeypatch.setattr(archive_module, "checkpoint_metadata", replace_before_read)
    assert archive_module.archive_latest(tmp_path)["event"] == "preserved"
    assert (latest.parent / "epoch30.ckpt").read_bytes() == original


def test_archive_handles_missing_and_unchanged_latest(archive_module, tmp_path):
    assert archive_module.archive_latest(tmp_path) is None
    write_checkpoint(tmp_path, 31)
    result = archive_module.archive_latest(tmp_path)
    assert archive_module.archive_latest(tmp_path, previous=result["identity"]) is None


def test_archive_rejects_incomplete_payload_and_cleans_capture(archive_module, tmp_path):
    latest = write_checkpoint(tmp_path, 30)
    torch.save({"epoch": 30}, latest)
    with pytest.raises(ValueError, match="Incomplete training checkpoint"):
        archive_module.archive_latest(tmp_path)
    assert not list((tmp_path / "checkpoint_archive").glob(".capture*"))
