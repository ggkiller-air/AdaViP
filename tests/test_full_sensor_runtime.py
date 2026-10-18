"""Regression coverage for short, writable PyTorch multiprocessing socket paths."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from adavip.manifeel.training_runtime import check_dataloader_ipc


def test_runtime_transfers_worker_tensors_with_short_tmpdir() -> None:
    import json
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="ipc", dir=tempfile.gettempdir()) as directory:
        result = subprocess.run(
            [sys.executable, "-c", "import json; from adavip.manifeel.training_runtime import check_dataloader_ipc; print(json.dumps(check_dataloader_ipc(10)))"],
            env={**os.environ, "TMPDIR": directory}, capture_output=True, text=True, check=True, timeout=40,
        )
    report = json.loads(result.stdout)
    assert report["event"] == "ipc_preflight_ok"
    assert report["socket_path_bytes"] < 108


def test_runtime_rejects_long_socket_path_before_starting_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    long_directory = tmp_path / ("long" * 40)
    monkeypatch.setenv("TMPDIR", str(long_directory))
    monkeypatch.setattr(tempfile, "tempdir", str(long_directory))
    with pytest.raises(RuntimeError, match="too long"):
        check_dataloader_ipc()


def test_runtime_rejects_silent_tmpdir_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tempfile

    monkeypatch.setenv("TMPDIR", str(tmp_path / "missing"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(RuntimeError, match="not usable"):
        check_dataloader_ipc()
