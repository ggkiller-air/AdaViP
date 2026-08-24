from pathlib import Path

from adavip.manifeel.video_selection import retain_outcome_videos


class _VideoValue:
    def __init__(self, path: Path) -> None:
        self._path = str(path)


def _make_video(path: Path) -> _VideoValue:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    return _VideoValue(path)


def test_retain_success_and_failure_videos(tmp_path: Path) -> None:
    media_dir = tmp_path / "raw"
    log_data = {
        "test/sim_success_0": False,
        "test/sim_success_1": True,
        "test/sim_success_2": False,
        "test/sim_video_0": _make_video(media_dir / "0.mp4"),
        "test/sim_video_1": _make_video(media_dir / "1.mp4"),
        "test/sim_video_2": _make_video(media_dir / "2.mp4"),
    }

    result = retain_outcome_videos(log_data, tmp_path, "peg_insertion")

    success_path = tmp_path / "videos/peg_insertion_success_env1.mp4"
    failure_path = tmp_path / "videos/peg_insertion_failure_env0.mp4"
    assert result["test/success_video"] == str(success_path)
    assert result["test/failure_video"] == str(failure_path)
    assert result["test/success_video_available"] is True
    assert result["test/failure_video_available"] is True
    assert success_path.is_file()
    assert failure_path.is_file()
    assert not (media_dir / "2.mp4").exists()
    assert not any(key.startswith("test/sim_video_") for key in result)


def test_retain_failure_when_no_success_exists(tmp_path: Path) -> None:
    media_dir = tmp_path / "raw"
    log_data = {
        "test/sim_success_0": False,
        "test/sim_success_1": False,
        "test/sim_video_0": _make_video(media_dir / "0.mp4"),
        "test/sim_video_1": _make_video(media_dir / "1.mp4"),
    }

    result = retain_outcome_videos(log_data, tmp_path, "usb_insertion")

    assert result["test/success_video_available"] is False
    assert "test/success_video" not in result
    assert result["test/failure_video_available"] is True
    assert result["test/failure_video_env"] == 0
    assert Path(result["test/failure_video"]).is_file()
    assert not (media_dir / "1.mp4").exists()
