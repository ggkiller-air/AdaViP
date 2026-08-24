"""Retain representative success and failure videos from ManiFeel evaluation."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any


VIDEO_KEY_PATTERN = re.compile(r"test/sim_video_(\d+)")


def _video_path(value: Any) -> Path:
    """Extract the local path from a W&B video value or path-like object."""
    return Path(getattr(value, "_path", value))


def retain_outcome_videos(
    log_data: dict[str, Any],
    output_dir: Path,
    task_id: str,
) -> dict[str, Any]:
    """Keep at most one successful and one failed trajectory video."""
    result = dict(log_data)
    video_paths: dict[int, Path] = {}
    video_keys: list[str] = []
    for key, value in log_data.items():
        match = VIDEO_KEY_PATTERN.fullmatch(key)
        if match is None:
            continue
        env_index = int(match.group(1))
        video_paths[env_index] = _video_path(value)
        video_keys.append(key)

    for key in video_keys:
        result.pop(key, None)

    selected: dict[str, int] = {}
    for outcome, expected_success in (("success", True), ("failure", False)):
        for env_index in sorted(video_paths):
            success_key = f"test/sim_success_{env_index}"
            if success_key in log_data and bool(log_data[success_key]) == expected_success:
                selected[outcome] = env_index
                break

    videos_dir = output_dir / "videos"
    retained_sources: set[Path] = set()
    for outcome in ("success", "failure"):
        env_index = selected.get(outcome)
        available_key = f"test/{outcome}_video_available"
        result[available_key] = env_index is not None
        if env_index is None:
            continue

        source = video_paths[env_index]
        if not source.is_file():
            result[available_key] = False
            continue

        videos_dir.mkdir(parents=True, exist_ok=True)
        destination = videos_dir / f"{task_id}_{outcome}_env{env_index}.mp4"
        shutil.move(str(source), destination)
        retained_sources.add(source)
        result[f"test/{outcome}_video"] = str(destination)
        result[f"test/{outcome}_video_env"] = env_index

    for source in video_paths.values():
        if source not in retained_sources:
            source.unlink(missing_ok=True)

    return result
