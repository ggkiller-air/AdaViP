#!/usr/bin/env python3
"""Validate a prepared Piper RDP Zarr dataset before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


EXPECTED_FIELDS = {
    "action": ((7,), None),
    "external_img": ((240, 320, 3), np.uint8),
    "right_wrist_img": ((240, 320, 3), np.uint8),
    "right_robot_qpos": ((7,), None),
    "right_gelsight_marker_offset_emb": ((15,), None),
}


def validate_dataset(dataset: Path) -> tuple[int, int]:
    """Validate required arrays and return frame and episode counts."""
    store = dataset.resolve() / "replay_buffer.zarr"
    if not store.is_dir():
        raise FileNotFoundError(f"Zarr store not found: {store}")

    root = zarr.open(str(store), mode="r")
    if "data" not in root or "meta" not in root:
        raise ValueError("Zarr must contain data and meta groups")

    data = root["data"]
    frame_count: int | None = None
    for key, (trailing_shape, expected_dtype) in EXPECTED_FIELDS.items():
        if key not in data:
            raise ValueError(f"Missing data array: {key}")
        array = data[key]
        if tuple(array.shape[1:]) != trailing_shape:
            raise ValueError(
                f"Invalid {key} shape: {array.shape}; expected [N, "
                f"{', '.join(map(str, trailing_shape))}]"
            )
        if expected_dtype is not None and array.dtype != expected_dtype:
            raise ValueError(
                f"Invalid {key} dtype: {array.dtype}; expected {expected_dtype}"
            )
        if frame_count is None:
            frame_count = int(array.shape[0])
        elif array.shape[0] != frame_count:
            raise ValueError(
                f"Frame count mismatch for {key}: {array.shape[0]} != {frame_count}"
            )

        if expected_dtype is None:
            for chunk_start in range(0, array.shape[0], 4096):
                values = array[chunk_start : chunk_start + 4096]
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite values found in {key}")

    if frame_count is None or frame_count == 0:
        raise ValueError("Dataset has no frames")
    if "episode_ends" not in root["meta"]:
        raise ValueError("Missing meta/episode_ends array")
    episode_ends = np.asarray(root["meta/episode_ends"][:])
    if episode_ends.ndim != 1 or not np.issubdtype(episode_ends.dtype, np.integer):
        raise ValueError("meta/episode_ends must be a one-dimensional integer array")
    if len(episode_ends) == 0 or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("meta/episode_ends must be non-empty and strictly increasing")
    if int(episode_ends[-1]) != frame_count:
        raise ValueError(
            f"Last episode end {episode_ends[-1]} does not equal {frame_count} frames"
        )
    return frame_count, len(episode_ends)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    frames, episodes = validate_dataset(args.dataset)
    print(f"RDP dataset valid: {frames} frames, {episodes} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
