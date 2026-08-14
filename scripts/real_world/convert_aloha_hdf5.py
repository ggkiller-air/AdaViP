#!/usr/bin/env python3
"""Convert compressed ALOHA-style HDF5 episodes to RDP replay-buffer Zarr."""

from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import h5py
import numcodecs
import numpy as np
import zarr
from PIL import Image


CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def inspect_episode(path: Path) -> tuple[int, float, tuple[int, int, int]]:
    """Validate one source episode and return length, rate, and image shape."""
    with h5py.File(path, "r") as source:
        length = len(source["action"])
        if source["action"].shape != (length, 14):
            raise ValueError(f"Expected action [T,14] in {path}, got {source['action'].shape}")
        if source["observations/qpos"].shape != (length, 14):
            raise ValueError(
                f"Expected observations/qpos [T,14] in {path}, "
                f"got {source['observations/qpos'].shape}"
            )
        encoded_lengths = source["compress_len"]
        source_cameras = tuple(str(value) for value in encoded_lengths.attrs["camera_names"])
        if source_cameras != CAMERA_NAMES:
            raise ValueError(f"Expected cameras {CAMERA_NAMES}, got {source_cameras}")
        encoded = bytes(
            source[f"observations/images/{CAMERA_NAMES[0]}"][0, : encoded_lengths[0, 0]]
        )
        with Image.open(io.BytesIO(encoded)) as image:
            width, height = image.size
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_shape = (height, width, 3)
        return length, float(source.attrs["frame_rate"]), image_shape


def decode_image(encoded: np.ndarray, length: int) -> np.ndarray:
    """Decode one padded JPEG row as RGB uint8."""
    with Image.open(io.BytesIO(encoded[:length].tobytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def convert(inputs: list[Path], output: Path, overwrite: bool) -> None:
    """Convert episodes without loading all decoded images into memory."""
    episode_info = [inspect_episode(path) for path in inputs]
    image_shapes = {info[2] for info in episode_info}
    frame_rates = {info[1] for info in episode_info}
    if len(image_shapes) != 1 or len(frame_rates) != 1:
        raise ValueError("All episodes must use the same image shape and frame rate")

    zarr_path = output / "replay_buffer.zarr"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    total_frames = sum(info[0] for info in episode_info)
    image_shape = episode_info[0][2]
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
    )
    root = zarr.open_group(str(zarr_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    action_array = data.empty(
        "action", shape=(total_frames, 14), chunks=(256, 14), dtype="f4", compressor=compressor
    )
    qpos_array = data.empty(
        "qpos", shape=(total_frames, 14), chunks=(256, 14), dtype="f4", compressor=compressor
    )
    image_arrays = {
        camera: data.empty(
            camera,
            shape=(total_frames, *image_shape),
            chunks=(1, *image_shape),
            dtype="u1",
            compressor=compressor,
        )
        for camera in CAMERA_NAMES
    }

    episode_ends: list[int] = []
    offset = 0
    for path, (length, _, _) in zip(inputs, episode_info):
        print(f"Converting {path} ({length} frames)", flush=True)
        with h5py.File(path, "r") as source:
            action_array[offset : offset + length] = source["action"][:]
            qpos_array[offset : offset + length] = source["observations/qpos"][:]
            encoded_lengths = source["compress_len"][:]
            for camera_index, camera in enumerate(CAMERA_NAMES):
                encoded_images = source[f"observations/images/{camera}"]
                for frame_index in range(length):
                    image_arrays[camera][offset + frame_index] = decode_image(
                        encoded_images[frame_index],
                        int(encoded_lengths[camera_index, frame_index]),
                    )
        offset += length
        episode_ends.append(offset)

    meta.array(
        "episode_ends",
        np.asarray(episode_ends, dtype=np.int64),
        chunks=(max(1, len(episode_ends)),),
        compressor=None,
    )
    root.attrs.update(
        {
            "source_format": "aloha_hdf5",
            "frame_rate": episode_info[0][1],
            "camera_names": list(CAMERA_NAMES),
            "source_files": [str(path.resolve()) for path in inputs],
        }
    )
    print(f"Wrote {total_frames} frames to {zarr_path}", flush=True)


def main() -> int:
    args = parse_args()
    convert(
        inputs=[path.resolve() for path in args.inputs],
        output=args.output.resolve(),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
