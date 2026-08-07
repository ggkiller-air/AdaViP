#!/usr/bin/env python3
"""Render one complete ManiFeel Zarr episode as a synchronized MP4 grid."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import av
import cv2
import numpy as np
import zarr

from manifeel.utils.shear_tactile_viz_utils import (
    visualize_tactile_shear_image,
)


PANEL_SPECS = (
    ("front", "Front"),
    ("side", "Side"),
    ("wrist", "Wrist A"),
    ("wrist_2", "Wrist B"),
    ("left_tactile_camera_taxim", "TacRGB Left"),
    ("right_tactile_camera_taxim", "TacRGB Right"),
    ("tactile_force_field_right", "TacFF Right: normal + shear"),
    ("tactile_depth_right", "TacDepth Right"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Zero-based episode index. Defaults to the longest episode.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--panel-height", type=int, default=256)
    return parser.parse_args()


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def letterbox(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return canvas


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    cv2.rectangle(
        output,
        (0, 0),
        (text_width + 16, text_height + baseline + 12),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        output,
        label,
        (8, text_height + 6),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output


def render_force_field(force_field: np.ndarray) -> np.ndarray:
    image_bgr = visualize_tactile_shear_image(
        force_field[..., 0],
        force_field[..., 1:],
        normal_force_threshold=0.004,
        shear_force_threshold=0.001,
        resolution=25,
    )
    return to_uint8_rgb(image_bgr[..., ::-1])


def render_depth(depth: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(depth / scale, 0.0, 1.0)
    grayscale = np.round(normalized * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(grayscale, cv2.COLORMAP_TURBO)
    return heatmap_bgr[..., ::-1]


def require_keys(keys: Iterable[str], available: Iterable[str]) -> None:
    available_set = set(available)
    missing = [key for key in keys if key not in available_set]
    if missing:
        raise KeyError(f"Dataset is missing required keys: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    root = zarr.open(str(args.zarr_path), mode="r")
    data = root["data"]
    require_keys((key for key, _ in PANEL_SPECS), data.keys())

    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    episode_lengths = episode_ends - episode_starts
    episode = (
        int(np.argmax(episode_lengths))
        if args.episode is None
        else args.episode
    )
    if episode < 0 or episode >= len(episode_ends):
        raise IndexError(
            f"Episode {episode} is outside [0, {len(episode_ends) - 1}]"
        )

    start = int(episode_starts[episode])
    end = int(episode_ends[episode])
    sequences: Dict[str, np.ndarray] = {
        key: np.asarray(data[key][start:end]) for key, _ in PANEL_SPECS
    }
    depth_scale = max(
        float(np.quantile(sequences["tactile_depth_right"], 0.995)),
        np.finfo(np.float32).eps,
    )

    panel_size = (args.panel_width, args.panel_height)
    output_width = args.panel_width * 4
    output_height = args.panel_height * 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    codec = "libx264"
    try:
        av.codec.Codec(codec, "w")
    except av.error.FFmpegError:
        codec = "mpeg4"

    with av.open(str(args.output), mode="w") as container:
        stream = container.add_stream(codec, rate=args.fps)
        stream.width = output_width
        stream.height = output_height
        stream.pix_fmt = "yuv420p"
        if codec == "libx264":
            stream.options = {"crf": "20", "preset": "medium"}

        for frame_index in range(end - start):
            panels = []
            for key, label in PANEL_SPECS:
                value = sequences[key][frame_index]
                if key == "tactile_force_field_right":
                    image = render_force_field(value)
                elif key == "tactile_depth_right":
                    image = render_depth(value, depth_scale)
                else:
                    image = to_uint8_rgb(value)

                if key == "front":
                    elapsed = frame_index / args.fps
                    label = (
                        f"{label} | episode {episode} | "
                        f"{elapsed:04.1f}s / {(end - start) / args.fps:04.1f}s"
                    )
                panels.append(add_label(letterbox(image, panel_size), label))

            top = np.concatenate(panels[:4], axis=1)
            bottom = np.concatenate(panels[4:], axis=1)
            grid = np.concatenate((top, bottom), axis=0)
            frame = av.VideoFrame.from_ndarray(grid, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    print(
        f"Rendered episode {episode} ({end - start} frames, "
        f"{(end - start) / args.fps:.1f}s) to {args.output}"
    )


if __name__ == "__main__":
    main()
