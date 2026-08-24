"""Offline GelSight marker tracking and task-local PCA fitting."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MarkerDetectorConfig:
    """Parameters for the fixed-view Piper GelSight marker detector."""

    marker_count: int = 63
    grid_rows: int = 7
    grid_columns: int = 9
    thresholds: tuple[int, ...] = (55, 50, 45, 40)
    min_component_area: int = 35
    max_component_area: int = 300
    roi_yx: tuple[int, int, int, int] = (35, 215, 40, 280)
    max_assignment_distance_px: float = 12.0

    def __post_init__(self) -> None:
        if self.grid_rows * self.grid_columns != self.marker_count:
            raise ValueError("grid_rows * grid_columns must equal marker_count")


@dataclass(frozen=True)
class DetectionResult:
    """Detected marker centers and the threshold that produced them."""

    centers_xy: np.ndarray
    threshold: int
    component_count: int
    success: bool


def detect_markers(
    image: np.ndarray, config: MarkerDetectorConfig = MarkerDetectorConfig()
) -> DetectionResult:
    """Detect dark circular markers inside the configured sensor ROI."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, received shape {image.shape}")
    gray = image.astype(np.float32).mean(axis=-1)
    y_min, y_max, x_min, x_max = config.roi_yx
    best_centers = np.empty((0, 2), dtype=np.float32)
    best_threshold = config.thresholds[0]
    best_error = float("inf")

    for threshold in config.thresholds:
        mask = gray < threshold
        labels, _ = ndimage.label(mask)
        areas = np.bincount(labels.ravel())[1:]
        component_ids = np.flatnonzero(
            (areas >= config.min_component_area)
            & (areas <= config.max_component_area)
        ) + 1
        if component_ids.size:
            centers_yx = np.asarray(
                ndimage.center_of_mass(mask, labels, component_ids), dtype=np.float32
            )
            in_roi = (
                (centers_yx[:, 0] >= y_min)
                & (centers_yx[:, 0] < y_max)
                & (centers_yx[:, 1] >= x_min)
                & (centers_yx[:, 1] < x_max)
            )
            centers_xy = centers_yx[in_roi][:, ::-1].copy()
        else:
            centers_xy = np.empty((0, 2), dtype=np.float32)
        error = abs(len(centers_xy) - config.marker_count)
        if error < best_error:
            best_centers = centers_xy
            best_threshold = threshold
            best_error = error
        if len(centers_xy) == config.marker_count:
            return DetectionResult(
                centers_xy=centers_xy,
                threshold=threshold,
                component_count=len(centers_xy),
                success=True,
            )

    return DetectionResult(
        centers_xy=best_centers,
        threshold=best_threshold,
        component_count=len(best_centers),
        success=False,
    )


def order_reference_grid(
    centers_xy: np.ndarray, config: MarkerDetectorConfig = MarkerDetectorConfig()
) -> np.ndarray:
    """Order a complete marker grid from top-left to bottom-right."""
    centers_xy = np.asarray(centers_xy, dtype=np.float32)
    if centers_xy.shape != (config.marker_count, 2):
        raise ValueError(
            f"Expected {(config.marker_count, 2)} centers, got {centers_xy.shape}"
        )
    y_order = np.argsort(centers_xy[:, 1], kind="stable")
    rows = centers_xy[y_order].reshape(config.grid_rows, config.grid_columns, 2)
    ordered_rows = [row[np.argsort(row[:, 0], kind="stable")] for row in rows]
    return np.concatenate(ordered_rows, axis=0)


def assign_to_reference(
    reference_xy: np.ndarray,
    detected_xy: np.ndarray,
    max_distance_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign an unordered detection set to fixed reference marker identities."""
    reference_xy = np.asarray(reference_xy, dtype=np.float32)
    detected_xy = np.asarray(detected_xy, dtype=np.float32)
    if detected_xy.shape != reference_xy.shape:
        raise ValueError(
            f"Detection shape {detected_xy.shape} does not match {reference_xy.shape}"
        )
    distances = np.linalg.norm(
        reference_xy[:, None, :] - detected_xy[None, :, :], axis=-1
    )
    reference_indices, detection_indices = linear_sum_assignment(distances)
    tracked = np.empty_like(reference_xy)
    tracked[reference_indices] = detected_xy[detection_indices]
    assignment_distance = distances[reference_indices, detection_indices]
    if float(assignment_distance.max()) > max_distance_px:
        raise RuntimeError(
            "Marker assignment exceeded the configured distance: "
            f"{assignment_distance.max():.3f} > {max_distance_px:.3f} px"
        )
    return tracked, assignment_distance.astype(np.float32)


def process_episodes(
    images: Any,
    episode_ends: Iterable[int],
    config: MarkerDetectorConfig = MarkerDetectorConfig(),
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Track markers with each episode's first image as its reference frame."""
    episode_ends_array = np.asarray(list(episode_ends), dtype=np.int64)
    if episode_ends_array.ndim != 1 or len(episode_ends_array) == 0:
        raise ValueError("episode_ends must be a non-empty one-dimensional sequence")
    frame_count = int(episode_ends_array[-1])
    if len(images) != frame_count:
        raise ValueError(f"Expected {frame_count} images, received {len(images)}")

    initial_markers = np.empty((frame_count, config.marker_count, 2), np.float32)
    offsets = np.empty_like(initial_markers)
    frame_audit: list[dict[str, Any]] = []
    episode_start = 0
    for episode_index, episode_end in enumerate(episode_ends_array):
        reference_detection = detect_markers(images[episode_start], config)
        if not reference_detection.success:
            raise RuntimeError(
                f"Episode {episode_index} reference frame detected "
                f"{reference_detection.component_count}/{config.marker_count} markers"
            )
        reference_xy = order_reference_grid(reference_detection.centers_xy, config)
        height, width = images[episode_start].shape[:2]
        scale_xy = np.asarray([width, height], dtype=np.float32)

        for frame_index in range(episode_start, int(episode_end)):
            detection = detect_markers(images[frame_index], config)
            if not detection.success:
                raise RuntimeError(
                    f"Frame {frame_index} detected "
                    f"{detection.component_count}/{config.marker_count} markers"
                )
            tracked_xy, assignment_distance = assign_to_reference(
                reference_xy,
                detection.centers_xy,
                config.max_assignment_distance_px,
            )
            offset_px = tracked_xy - reference_xy
            initial_markers[frame_index] = reference_xy / scale_xy
            offsets[frame_index] = offset_px / scale_xy
            frame_audit.append(
                {
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "episode_frame_index": frame_index - episode_start,
                    "detected_markers": detection.component_count,
                    "detection_success": detection.success,
                    "threshold": detection.threshold,
                    "assignment_mean_px": float(assignment_distance.mean()),
                    "assignment_max_px": float(assignment_distance.max()),
                    "contact_rms_px": float(
                        np.sqrt(np.mean(np.sum(offset_px**2, axis=-1)))
                    ),
                    "contact_max_px": float(np.linalg.norm(offset_px, axis=-1).max()),
                }
            )
        episode_start = int(episode_end)
    return initial_markers, offsets, frame_audit


def fit_task_pca(
    marker_offsets: np.ndarray, n_components: int = 15
) -> tuple[np.ndarray, Any]:
    """Fit PCA to flattened normalized marker offsets and transform all frames."""
    from sklearn.decomposition import PCA

    offsets = np.asarray(marker_offsets, dtype=np.float32)
    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError(f"Expected [T, markers, 2] offsets, got {offsets.shape}")
    flattened = offsets.reshape(len(offsets), -1)
    if n_components > min(flattened.shape):
        raise ValueError(f"Cannot fit {n_components} components to {flattened.shape}")
    pca = PCA(n_components=n_components, svd_solver="full")
    embedding = pca.fit_transform(flattened).astype(np.float32)
    return embedding, pca


def build_audit_report(
    frame_audit: list[dict[str, Any]],
    marker_offsets: np.ndarray,
    embedding: np.ndarray,
    pca: Any,
    episode_ends: Iterable[int],
    config: MarkerDetectorConfig,
) -> dict[str, Any]:
    """Build a JSON-serializable detection, contact, and PCA audit summary."""
    flattened = np.asarray(marker_offsets).reshape(len(marker_offsets), -1)
    reconstructed = pca.inverse_transform(embedding)
    reconstruction_error = flattened - reconstructed
    ends = np.asarray(list(episode_ends), dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    episode_contact = []
    for episode_index, (start, end) in enumerate(zip(starts, ends)):
        curve = [row["contact_rms_px"] for row in frame_audit[int(start) : int(end)]]
        episode_contact.append(
            {
                "episode_index": episode_index,
                "start_frame": int(start),
                "end_frame_exclusive": int(end),
                "mean_rms_px": float(np.mean(curve)),
                "peak_rms_px": float(np.max(curve)),
                "peak_episode_frame": int(np.argmax(curve)),
            }
        )
    successes = sum(bool(row["detection_success"]) for row in frame_audit)
    explained = pca.explained_variance_ratio_.astype(float)
    return {
        "format": "adavip.gelsight_marker_audit.v1",
        "detector": asdict(config),
        "frames": len(frame_audit),
        "episodes": len(ends),
        "detection": {
            "successful_frames": successes,
            "failed_frames": len(frame_audit) - successes,
            "success_rate": successes / len(frame_audit),
            "minimum_count": min(row["detected_markers"] for row in frame_audit),
            "maximum_count": max(row["detected_markers"] for row in frame_audit),
            "assignment_mean_px": float(
                np.mean([row["assignment_mean_px"] for row in frame_audit])
            ),
            "assignment_max_px": float(
                np.max([row["assignment_max_px"] for row in frame_audit])
            ),
        },
        "pca": {
            "input_dimensions": int(flattened.shape[1]),
            "components": int(pca.n_components_),
            "explained_variance_ratio": explained.tolist(),
            "cumulative_explained_variance_ratio": np.cumsum(explained).tolist(),
            "total_explained_variance_ratio": float(explained.sum()),
            "reconstruction_rmse_normalized": float(
                np.sqrt(np.mean(reconstruction_error**2))
            ),
            "reconstruction_max_abs_normalized": float(
                np.max(np.abs(reconstruction_error))
            ),
        },
        "contact": {
            "metric": "RMS marker displacement in pixels relative to episode frame 0",
            "episodes": episode_contact,
        },
    }


def write_audit_files(
    output_dir: Path,
    frame_audit: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    """Write machine-readable audit data and a contact curve plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "gelsight_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "gelsight_frame_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(frame_audit[0]))
        writer.writeheader()
        writer.writerows(frame_audit)

    from PIL import Image, ImageDraw

    width, height = 1600, 520
    margins = (90, 40, 30, 70)
    plot_left, plot_top = margins[0], margins[1]
    plot_right, plot_bottom = width - margins[2], height - margins[3]
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    values = np.asarray([row["contact_rms_px"] for row in frame_audit])
    y_max = max(float(values.max()) * 1.05, 1e-6)

    def point(frame: int, value: float) -> tuple[float, float]:
        x = plot_left + frame * (plot_right - plot_left) / max(len(values) - 1, 1)
        y = plot_bottom - value * (plot_bottom - plot_top) / y_max
        return x, y

    for grid_index in range(6):
        y_value = y_max * grid_index / 5
        y = point(0, y_value)[1]
        draw.line((plot_left, y, plot_right, y), fill=(220, 220, 220), width=1)
        draw.text((8, y - 7), f"{y_value:.2f} px", fill="black")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
    colors = ((0, 110, 170), (210, 80, 30), (20, 140, 80), (130, 70, 160))
    for episode in report["contact"]["episodes"]:
        episode_index = episode["episode_index"]
        start = episode["start_frame"]
        end = episode["end_frame_exclusive"]
        curve = [point(i, values[i]) for i in range(start, end)]
        if len(curve) > 1:
            draw.line(curve, fill=colors[episode_index % len(colors)], width=3)
        if start:
            x = point(start, 0)[0]
            draw.line((x, plot_top, x, plot_bottom), fill=(80, 80, 80), width=1)
        draw.text(
            (point(start, y_max)[0] + 5, plot_top + 5),
            f"episode {episode_index}",
            fill=colors[episode_index % len(colors)],
        )
    draw.text((plot_left, 10), "GelSight contact curve", fill="black")
    draw.text(
        ((plot_left + plot_right) // 2 - 50, height - 35),
        "Dataset frame",
        fill="black",
    )
    canvas.save(output_dir / "gelsight_contact_curve.png")
