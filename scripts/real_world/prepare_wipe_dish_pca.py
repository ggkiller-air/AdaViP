#!/usr/bin/env python3
"""Prepare bilateral GelSight marker/PCA features for wipe-dish DP runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import zarr
from numcodecs import Blosc

from adavip.real_world.gelsight_marker_processor import (
    MarkerDetectorConfig,
    build_audit_report,
    fit_task_pca,
    process_episodes,
    write_audit_files,
)


DEFAULT_SOURCE = Path("/data/wangzihao/wipe-dish-rdp-both")
DEFAULT_OUTPUT = Path("/data/wangzihao/datasets/real_world/wipe_dish_pca")
VALIDATION_EPISODES = (3, 9, 17, 25, 29, 30, 36, 40, 42)
WIPE_DISH_MAX_ASSIGNMENT_DISTANCE_PX = 24.0


def frame_mask(episode_ends: Iterable[int], validation: set[int]) -> np.ndarray:
    """Return a frame mask for training episodes only."""
    ends = np.asarray(list(episode_ends), dtype=np.int64)
    mask = np.zeros(int(ends[-1]), dtype=bool)
    start = 0
    for episode_index, end in enumerate(ends):
        if episode_index not in validation:
            mask[start : int(end)] = True
        start = int(end)
    return mask


def _copy_tree_with_hardlinks(source: Path, target: Path) -> None:
    """Copy a Zarr directory without duplicating immutable source chunks."""
    try:
        shutil.copytree(source, target, copy_function=os.link)
    except OSError:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def _fit_and_transform(offsets: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, object]:
    """Fit PCA on training frames and transform all frames."""
    _, pca = fit_task_pca(offsets[train_mask], n_components=15)
    embedding = pca.transform(offsets.reshape(len(offsets), -1)).astype(np.float32)
    return embedding, pca


def prepare_dataset(
    source: Path,
    output: Path,
    validation_episodes: Iterable[int] = VALIDATION_EPISODES,
    overwrite: bool = False,
) -> dict:
    """Create a hard-linked prepared dataset with bilateral PCA arrays."""
    source = source.resolve()
    output = output.resolve()
    source_zarr = source / "replay_buffer.zarr"
    if not source_zarr.is_dir():
        raise FileNotFoundError(f"Source replay buffer is unavailable: {source_zarr}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)

    validation = set(int(index) for index in validation_episodes)
    try:
        temporary.mkdir(parents=True)
        _copy_tree_with_hardlinks(source_zarr, temporary / "replay_buffer.zarr")
        root = zarr.open(str(temporary / "replay_buffer.zarr"), mode="a")
        ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        train_mask = frame_mask(ends, validation)
        data = root["data"]
        audits: dict[str, dict] = {}
        # Wipe-dish demonstrations have larger marker motion than the
        # pick-and-place default; retain the audit so this wider gate is
        # observable and does not silently accept failed detections.
        detector = MarkerDetectorConfig(
            max_assignment_distance_px=WIPE_DISH_MAX_ASSIGNMENT_DISTANCE_PX
        )
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        pca_artifacts = temporary / "artifacts" / "gelsight_pca"
        pca_artifacts.mkdir(parents=True)

        for side in ("left", "right"):
            key = f"gelsight_{side}"
            initial, offsets, frame_audit = process_episodes(
                root[f"data/{key}"], ends, detector
            )
            embedding, pca = _fit_and_transform(offsets, train_mask)
            for array_name, values in (
                (f"gelsight_{side}_initial_marker", initial),
                (f"gelsight_{side}_marker_offset", offsets),
                (f"gelsight_{side}_marker_offset_emb", embedding),
            ):
                if array_name in data:
                    del data[array_name]
                data.array(
                    array_name,
                    values,
                    chunks=(min(256, len(values)),) + values.shape[1:],
                    compressor=compressor,
                )

            np.save(pca_artifacts / f"{side}_transform_matrix.npy", pca.components_.T.astype(np.float32))
            np.save(pca_artifacts / f"{side}_mean_matrix.npy", pca.mean_.astype(np.float32))
            report = build_audit_report(
                frame_audit, offsets, embedding, pca, ends, detector
            )
            report["split"] = {
                "seed": 42,
                "validation_episode_indices": sorted(validation),
                "fit_frames": int(train_mask.sum()),
            }
            write_audit_files(temporary / "audit" / f"gelsight_{side}", frame_audit, report)
            audits[side] = report

        manifest = {
            "format": "adavip.wipe_dish_dp_pca.v1",
            "source_path": str(source),
            "source_format": "piper_both_rdp_v2",
            "control_frequency_hz": 30.0,
            "validation_episode_indices": sorted(validation),
            "training_episode_count": int(len(ends) - len(validation)),
            "action": {
                "key": "action",
                "shape": [14],
                "semantics": "absolute bilateral Piper joint and gripper targets",
            },
            "observations": {
                "visual": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                "proprioception": "robot_qpos",
                "tactile_embeddings": {
                    "left": "gelsight_left_marker_offset_emb",
                    "right": "gelsight_right_marker_offset_emb",
                },
            },
            "pca": {
                "components_per_sensor": 15,
                "fit_scope": "training episodes only",
                "marker_count": 63,
                "max_assignment_distance_px": WIPE_DISH_MAX_ASSIGNMENT_DISTANCE_PX,
                "artifacts": {
                    "left_transform": "artifacts/gelsight_pca/left_transform_matrix.npy",
                    "left_mean": "artifacts/gelsight_pca/left_mean_matrix.npy",
                    "right_transform": "artifacts/gelsight_pca/right_transform_matrix.npy",
                    "right_mean": "artifacts/gelsight_pca/right_mean_matrix.npy",
                },
            },
            "audits": {
                "left": "audit/gelsight_left/gelsight_audit.json",
                "right": "audit/gelsight_right/gelsight_audit.json",
            },
        }
        with (temporary / "dataset_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if output.exists():
            shutil.rmtree(output) if output.is_dir() else output.unlink()
        os.replace(temporary, output)
        return {"manifest": manifest, "audits": audits}
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_dataset(args.source, args.output, overwrite=args.overwrite)
    print(f"output={args.output.resolve()}")
    for side, report in result["audits"].items():
        print(
            f"{side}_detection_success_rate="
            f"{report['detection']['success_rate']:.6f} "
            f"pca_variance={report['pca']['total_explained_variance_ratio']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
