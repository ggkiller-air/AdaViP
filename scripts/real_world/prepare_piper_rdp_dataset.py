#!/usr/bin/env python3
"""Create an immutable-source Piper RDP dataset with GelSight PCA features."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adavip.real_world.gelsight_marker_processor import (
    MarkerDetectorConfig,
    build_audit_report,
    fit_task_pca,
    process_episodes,
    write_audit_files,
)


DEFAULT_SOURCE = REPO_ROOT / "pick_and_place"
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "piper_rdp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_dataset(source: Path, output: Path, overwrite: bool = False) -> dict:
    """Copy the source replay buffer, derive tactile arrays, and write audits."""
    source = source.resolve()
    output = output.resolve()
    if (
        source == output
        or source.is_relative_to(output)
        or output.is_relative_to(source)
    ):
        raise ValueError("Source and output paths must not contain one another")
    source_zarr = source / "replay_buffer.zarr"
    if not source_zarr.is_dir():
        raise FileNotFoundError(f"Source replay buffer is unavailable: {source_zarr}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    temporary_output = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary_output.exists():
        shutil.rmtree(temporary_output)
    try:
        temporary_output.mkdir(parents=True)
        shutil.copytree(source_zarr, temporary_output / "replay_buffer.zarr")
        root = zarr.open(str(temporary_output / "replay_buffer.zarr"), mode="a")
        images = root["data/right_gelsight_img"]
        episode_ends = root["meta/episode_ends"][:]
        config = MarkerDetectorConfig()
        initial_markers, offsets, frame_audit = process_episodes(
            images, episode_ends, config
        )
        embedding, pca = fit_task_pca(offsets, n_components=15)

        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        data = root["data"]
        arrays = {
            "right_gelsight_initial_marker": initial_markers,
            "right_gelsight_marker_offset": offsets,
            "right_gelsight_marker_offset_emb": embedding,
        }
        for key, values in arrays.items():
            if key in data:
                del data[key]
            data.array(
                key,
                values,
                chunks=(min(256, len(values)),) + values.shape[1:],
                compressor=compressor,
            )

        artifacts_dir = temporary_output / "artifacts" / "gelsight_pca"
        artifacts_dir.mkdir(parents=True)
        np.save(
            artifacts_dir / "pca_transform_matrix.npy",
            pca.components_.T.astype(np.float32),
        )
        np.save(
            artifacts_dir / "pca_mean_matrix.npy", pca.mean_.astype(np.float32)
        )
        report = build_audit_report(
            frame_audit, offsets, embedding, pca, episode_ends, config
        )
        write_audit_files(temporary_output / "audit", frame_audit, report)
        manifest = {
            "format": "adavip.piper_rdp_dataset.v1",
            "source_dataset": "ggkiller-air/pick_and_place",
            "source_revision": "cd51cf4c55f45115b984fd469e102b9152dede61",
            "source_path": str(source),
            "control_frequency_hz": 18.75,
            "action": {
                "key": "action",
                "shape": [7],
                "semantics": "absolute Piper joint_1..joint_6 and gripper target",
            },
            "observations": {
                "slow": [
                    "external_img",
                    "right_wrist_img",
                    "right_robot_qpos",
                    "right_gelsight_marker_offset_emb",
                ],
                "extended": ["right_gelsight_marker_offset_emb"],
            },
            "tactile_processing": {
                "reference": "first frame of each episode",
                "marker_count": 63,
                "offset_key": "right_gelsight_marker_offset",
                "offset_shape": [63, 2],
                "offset_units": "x/image_width, y/image_height",
                "embedding_key": "right_gelsight_marker_offset_emb",
                "embedding_shape": [15],
                "transform": "(offset.reshape(126) - mean) @ transform_matrix",
                "transform_matrix": "artifacts/gelsight_pca/pca_transform_matrix.npy",
                "mean": "artifacts/gelsight_pca/pca_mean_matrix.npy",
            },
            "audit": "audit/gelsight_audit.json",
        }
        with (temporary_output / "dataset_manifest.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        os.replace(temporary_output, output)
        return report
    except BaseException:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        raise


def main() -> int:
    args = parse_args()
    report = prepare_dataset(args.source.resolve(), args.output.resolve(), args.overwrite)
    print(f"output={args.output.resolve()}")
    print(
        "detection_success_rate="
        f"{report['detection']['success_rate']:.6f} "
        f"({report['detection']['successful_frames']}/{report['frames']})"
    )
    print(
        "pca_explained_variance="
        f"{report['pca']['total_explained_variance_ratio']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
