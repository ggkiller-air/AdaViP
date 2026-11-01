#!/usr/bin/env python3
"""Convert the whole release to UViP windows and verified MP4-backed observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import av
import numpy as np
import pyarrow.parquet as pq

from adavip.openvitac.schema import ACTION_CODEC, HF_REVISION, IMAGE_KEYS, SCHEMA_VERSION, complete_anchors


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_language(texts):
    """Encode unique actual instructions once with the existing frozen CLIP model."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).eval().requires_grad_(False)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        values = torch.nn.functional.normalize(model.get_text_features(**inputs), dim=-1).numpy()
    return values.astype(np.float32), model_name


def convert(raw_root, output, horizon=16, history=2, val_ratio=0.1, verify_videos=True):
    """Keep source episodes intact, derive explicitly labelled targets, and publish atomically."""
    raw_root, output = Path(raw_root).resolve(), Path(output).resolve()
    if (output / "COMPLETE.json").exists():
        raise FileExistsError("Converted dataset already complete; use a new destination")
    download = json.loads((raw_root / "COMPLETE.json").read_text())
    if download["revision"] != HF_REVISION or download["files"] != 6599 or download.get("failures"):
        raise ValueError("Missing complete, pinned, hash-verified source")
    output.mkdir(parents=True, exist_ok=True)
    roots = sorted((raw_root / "Simulation/GelSight_mini").iterdir())
    if len(roots) != 11:
        raise ValueError("Expected all eleven tasks")
    texts = sorted({row["task"] for root in roots for row in json_lines(root / "meta/tasks.jsonl")})
    embeddings, model_name = frozen_language(texts)
    text_ids = {text: index for index, text in enumerate(texts)}
    np.savez(output / "language.npz", texts=np.array(texts), embeddings=embeddings,
             backend=np.array("frozen_transformer"), model=np.array(model_name))
    manifest = {"schema": SCHEMA_VERSION, "source_revision": HF_REVISION,
                "source_inventory_sha256": file_hash(raw_root / "source_inventory.json"),
                "raw_root": str(raw_root), "action_codec": ACTION_CODEC,
                "horizon": horizon, "n_obs_steps": history, "nominal_hz": 60,
                "language_model": model_name, "language_texts": texts,
                "split": {"seed": 42, "validation_ratio": val_ratio, "unit": "source_episode",
                          "stratify": "source_variant (not exposed as an input)"},
                "image_storage": "immutable source MP4 references; RGB decoded without channel swap",
                "tasks": [], "files": {}}
    totals = {"episodes": 0, "frames": 0, "complete_windows": 0, "source_gap_edges": 0,
              "videos_header_verified": 0, "valid_derived_actions": 0}
    started = time.monotonic()
    for task_index, root in enumerate(roots):
        info = json.loads((root / "meta/info.json").read_text())
        metadata = json.loads((root / "meta/dataset_metadata.json").read_text())
        rows = json_lines(root / "meta/conversion_manifest.jsonl")
        if [r["episode_index"] for r in rows] != list(range(info["total_episodes"])):
            raise ValueError("Conversion manifest episode ordering changed")
        task_text = {r["task_index"]: r["task"] for r in json_lines(root / "meta/tasks.jsonl")}
        if "action" in info["features"] or metadata["action"] is not None or info["fps"] != 60:
            raise ValueError("Source action/timing contract changed")
        strata = {}
        for row in rows:
            strata.setdefault(row["source_task"], []).append(row["episode_index"])
        rng = np.random.default_rng(42 + task_index)
        held = set()
        for ids in strata.values():
            count = min(len(ids) - 1, max(1, round(len(ids) * val_ratio))) if val_ratio else 0
            held.update(rng.permutation(ids)[:count].tolist())
        arrays = {key: [] for key in ("state", "action", "action_valid", "language_id", "timestamp",
                                     "source_time", "eef_pose", "finger_position")}
        episodes, windows = [], []
        offset = 0
        for row in rows:
            episode = row["episode_index"]
            chunk = episode // info["chunks_size"]
            stem = "episode_%06d" % episode
            table = pq.read_table(root / ("data/chunk-%03d" % chunk) / (stem + ".parquet"))
            if any("action" in key for key in table.column_names):
                raise ValueError("Unexpected action column; do not silently replace recorded commands")
            data = table.to_pydict()
            state = np.array(data["observation.state"], dtype=np.float32)
            joint = np.array(data["observation.joint_position"], dtype=np.float32)
            fingers = np.array(data["observation.finger_position"], dtype=np.float32)
            pose = np.array(data["observation.eef_pose_quaternion"], dtype=np.float32)
            timestamp = np.array(data["timestamp"], dtype=np.float32)
            length = len(state)
            if length != row["frame_count"] or state.shape != (length, 8) or fingers.shape != (length, 2):
                raise ValueError("Source episode shape mismatch")
            np.testing.assert_array_equal(state[:, :7], joint)
            np.testing.assert_allclose(state[:, 7], np.clip(fingers.sum(axis=1) / .08, 0, 1), atol=2e-6)
            np.testing.assert_allclose(np.linalg.norm(pose[:, 3:], axis=1), 1, atol=2e-5)
            np.testing.assert_array_equal(data["frame_index"], np.arange(length))
            np.testing.assert_array_equal(data["episode_index"], np.full(length, episode))
            np.testing.assert_array_equal(data["index"], offset + np.arange(length))
            np.testing.assert_allclose(timestamp, np.arange(length) / 60, atol=2e-6)
            local_language = np.array([text_ids[task_text[int(i)]] for i in data["task_index"]], dtype=np.int64)
            if len(set(local_language.tolist())) != 1 or texts[local_language[0]] != row["instruction"]:
                raise ValueError("Episode instruction disagrees with task_index/manifest")
            source_delta = np.full(length - 1, 2, dtype=np.int64)
            for gap in row["source_step_gaps"]:
                source_delta[gap["after_frame_index"]] = gap["source_step_delta"]
            if np.any(source_delta <= 0):
                raise ValueError("Non-monotonic source simulation time")
            source_time = np.r_[0., np.cumsum(source_delta) / 120.]
            action = np.zeros((length, 8), dtype=np.float32)
            action[:-1, :7] = joint[1:]
            action[:-1, 7] = fingers[1:, 0]
            valid = np.r_[source_delta == 2, False]
            action[~valid] = 0
            for value in (state, action, fingers, pose):
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite numeric source")
            videos = {}
            for key in IMAGE_KEYS:
                feature = "observation.images." + key
                path = root / ("videos/chunk-%03d" % chunk) / feature / (stem + ".mp4")
                if not path.is_file():
                    raise FileNotFoundError(path)
                if verify_videos:
                    with av.open(str(path)) as container:
                        stream = container.streams.video[0]
                        expected_shape = info["features"][feature]["shape"]
                        if (stream.height, stream.width) != tuple(expected_shape[:2]) or float(stream.average_rate) != 60 or stream.frames != length:
                            raise ValueError("Video header/frame alignment mismatch: " + str(path))
                    totals["videos_header_verified"] += 1
                videos[key] = path.relative_to(raw_root).as_posix()
            episodes.append({"episode_index": episode, "start": offset, "stop": offset + length,
                             "language_id": int(local_language[0]), "split": "validation" if episode in held else "train",
                             "source_variant": row["source_task"], "source_step_gaps": row["source_step_gaps"],
                             "videos": videos})
            for t, seg_start, seg_stop in complete_anchors(length, row["source_step_gaps"], horizon, history):
                windows.append((episode, offset + t, offset + seg_start, offset + seg_stop))
            for key, value in {"state": state, "action": action, "action_valid": valid,
                               "language_id": local_language, "timestamp": timestamp, "source_time": source_time,
                               "eef_pose": pose, "finger_position": fingers}.items():
                arrays[key].append(value)
            offset += length
            totals["source_gap_edges"] += len(row["source_step_gaps"])
            totals["valid_derived_actions"] += int(valid.sum())
        if offset != info["total_frames"] or len(episodes) != info["total_episodes"]:
            raise ValueError("Episode metadata totals disagree")
        task_dir = output / root.name
        task_dir.mkdir(exist_ok=True)
        for key, values in arrays.items():
            np.save(task_dir / (key + ".npy"), np.concatenate(values), allow_pickle=False)
        np.save(task_dir / "episode_ends.npy", np.array([e["stop"] for e in episodes], dtype=np.int64), allow_pickle=False)
        np.save(task_dir / "windows.npy", np.array(windows, dtype=np.int64), allow_pickle=False)
        (task_dir / "episodes.json").write_text(json.dumps(episodes, indent=2))
        manifest["tasks"].append({"task_id": root.name, "task_index": task_index, "frames": offset,
                                   "episodes": len(episodes), "windows": len(windows), "directory": root.name})
        totals["episodes"] += len(episodes)
        totals["frames"] += offset
        totals["complete_windows"] += len(windows)
        print(json.dumps({"task": root.name, **manifest["tasks"][-1], "seconds": round(time.monotonic() - started, 1)}), flush=True)
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in ("manifest.json", "COMPLETE.json"):
            manifest["files"][path.relative_to(output).as_posix()] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    manifest["totals"] = totals
    content = json.dumps(manifest, indent=2, allow_nan=False)
    (output / "manifest.json").write_text(content)
    marker = {"status": "complete", "manifest_sha256": hashlib.sha256(content.encode()).hexdigest(), **totals}
    (output / "COMPLETE.json.partial").write_text(json.dumps(marker, indent=2))
    (output / "COMPLETE.json.partial").replace(output / "COMPLETE.json")
    print(json.dumps(marker), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.raw_root, args.output)
