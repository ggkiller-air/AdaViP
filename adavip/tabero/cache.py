"""Decode LeRobot parquet once, preserving native RGB and all marker histories."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from .schema import CACHE_VERSION, DATA_REVISION, IMAGE_KEYS, fingerprint


def decode_episode(item: tuple) -> dict:
    """Write one disjoint episode; publish completion only after every array flush."""
    from PIL import Image
    import pyarrow.parquet as pq

    raw_path, output_path, episode, tasks = item
    raw, output = Path(raw_path), Path(output_path)
    done = output / "episodes" / f"{episode['episode_index']:06d}.json"
    source_digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if done.exists():
        result = json.loads(done.read_text())
        if result["source_sha256"] != source_digest:
            raise ValueError(f"Changed source episode {raw}")
        return result
    columns = [*IMAGE_KEYS, "state", "actions", "tactile_marker_motion", "task_index", "frame_index", "episode_index"]
    rows = pq.read_table(raw, columns=columns).to_pydict()
    count = episode["end"] - episode["start"]
    if len(rows["state"]) != count or rows["frame_index"] != list(range(count)):
        raise ValueError(f"Noncontiguous episode {raw}")
    if set(rows["episode_index"]) != {episode["episode_index"]} or len(set(rows["task_index"])) != 1:
        raise ValueError(f"Unexpected episode/task identity {raw}")
    task_index = rows["task_index"][0]
    if tasks[task_index] not in episode["tasks"]:
        raise ValueError("Parquet task does not match episode metadata")
    section = slice(episode["start"], episode["end"])
    for target, source, shape in (("state", "state", (7,)), ("action", "actions", (13,)),
                                  ("marker", "tactile_marker_motion", (9, 198, 2))):
        value = np.asarray(rows[source], dtype=np.float32)
        if value.shape != (count, *shape) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {source} in {raw}")
        array = np.load(output / f"{target}.npy", mmap_mode="r+")
        array[section] = value
        array.flush()
        del array
    for key in IMAGE_KEYS:
        array = np.load(output / f"{key}.npy", mmap_mode="r+")
        for index, encoded in enumerate(rows[key]):
            with Image.open(io.BytesIO(encoded["bytes"])) as image:
                value = np.asarray(image.convert("RGB"))
            if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                raise ValueError(f"Invalid image shape in {raw}")
            array[episode["start"] + index] = value.transpose(2, 0, 1)
        array.flush()
        del array
    result = {**episode, "task_index": task_index, "source_sha256": source_digest}
    temporary = done.with_suffix(".partial")
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n")
    os.replace(temporary, done)
    return result


def build_cache(source: Path, output: Path, workers: int) -> None:
    """Resume incomplete episodes and publish a manifest after CLIP and data validation."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    if (output / "COMPLETE.json").exists():
        print("Cache already complete", flush=True)
        return
    info = json.loads((source / "meta/info.json").read_text())
    task_rows = [json.loads(line) for line in (source / "meta/tasks.jsonl").read_text().splitlines()]
    task_rows.sort(key=lambda row: row["task_index"])
    if [row["task_index"] for row in task_rows] != list(range(info["total_tasks"])):
        raise ValueError("Task indices must be contiguous")
    tasks = [row["task"] for row in task_rows]
    episodes, cursor = [], 0
    for line in (source / "meta/episodes.jsonl").read_text().splitlines():
        row = json.loads(line)
        episodes.append({**row, "start": cursor, "end": cursor + row["length"]})
        cursor += row["length"]
    if cursor != info["total_frames"] or len(episodes) != info["total_episodes"]:
        raise ValueError("Episode counts disagree with metadata")
    spec = {"version": CACHE_VERSION, "source_revision": DATA_REVISION,
            "source": str(source.resolve()), "total_steps": cursor, "tasks": tasks,
            "episodes": episodes, "image_size": 224, "color": "RGB",
            "action_storage": "absolute_eef_axis_angle_gripper_force13",
            "action_training": "delta_first6_from_current_observation_anchor;absolute_gripper_force",
            "marker": "reference+8_native_frames;no_subtraction;bilateral198x2",
            "language_model": "openai/clip-vit-base-patch32", "language_l2_normalized": True}
    spec_path = output / "build_spec.json"
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError("Existing partial cache belongs to a different source/protocol")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    (output / "episodes").mkdir(exist_ok=True)
    shapes = {**{key: (cursor, 3, 224, 224) for key in IMAGE_KEYS},
              "state": (cursor, 7), "action": (cursor, 13), "marker": (cursor, 9, 198, 2)}
    # Allocate all arrays before workers write disjoint regions.
    for key, shape in shapes.items():
        path = output / f"{key}.npy"
        dtype = np.uint8 if key in IMAGE_KEYS else np.float32
        if path.exists():
            existing = np.load(path, mmap_mode="r")
            if existing.shape != shape or existing.dtype != dtype:
                raise ValueError(f"Partial array incompatible: {key}")
            del existing
        else:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            array.flush()
            del array
    language_path = output / "language.npy"
    if not language_path.exists():
        torch.set_num_threads(min(4, workers))
        model_name = spec["language_model"]
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True).eval().requires_grad_(False)
        encoded = tokenizer(tasks, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            language = torch.nn.functional.normalize(model.get_text_features(**encoded), dim=-1).numpy()
        np.save(language_path, language.astype(np.float32))
        del model
    jobs = [(str(source / info["data_path"].format(episode_chunk=row["episode_index"] // info["chunks_size"],
                                                 episode_index=row["episode_index"])), str(output), row, tasks)
            for row in episodes]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = []
        for i, result in enumerate(pool.map(decode_episode, jobs), 1):
            results.append(result)
            if i % 20 == 0 or i == len(jobs):
                print(json.dumps({"episodes_done": i, "total": len(jobs)}), flush=True)
    spec["episodes"] = results
    spec["cache_id"] = fingerprint(spec)
    (output / "manifest.json").write_text(json.dumps(spec, indent=2) + "\n")
    (output / "COMPLETE.json").write_text(json.dumps({"cache_id": spec["cache_id"], "frames": cursor,
                                                     "episodes": len(results)}) + "\n")


def main() -> None:
    """Run under one external per-cache flock to avoid concurrent builders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    build_cache(args.source, args.output, args.workers)


if __name__ == "__main__":
    main()
