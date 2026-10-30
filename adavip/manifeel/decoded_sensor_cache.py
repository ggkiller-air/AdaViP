"""Lossless, reusable decoded arrays for immutable ManiFeel trajectory stores."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from filelock import FileLock

from adavip.manifeel.full_sensor_schema import IMAGE_KEYS


CACHE_SCHEMA = "adavip.decoded_sensor_cache.v1"


def fingerprint(value: Any) -> str:
    """Hash a JSON-compatible cache contract deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_digest(values: np.ndarray) -> str:
    """Hash contiguous decoded values without allocating a second array."""
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def cache_contract(dataset: Any) -> dict:
    """Bind a cache to the source revision, selected trajectories and exact shapes."""
    metadata = []
    for store in dataset.manifest["stores"]:
        root = Path(store["path"])
        entries = {}
        for key in (*IMAGE_KEYS, "state", "action"):
            path = root / "data" / key / ".zarray"
            entries[key] = {"metadata": path.read_text(), "mtime_ns": path.stat().st_mtime_ns}
        metadata.append(entries)
    return {"schema": CACHE_SCHEMA, "manifest": dataset.manifest, "source_metadata": metadata}


def selected_rows(dataset: Any, task_index: int) -> np.ndarray:
    """Return sorted original row indices for train plus held-out episodes only."""
    ends = dataset.episode_ends[task_index]
    episodes = np.union1d(dataset.train_episodes[task_index], dataset.val_episodes[task_index])
    return np.concatenate([
        np.arange(0 if episode == 0 else ends[episode - 1], ends[episode], dtype=np.int64)
        for episode in episodes
    ])


class SourceIndexedArray:
    """Expose original trajectory indices over compact, read-only decoded rows."""

    def __init__(self, values: np.ndarray, rows: np.ndarray, source_length: int) -> None:
        self.values = values
        self.values.flags.writeable = False
        self.row_map = np.full(source_length, -1, dtype=np.int64)
        self.row_map[rows] = np.arange(len(rows))

    def __getitem__(self, index: Any) -> np.ndarray:
        mapped = self.row_map[index]
        if np.any(mapped < 0):
            raise IndexError("Decoded cache does not contain an unused trajectory")
        return self.values[mapped]

    def __getstate__(self) -> dict:
        if isinstance(self.values, np.memmap):
            return {"filename": str(self.values.filename), "row_map": self.row_map}
        return self.__dict__

    def __setstate__(self, state: dict) -> None:
        if "filename" in state:
            self.row_map = state["row_map"]
            self.values = np.load(state["filename"], mmap_mode="r", allow_pickle=False)
        else:
            self.__dict__.update(state)
        self.values.flags.writeable = False


def build_decoded_cache(dataset: Any, directory: str, workers: int = 8, block_frames: int = 48) -> dict:
    """Decode selected data once, retaining float32 values and atomic completion markers."""
    if workers < 1 or block_frames < 1:
        raise ValueError("Cache workers and block size must be positive")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    contract = cache_contract(dataset)
    contract_hash = fingerprint(contract)
    started = time.perf_counter()
    with FileLock(str(root / "build.lock")):
        contract_path = root / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Decoded cache contract differs; use a new cache directory")
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
        rows_by_task = [selected_rows(dataset, index) for index in range(len(dataset.groups))]

        def decode(item: tuple[int, str]) -> dict:
            task_index, key = item
            rows = rows_by_task[task_index]
            source = dataset.groups[task_index]["data"][key]
            tail = tuple(source.shape[1:])
            if key in IMAGE_KEYS:
                channels, height, width = dataset.shape_meta["obs"][key]["shape"]
                tail = (height, width, channels)
            shape = (len(rows), *tail)
            filename = f"task{task_index}_{key}.npy"
            path = root / filename
            marker = root / f"{filename}.json"
            if marker.exists() and path.exists():
                record = json.loads(marker.read_text())
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if (record.get("contract") == contract_hash and list(values.shape) == list(shape)
                        and values.dtype == np.float32 and array_digest(values) == record["sha256"]):
                    return record
                raise ValueError(f"Corrupt decoded cache file: {path}")
            partial = root / f"{filename}.partial"
            values = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
            boundaries = np.r_[0, np.flatnonzero(np.diff(rows) != 1) + 1, len(rows)]
            for first, last in zip(boundaries[:-1], boundaries[1:]):
                for offset in range(int(first), int(last), block_frames):
                    stop = min(offset + block_frames, int(last))
                    block = np.asarray(source[int(rows[offset]):int(rows[stop - 1]) + 1], dtype=np.float32)
                    if not np.isfinite(block).all():
                        raise ValueError(f"Nonfinite source values in task{task_index}/{key}")
                    if key in IMAGE_KEYS:
                        if block.min() < 0 or block.max() > 1:
                            raise ValueError(f"{key}: images must be in [0,1]")
                        if block.shape[1:] != tail:
                            block = np.stack([
                                cv2.resize(image, (tail[1], tail[0]), interpolation=cv2.INTER_LINEAR)
                                for image in block
                            ])
                    values[offset:stop] = block
            values.flush()
            record = {
                "task_index": task_index, "key": key, "filename": filename,
                "shape": list(shape), "dtype": "float32", "nbytes": values.nbytes,
                "sha256": array_digest(values), "contract": contract_hash,
            }
            del values
            os.replace(partial, path)
            temporary_marker = marker.with_suffix(".partial")
            temporary_marker.write_text(json.dumps(record, sort_keys=True) + "\n")
            os.replace(temporary_marker, marker)
            print(json.dumps({"event": "decoded_array_ready", **record}), flush=True)
            return record

        items = [(index, key) for index in range(len(dataset.groups)) for key in (*IMAGE_KEYS, "state", "action")]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(decode, items))
        manifest = {"contract": contract_hash, "arrays": records, "nbytes": sum(item["nbytes"] for item in records)}
        temporary = root / "manifest.partial"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, root / "manifest.json")
    print(json.dumps({"event": "decoded_cache_ready", "directory": str(root),
                      "seconds": time.perf_counter() - started, "nbytes": manifest["nbytes"]}), flush=True)
    return manifest


def load_decoded_cache(dataset: Any, directory: str, mode: str = "ram") -> list[dict]:
    """Load verified arrays once; forked workers share read-only RAM without decoding."""
    if mode not in ("ram", "mmap"):
        raise ValueError("Decoded cache mode must be ram or mmap")
    root = Path(directory).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["contract"] != fingerprint(cache_contract(dataset)):
        raise ValueError("Decoded cache contract does not match this dataset")
    expected = {(index, key) for index in range(len(dataset.groups)) for key in (*IMAGE_KEYS, "state", "action")}
    actual = [(item["task_index"], item["key"]) for item in manifest["arrays"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Decoded cache must contain every task and sensor exactly once")
    allocated_mib = os.environ.get("SLURM_MEM_PER_NODE")
    if mode == "ram" and allocated_mib:
        allocated_bytes = int(allocated_mib) * 1024 ** 2
        if manifest["nbytes"] + 32 * 1024 ** 3 > allocated_bytes:
            raise MemoryError("Decoded RAM cache requires its full size plus at least 32 GiB of job memory")
    started = time.perf_counter()
    groups = [{"data": {}} for _ in dataset.groups]
    rows = [selected_rows(dataset, index) for index in range(len(groups))]
    def load_array(record: dict) -> None:
        task_index, key = record["task_index"], record["key"]
        mapped = np.load(root / record["filename"], mmap_mode="r", allow_pickle=False)
        # Match Stage2's sequential mmap-to-RAM transport before hashing. Forked
        # workers retain an owned array and never fault cache files during training.
        values = np.array(mapped, copy=True) if mode == "ram" else mapped
        if (list(values.shape) != record["shape"] or values.shape[0] != len(rows[task_index])
                or values.dtype != np.float32 or array_digest(values) != record["sha256"]):
            raise ValueError(f"Corrupt decoded cache: {record['filename']}")
        groups[task_index]["data"][key] = SourceIndexedArray(values, rows[task_index], int(dataset.episode_ends[task_index][-1]))
        print(json.dumps({"event": "decoded_array_loaded", "key": key, "task_index": task_index,
                          "mode": mode, "nbytes": values.nbytes}), flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(load_array, manifest["arrays"]))
    print(json.dumps({"event": "decoded_cache_loaded", "mode": mode, "nbytes": manifest["nbytes"],
                      "seconds": time.perf_counter() - started}), flush=True)
    return groups
