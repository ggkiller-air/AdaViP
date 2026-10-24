"""Chunk-verified, demand-filled shared RAM for isolated Stage1 benchmarks."""

from __future__ import annotations

import hashlib
import json
import mmap
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray, fingerprint
from adavip.manifeel.stage1_benchmark_data import BenchmarkDataset


BLOCK_SCHEMA = "adavip.stage1_verified_blocks.v1"


def source_identity(path: Path) -> dict:
    """Bind block checksums to an unchanged decoded file."""
    status = path.stat()
    return {"size": status.st_size, "mtime_ns": status.st_mtime_ns}


def build_block_index(directory: Path, output: Path, workers: int = 8,
                      block_bytes: int = 64 * 1024 ** 2) -> dict:
    """Validate full original hashes on CPU and publish reusable block hashes."""
    if workers < 1 or block_bytes < 1:
        raise ValueError("Positive workers and block size are required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite block index: {output}")
    manifest = json.loads((directory / "manifest.json").read_text())
    started = time.perf_counter()

    def hash_array(record: dict) -> dict:
        path = directory / record["filename"]
        identity = source_identity(path)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != record["shape"] or values.dtype != np.float32:
            raise ValueError(f"Cache shape/dtype differs: {path}")
        rows_per_block = max(1, block_bytes // max(1, values[0].nbytes))
        digest = hashlib.sha256()
        hashes = []
        for offset in range(0, len(values), rows_per_block):
            block = np.array(values[offset:offset + rows_per_block], copy=True, order="C")
            data = memoryview(block).cast("B")
            digest.update(data)
            hashes.append(hashlib.sha256(data).hexdigest())
        if digest.hexdigest() != record["sha256"] or source_identity(path) != identity:
            raise ValueError(f"Full cache checksum or identity differs: {path}")
        result = {**record, "source_identity": identity, "rows_per_block": rows_per_block,
                  "block_sha256": hashes}
        print(json.dumps({"event": "block_index_array", "filename": record["filename"],
                          "blocks": len(hashes), "nbytes": values.nbytes,
                          "elapsed_seconds": time.perf_counter() - started}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        arrays = list(executor.map(hash_array, manifest["arrays"]))
    result = {"schema": BLOCK_SCHEMA, "cache_manifest_sha256": fingerprint(manifest),
              "contract": manifest["contract"], "nbytes": manifest["nbytes"], "arrays": arrays}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, output)
    temporary.unlink()
    print(json.dumps({"event": "block_index_ready", "output": str(output),
                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return result


class SharedDecodedArray(SourceIndexedArray):
    """Fill each independently verified block once across forked ranks/workers."""

    def __init__(self, path: Path, record: dict, source_length: int) -> None:
        if source_identity(path) != record["source_identity"]:
            raise ValueError(f"Indexed cache file identity differs: {path}")
        self.source = np.load(path, mmap_mode="r", allow_pickle=False)
        if (list(self.source.shape) != record["shape"] or self.source.dtype != np.float32
                or len(self.source) != source_length):
            raise ValueError("Shared cache shape/dtype/coverage differs")
        self.record = record
        self.mapping = mmap.mmap(-1, self.source.nbytes, flags=mmap.MAP_SHARED)
        self.writable = np.ndarray(self.source.shape, dtype=np.float32, buffer=self.mapping)
        super().__init__(self.writable.view(), np.arange(source_length), source_length)
        context = multiprocessing.get_context("fork")
        self.ready = context.RawArray("b", len(record["block_sha256"]))
        self.locks = [context.Lock() for _ in range(min(4, len(self.ready)))]

    def ensure_block(self, block_index: int) -> None:
        """Publish readiness only after copy and SHA256 verification finish."""
        with self.locks[block_index % len(self.locks)]:
            if self.ready[block_index] == 1:
                return
            if self.ready[block_index] < 0:
                raise ValueError("Shared cache block previously failed verification")
            start = block_index * self.record["rows_per_block"]
            stop = min(len(self.values), start + self.record["rows_per_block"])
            try:
                target = self.writable[start:stop]
                np.copyto(target, self.source[start:stop], casting="no")
                if hashlib.sha256(memoryview(target).cast("B")).hexdigest() != self.record["block_sha256"][block_index]:
                    raise ValueError(f"Shared cache block checksum failed: {self.record['filename']}:{block_index}")
                self.ready[block_index] = 1
            except BaseException:
                self.ready[block_index] = -1
                raise

    def ensure_rows(self, mapped: np.ndarray) -> None:
        """Read requested original rows without changing their order or sampling."""
        mapped = np.asarray(mapped)
        if np.any(mapped < 0) or np.any(mapped >= len(self.values)):
            raise IndexError("Shared cache row is outside the source")
        for block_index in np.unique(mapped // self.record["rows_per_block"]):
            self.ensure_block(int(block_index))

    def __getitem__(self, index: Any) -> np.ndarray:
        self.ensure_rows(self.row_map[index])
        return super().__getitem__(index)

    def warm(self) -> None:
        """Load remaining blocks in the background while real training runs."""
        for block_index in range(len(self.ready)):
            self.ensure_block(block_index)


class StreamingBenchmarkDataset(BenchmarkDataset):
    """Keep the original sampler and values, but do not gate startup on all RAM."""

    def __init__(self, *args: Any, block_index_path: str, **kwargs: Any) -> None:
        self.block_index_path = Path(block_index_path)
        super().__init__(*args, **kwargs)

    def _load_compatible_cache(self, directory: str, mode: str) -> None:
        root, manifest, contract = self._verified_cache_manifest(directory, mode)
        index = json.loads(self.block_index_path.read_text())
        if (index["schema"] != BLOCK_SCHEMA or index["cache_manifest_sha256"] != fingerprint(manifest)
                or index["contract"] != manifest["contract"]):
            raise ValueError("Block index does not match the original cache manifest")
        if len(index["arrays"]) != len(manifest["arrays"]):
            raise ValueError("Block index coverage differs")
        groups = [{"data": {}} for _ in self.groups]
        for original, record in zip(manifest["arrays"], index["arrays"]):
            if any(record.get(key) != value for key, value in original.items()):
                raise ValueError("Block index array differs from the cache manifest")
            rows_per_block = record["rows_per_block"]
            expected_blocks = (record["shape"][0] + rows_per_block - 1) // rows_per_block if rows_per_block > 0 else 0
            if rows_per_block < 1 or len(record["block_sha256"]) != expected_blocks:
                raise ValueError("Block index has invalid chunk coverage")
            task_index = record["task_index"]
            length = int(self.episode_ends[task_index][-1])
            groups[task_index]["data"][record["key"]] = SharedDecodedArray(
                root / record["filename"], record, length,
            )
        self.groups = groups
        self.manifest["decoded_cache_contract"] = contract["schema"]
        print(json.dumps({"event": "shared_ram_reserved", "nbytes": manifest["nbytes"],
                          "block_index_sha256": hashlib.sha256(self.block_index_path.read_bytes()).hexdigest(),
                          "full_data_loaded": False}), flush=True)

    def _cached_image_view(self, array: Any, indices: np.ndarray, key: str) -> np.ndarray:
        if isinstance(array, SharedDecodedArray):
            array.ensure_rows(array.row_map[indices])
        return super()._cached_image_view(array, indices, key)

    def cache_status(self) -> dict:
        """Report shared readiness without accessing unverified image bytes."""
        states = [np.frombuffer(array.ready, dtype=np.int8)
                  for group in self.groups for array in group["data"].values()]
        if any(np.any(state < 0) for state in states):
            raise ValueError("A shared cache block failed verification")
        return {"ready_blocks": sum(int(np.sum(state == 1)) for state in states),
                "total_blocks": sum(len(state) for state in states),
                "complete": all(bool(np.all(state == 1)) for state in states)}

    def warm_cache(self, workers: int) -> None:
        """Verify the rest of shared RAM concurrently with productive warmup."""
        arrays = [array for group in self.groups for array in group["data"].values()]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(lambda array: array.warm(), arrays))
