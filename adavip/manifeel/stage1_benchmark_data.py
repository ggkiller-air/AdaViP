"""Auditable batches and read-only RAM loading for isolated DDP benchmarks."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator, List

import numpy as np
import torch
from torch.utils.data import Sampler

from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray, cache_contract, fingerprint
from adavip.manifeel.stage1_parallel_data import CachedStage1Dataset
from adavip.manifeel.stage1_v2_training import BalancedTaskSampler


class BenchmarkDataset(CachedStage1Dataset):
    """Verify the original cache contract, then copy sequentially into shared RAM."""

    def _verified_cache_manifest(self, directory: str, mode: str) -> tuple:
        """Check source coverage and memory without reading image payloads."""
        import copy

        if mode != "ram":
            raise ValueError("This benchmark requires verified resident RAM, not mmap reads")
        root = Path(directory)
        contract = json.loads((root / "contract.json").read_text())
        cached = contract["manifest"]
        for name in ("shape_meta", "task_embedding_sha256", "action_codec"):
            if self.manifest[name] != cached[name]:
                raise ValueError(f"Cache changes {name}")
        if len(self.manifest["stores"]) != len(cached["stores"]):
            raise ValueError("Cache task count differs")
        for current, original in zip(self.manifest["stores"], cached["stores"]):
            for name in ("task", "path", "arrays", "episode_ends"):
                if current[name] != original[name]:
                    raise ValueError(f"Cache source changes {name}")
            if sorted(original["train_episodes"] + original["val_episodes"]) != list(
                    range(len(current["episode_ends"]))):
                raise ValueError("Cache must cover every demonstration exactly once")
        proxy = copy.copy(self)
        proxy.manifest = cached
        manifest = json.loads((root / "manifest.json").read_text())
        allocated_mib = os.environ.get("SLURM_MEM_PER_NODE")
        if allocated_mib and 2 * manifest["nbytes"] + 96 * 1024 ** 3 > int(allocated_mib) * 1024 ** 2:
            raise MemoryError("Reserve RAM for both decoded arrays and source page cache plus workers/models")
        if manifest["contract"] != fingerprint(cache_contract(proxy)):
            raise ValueError("Cache metadata/source contract differs")
        expected = {(index, key) for index in range(len(self.groups))
                    for key in (*self.shape_meta["obs"].keys(), "action") if key != "task_embedding"}
        actual = [(item["task_index"], item["key"]) for item in manifest["arrays"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("Cache arrays are missing or duplicated")
        return root, manifest, contract

    def _load_compatible_cache(self, directory: str, mode: str) -> None:
        root, manifest, contract = self._verified_cache_manifest(directory, mode)
        started = time.perf_counter()
        groups = [{"data": {}} for _ in self.groups]
        for record in manifest["arrays"]:
            source = np.load(root / record["filename"], mmap_mode="r", allow_pickle=False)
            if list(source.shape) != record["shape"] or source.dtype != np.float32:
                raise ValueError("Cache array shape/dtype differs")
            values = np.empty(source.shape, dtype=np.float32)
            rows_per_block = max(1, (32 * 1024 ** 2) // max(1, source[0].nbytes))
            digest = hashlib.sha256()
            array_started = time.perf_counter()
            for offset in range(0, len(source), rows_per_block):
                stop = min(len(source), offset + rows_per_block)
                values[offset:stop] = source[offset:stop]
                digest.update(memoryview(values[offset:stop]).cast("B"))
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Cache checksum failed: {record['filename']}")
            task_index = record["task_index"]
            source_length = int(self.episode_ends[task_index][-1])
            if len(values) != source_length:
                raise ValueError("All-demo cache must contain every original frame")
            groups[task_index]["data"][record["key"]] = SourceIndexedArray(
                values, np.arange(source_length), source_length,
            )
            print(json.dumps({"event": "verified_ram_array", "filename": record["filename"],
                              "nbytes": values.nbytes,
                              "seconds": time.perf_counter() - array_started}), flush=True)
            del source
        self.groups = groups
        self.manifest["decoded_cache_contract"] = contract["schema"]
        print(json.dumps({"event": "verified_ram_ready", "nbytes": manifest["nbytes"],
                          "seconds": time.perf_counter() - started}), flush=True)

    def __getitem__(self, index: int) -> dict:
        """Keep sample identity alongside the original policy inputs for audits."""
        result = super().__getitem__(index)
        result["dataset_index"] = torch.tensor(index, dtype=torch.long)
        return result


class BalancedStreamBatches(Sampler[List[int]]):
    """Continue balanced epochs across batch boundaries without dropping any rows."""

    def __init__(self, dataset: Any, seed: int, local_batch: int, rank: int,
                 world_size: int, steps: int | None, consumed: int = 0) -> None:
        if local_batch < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid batch/rank/world size")
        if (steps is not None and steps < 1) or consumed < 0:
            raise ValueError("Invalid step count or sample cursor")
        self.dataset, self.seed = dataset, seed
        self.local_batch, self.rank, self.world_size = local_batch, rank, world_size
        self.steps, self.consumed = steps, consumed

    def __len__(self) -> int:
        if self.steps is None:
            raise TypeError("An unbounded balanced stream has no length")
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        epoch_size = len(BalancedTaskSampler(self.dataset, self.seed))
        epoch, offset = divmod(self.consumed, epoch_size)

        def stream() -> Iterator[int]:
            nonlocal epoch, offset
            while True:
                yield from BalancedTaskSampler(self.dataset, self.seed, epoch, offset)
                epoch, offset = epoch + 1, 0

        iterator = stream()
        global_batch = self.local_batch * self.world_size
        for _ in itertools.count() if self.steps is None else range(self.steps):
            batch = list(itertools.islice(iterator, global_batch))
            start = self.rank * self.local_batch
            yield batch[start:start + self.local_batch]


class PinnedPrefetch:
    """Pin CPU-worker batches in a thread after all worker processes have forked."""

    def __init__(self, iterator: Iterator, device: torch.device) -> None:
        self.iterator, self.device = iterator, device
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        from diffusion_policy.common.pytorch_util import dict_apply

        torch.cuda.set_device(self.device)
        try:
            for batch in self.iterator:
                if self.stop.is_set():
                    return
                value = dict_apply(batch, lambda tensor: tensor.pin_memory())
                while not self.stop.is_set():
                    try:
                        self.queue.put(value, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception as error:
            self.queue.put(error)

    def __next__(self) -> dict:
        result = self.queue.get(timeout=600)
        if isinstance(result, Exception):
            raise result
        return result

    def close(self) -> None:
        """Stop the pinning thread without waiting on a full prefetched queue."""
        self.stop.set()
        self.thread.join(timeout=5)
