"""Read-only decoded mappings with bounded sequential background verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def warm_cache(directory: Path) -> None:
    """Warm shared file pages without allocating a full anonymous copy."""
    manifest = json.loads((directory / "manifest.json").read_text())
    buffer = bytearray(8 * 1024 ** 2)
    total = 0
    for record in manifest["arrays"]:
        path = directory / record["filename"]
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        offset, size = mapped.offset, mapped.nbytes
        del mapped
        digest = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            stream.seek(offset)
            remaining = size
            while remaining:
                count = stream.readinto(memoryview(buffer)[:min(remaining, len(buffer))])
                if not count:
                    raise IOError(f"Truncated decoded cache: {path}")
                digest.update(memoryview(buffer)[:count])
                remaining -= count
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Corrupt decoded cache: {path}")
        total += size
        print(json.dumps({"event": "cache_array_verified", "filename": record["filename"],
                          "bytes_verified": total, "total_bytes": manifest["nbytes"]}), flush=True)
    if total != manifest["nbytes"]:
        raise ValueError("Decoded cache byte count mismatch")


class StreamingCache:
    """Expose all samples immediately; the OS shares cached pages across workers."""

    def __init__(self, source, directory: str, manifest: dict, output: Path) -> None:
        self.source, self.directory = source, Path(directory)
        self.manifest, self.output = manifest, Path(output)
        self.process = None
        self.finished = False
        self.started = 0.0

    def __call__(self):
        from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray

        groups = [{"data": {}} for _ in self.source.task_specs]
        for record in self.manifest["arrays"]:
            task = int(record["task_index"])
            values = np.load(self.directory / record["filename"], mmap_mode="r", allow_pickle=False)
            length = int(self.source.episode_ends[task][-1])
            if values.dtype != np.float32 or list(values.shape) != record["shape"] or len(values) != length:
                raise ValueError("Decoded mapping metadata changed")
            groups[task]["data"][record["key"]] = SourceIndexedArray(values, np.arange(length), length)
        self.source.groups = groups
        # Keep per-window finite/range checks until full verification; workers
        # retain this conservative setting even after the background scan ends.
        self.source.verified_ram_cache = False
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.log_path = self.output / f"cache_warm_{os.getpid()}.log"
        with self.log_path.open("w") as stream:
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), str(self.directory)],
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            )
        return self

    def check(self) -> None:
        """Surface scan failures before another optimizer update."""
        if self.finished or self.process is None:
            return
        code = self.process.poll()
        if code is None:
            return
        if code != 0:
            raise RuntimeError(f"Background cache verification failed ({code}); see {self.log_path}")
        self.finished = True
        print(json.dumps({"event": "mapped_cache_verified", "bytes": self.manifest["nbytes"],
                          "seconds": time.monotonic() - self.started}), flush=True)

    def close(self) -> None:
        """Do not leave a cache reader behind after training exits."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=30)


def preserve_training_contract(current: dict, previous: dict) -> dict:
    """Allow an explicit transport-only source migration, no scientific changes."""
    changed = {key for key in current.keys() | previous.keys()
               if current.get(key) != previous.get(key)}
    if changed - {"source_sha256"}:
        raise ValueError(f"Training contract changed beyond data transport: {sorted(changed)}")
    return previous.copy()


if __name__ == "__main__":
    warm_cache(Path(sys.argv[1]))
