"""Portable full-window utility label indices, deterministic draws and resumable shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from adavip.manifeel.utility_k64_precision import BASE_COUNTS, STRATA
from adavip.manifeel.utility_label_pilot import aggregate_losses


INDEX_DTYPE = np.dtype([(name, "<i8") for name in (
    "window_id", "task_index", "episode_index", "timestep", "source_index",
    "dataset_index", "episode_length",
)])


def complete_window_index(manifest: dict) -> np.ndarray:
    """Enumerate the same complete-history/horizon anchors used by the accepted pilot."""
    if manifest["horizon"] != 16 or manifest["n_obs_steps"] != 2:
        raise ValueError("Expected the frozen horizon16/history2 protocol")
    rows, dataset_offset = [], 0
    for task_index, store in enumerate(manifest["stores"]):
        ends = store["episode_ends"]
        if store["train_episodes"] != list(range(len(ends))) or store["val_episodes"]:
            raise ValueError("Expected the frozen all-episode index")
        previous = 0
        for episode_index, end in enumerate(ends):
            length = end - previous
            if length <= 0:
                raise ValueError("Invalid source episode boundaries")
            for timestep in range(1, length - 14):
                rows.append((len(rows), task_index, episode_index, timestep,
                             previous + timestep, dataset_offset + timestep, length))
            dataset_offset += length
            previous = end
    return np.array(rows, dtype=INDEX_DTYPE)


def window_draws(window_id: int, timestep_seed: int, noise_seed: int) -> tuple[np.ndarray, int]:
    """Derive independent per-window streams unaffected by shard size or restart order."""
    if window_id < 0:
        raise ValueError("Window id must be nonnegative")
    generator = np.random.default_rng(np.random.SeedSequence([timestep_seed, window_id]))
    timesteps = np.concatenate([
        np.concatenate([generator.integers(lower, upper, size=count, dtype=np.int64)
                        for (lower, upper), count in zip(STRATA, BASE_COUNTS)])
        for block in range(4)
    ])
    gaussian_seed = int(np.random.SeedSequence([noise_seed, window_id]).generate_state(1, dtype=np.uint64)[0])
    return timesteps, gaussian_seed


def shard_ranges(count: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return contiguous, nonoverlapping ranges, including the final partial shard."""
    if count <= 0 or chunk_size <= 0:
        raise ValueError("Window count and chunk size must be positive")
    return [(start, min(count, start + chunk_size)) for start in range(0, count, chunk_size)]


def worker_ranges(count: int, chunk_size: int, rank: int, world_size: int) -> list[tuple[int, int]]:
    """Partition existing shard identities without changing any window or draw."""
    if world_size not in (1, 2) or not 0 <= rank < world_size:
        raise ValueError("Expected one or two independent label workers")
    return shard_ranges(count, chunk_size)[rank::world_size]


def file_hash(path: Path) -> str:
    """Hash persistent artifacts without materializing them in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Commit portable JSON only after flushing its complete contents."""
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def commit_shard(root: Path, start: int, stop: int, contract_hash: str, arrays: dict) -> None:
    """Publish a shard then its hash marker; interrupted unpublished shards are recomputable."""
    destination = root / f"shard_{start:06d}_{stop:06d}.npz"
    marker = destination.with_suffix(".json")
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite committed shard: {marker}")
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    atomic_json(marker, {"start": start, "stop": stop, "contract_sha256": contract_hash,
                         "sha256": file_hash(destination), "bytes": destination.stat().st_size})


def read_shard(root: Path, start: int, stop: int, contract_hash: str,
               expected_index: np.ndarray) -> dict | None:
    """Reject corrupted committed evidence and return only complete, matching shards."""
    destination = root / f"shard_{start:06d}_{stop:06d}.npz"
    marker = destination.with_suffix(".json")
    if not marker.exists():
        return None
    metadata = json.loads(marker.read_text())
    if (metadata["start"], metadata["stop"], metadata["contract_sha256"]) != (start, stop, contract_hash):
        raise ValueError("Shard belongs to a different contract")
    if destination.stat().st_size != metadata["bytes"] or file_hash(destination) != metadata["sha256"]:
        raise ValueError(f"Committed shard integrity failure: {destination}")
    with np.load(destination, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(arrays["window_index"], expected_index)
    count = stop - start
    if arrays["noise"].shape != (count, 64, 16, 7) or not np.isfinite(arrays["noise"]).all():
        raise ValueError("Invalid saved Gaussian draws")
    np.testing.assert_array_equal(arrays["raw_signed_utility"], arrays["losses"][:, :, 1:] - arrays["losses"][:, :, :1])
    reduced = aggregate_losses(arrays["losses"], arrays["timesteps"])
    for key, expected in reduced.items():
        np.testing.assert_allclose(arrays[key], expected, rtol=1e-12, atol=1e-14)
    if arrays["utility"].shape != (count, 9) or not np.isfinite(arrays["utility"]).all():
        raise ValueError("Invalid signed labels")
    return arrays
