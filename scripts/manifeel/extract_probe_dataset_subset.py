#!/usr/bin/env python3
"""Retain a bounded complete-episode Zarr subset from an official local archive."""

from __future__ import annotations

import argparse
from collections.abc import MutableMapping
import itertools
import json
from pathlib import Path
import shutil
from typing import Iterator
import zipfile

import numpy as np
import zarr


class ArchiveStore(MutableMapping):
    """Expose one Zarr root in a zip as a read-only mapping."""

    def __init__(self, archive: zipfile.ZipFile, prefix: str) -> None:
        self.archive = archive
        self.prefix = prefix
        self.names = {name[len(prefix):] for name in archive.namelist()
                      if name.startswith(prefix) and not name.endswith("/")}

    def __getitem__(self, key: str) -> bytes:
        if key not in self.names:
            raise KeyError(key)
        return self.archive.read(self.prefix + key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __setitem__(self, key: str, value: bytes) -> None:
        raise TypeError("archive store is read-only")

    def __delitem__(self, key: str) -> None:
        raise TypeError("archive store is read-only")


def extract_subset(archive_path: Path, output: Path, episodes: int, keys: list[str]) -> dict:
    """Copy all required chunks, truncate metadata, and reject missing source chunks."""
    if output.exists():
        raise FileExistsError(output)
    if episodes <= 0 or not keys or len(set(keys)) != len(keys):
        raise ValueError("positive episode count and unique observation keys are required")
    if any(not key or "/" in key or key in (".", "..") for key in keys):
        raise ValueError("dataset keys must be single safe path components")
    with zipfile.ZipFile(archive_path) as archive:
        roots = [name[:-len(".zgroup")] for name in archive.namelist() if name.endswith(".zgroup")]
        roots = [prefix for prefix in roots if prefix + "meta/episode_ends/.zarray" in archive.namelist()]
        if len(roots) != 1:
            raise ValueError("archive must contain one unambiguous trajectory store")
        store = ArchiveStore(archive, roots[0])
        source = zarr.open_group(store=store, mode="r")
        ends = np.asarray(source["meta/episode_ends"][:])
        if episodes > len(ends) or not np.issubdtype(ends.dtype, np.integer) or np.any(np.diff(ends) <= 0):
            raise ValueError("invalid episode boundaries or requested episode count")
        frames = int(ends[episodes - 1])
        copies, metadata = {}, {}
        for key in keys:
            relative = f"data/{key}"
            config = json.loads(store[f"{relative}/.zarray"])
            if config["shape"][0] != int(ends[-1]):
                raise ValueError(f"array length disagrees with episodes: {key}")
            shape = [frames] + config["shape"][1:]
            grid = [range((size + chunk - 1) // chunk) for size, chunk in zip(shape, config["chunks"])]
            separator = config.get("dimension_separator", ".")
            chunks = [f"{relative}/" + separator.join(map(str, index)) for index in itertools.product(*grid)]
            missing = [name for name in chunks if name not in store]
            if missing:
                raise ValueError(f"missing required chunks for {key}: {missing[:3]}")
            copies[key] = chunks
            config["shape"] = shape
            metadata[key] = config
        required = sum(archive.getinfo(roots[0] + name).file_size for chunks in copies.values() for name in chunks)
        ancestor = output.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < required + 20 * 1024**3:
            raise OSError("subset would leave less than 20 GiB free")
        output.mkdir(parents=True)
        target = zarr.open_group(str(output), mode="w")
        target.attrs.update(dict(source.attrs))
        target.create_group("data")
        target.create_group("meta").create_dataset("episode_ends", data=ends[:episodes])
        for key, chunks in copies.items():
            directory = output / "data" / key
            directory.mkdir()
            (directory / ".zarray").write_text(json.dumps(metadata[key]))
            attributes = f"data/{key}/.zattrs"
            if attributes in store:
                (directory / ".zattrs").write_bytes(store[attributes])
            for name in chunks:
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(store[name])
            array = zarr.open_array(str(directory), mode="r")
            if array.shape[0] != frames or not np.isfinite(array[0]).all() or not np.isfinite(array[-1]).all():
                raise ValueError(f"invalid decoded endpoint frames: {key}")
        manifest = {"archive": str(archive_path.resolve()), "output": str(output.resolve()),
                    "episodes": episodes, "frames": frames, "keys": keys, "chunk_bytes": required,
                    "original_episode_count": len(ends), "scope": "Analysis-only complete-episode subset, not training data."}
        (output / "probe_subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


def main() -> None:
    """Extract only explicitly selected modalities and first complete episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--keys", nargs="+", default=["wrist", "right_tactile_camera_taxim", "state", "action"])
    args = parser.parse_args()
    print(json.dumps(extract_subset(args.archive, args.output_dir, args.episodes, args.keys), indent=2))


if __name__ == "__main__":
    main()
