"""Check that storage-bounded probe subsets contain complete valid trajectories."""

from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import zarr

from scripts.manifeel.extract_probe_dataset_subset import extract_subset


@pytest.fixture(autouse=True)
def sufficient_disk_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tiny unit tests independent of the host's production free-space policy."""
    monkeypatch.setattr("scripts.manifeel.extract_probe_dataset_subset.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=100 * 1024**3))


def make_archive(path: Path, missing: bool = False) -> Path:
    """Build a small nested official-style archive with an optional corrupt chunk."""
    root = path / "original"
    store = zarr.open_group(str(root), mode="w")
    store.create_group("meta").create_dataset("episode_ends", data=np.array([3, 7]))
    data = store.create_group("data")
    data.create_dataset("state", data=np.arange(14).reshape(7, 2), chunks=(2, 2))
    data.create_dataset("unused", data=np.ones((7, 3)), chunks=(2, 3))
    archive = path / "demo.zip"
    with zipfile.ZipFile(archive, "w") as target:
        for entry in root.rglob("*"):
            if entry.is_file() and not (missing and entry.relative_to(root).as_posix() == "data/state/1.0"):
                target.write(entry, "nested/" + entry.relative_to(root).as_posix())
    return archive


def test_subset_keeps_episode_boundaries_and_omits_unused_modalities(tmp_path: Path) -> None:
    """Chunk-level copies preserve exact values even when the last chunk crosses the cut."""
    archive = make_archive(tmp_path)
    output = tmp_path / "subset"
    report = extract_subset(archive, output, 1, ["state"])
    subset = zarr.open_group(str(output), mode="r")
    np.testing.assert_array_equal(subset["data/state"][:], np.arange(6).reshape(3, 2))
    np.testing.assert_array_equal(subset["meta/episode_ends"][:], [3])
    assert "unused" not in subset["data"]
    assert report["frames"] == 3
    with pytest.raises(FileExistsError):
        extract_subset(archive, output, 1, ["state"])


def test_subset_rejects_missing_chunks_instead_of_silent_zero_fill(tmp_path: Path) -> None:
    """A partial source archive cannot silently create fabricated demonstration data."""
    archive = make_archive(tmp_path, missing=True)
    with pytest.raises(ValueError, match="missing required chunks"):
        extract_subset(archive, tmp_path / "subset", 1, ["state"])


def test_subset_preserves_storage_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse extraction before creating a partial store on an almost-full volume."""
    archive = make_archive(tmp_path)
    monkeypatch.setattr("scripts.manifeel.extract_probe_dataset_subset.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=1024))
    with pytest.raises(OSError, match="20 GiB"):
        extract_subset(archive, tmp_path / "subset", 1, ["state"])
    assert not (tmp_path / "subset").exists()
