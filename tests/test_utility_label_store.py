"""Full label index coverage, deterministic streams and transactional resume evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adavip.manifeel.utility_label_pilot import aggregate_losses
from adavip.manifeel.utility_label_store import (
    INDEX_DTYPE, commit_shard, complete_window_index, read_shard, shard_ranges, window_draws, worker_ranges,
)


def synthetic_manifest() -> dict:
    """Build a small all-episode schema with exact, known valid-anchor counts."""
    return {"horizon": 16, "n_obs_steps": 2, "stores": [
        {"episode_ends": [20, 37], "train_episodes": [0, 1], "val_episodes": []},
        {"episode_ends": [18], "train_episodes": [0], "val_episodes": []},
    ]}


def synthetic_shard() -> dict:
    """Generate finite paired losses using the authorized ordered stratification."""
    index = complete_window_index(synthetic_manifest())[:8]
    timesteps = np.stack([window_draws(int(row["window_id"]), 20261065, 20261066)[0] for row in index])
    losses = np.random.default_rng(11).uniform(0.001, 0.1, (8, 64, 10)).astype(np.float32)
    return {"window_index": index, "timesteps": timesteps,
            "noise": np.zeros((8, 64, 16, 7), dtype=np.float32), "losses": losses,
            "raw_signed_utility": losses[:, :, 1:] - losses[:, :, :1], **aggregate_losses(losses, timesteps)}


def test_complete_index_excludes_only_incomplete_anchors() -> None:
    index = complete_window_index(synthetic_manifest())
    assert len(index) == 5 + 2 + 3
    np.testing.assert_array_equal(index["window_id"], np.arange(10))
    np.testing.assert_array_equal(index["timestep"], [1, 2, 3, 4, 5, 1, 2, 1, 2, 3])
    np.testing.assert_array_equal(index["source_index"], [1, 2, 3, 4, 5, 21, 22, 1, 2, 3])
    np.testing.assert_array_equal(index["dataset_index"], [1, 2, 3, 4, 5, 21, 22, 38, 39, 40])
    assert ((index["timestep"] >= 1) & (index["timestep"] <= index["episode_length"] - 15)).all()


def test_complete_index_rejects_other_split_and_schema() -> None:
    manifest = synthetic_manifest()
    manifest["stores"][0]["val_episodes"] = [1]
    with pytest.raises(ValueError, match="all-episode"):
        complete_window_index(manifest)
    manifest = synthetic_manifest()
    manifest["horizon"] = 8
    with pytest.raises(ValueError, match="horizon16"):
        complete_window_index(manifest)


def test_draw_streams_are_window_stable_and_order_independent() -> None:
    first, seed = window_draws(7, 20261065, 20261066)
    window_draws(8, 20261065, 20261066)
    repeated, repeated_seed = window_draws(7, 20261065, 20261066)
    np.testing.assert_array_equal(first, repeated)
    assert seed == repeated_seed
    second, second_seed = window_draws(8, 20261065, 20261066)
    assert second_seed != seed and not np.array_equal(first, second)
    assert [int(((first >= lower) & (first < upper)).sum()) for lower, upper in
            [(0, 12), (12, 25), (25, 50), (50, 75), (75, 100)]] == [16, 12, 12, 12, 12]


def test_shard_ranges_cover_all_65265_rows_once() -> None:
    ranges = shard_ranges(65265, 256)
    assert len(ranges) == 255
    assert ranges[-1] == (65024, 65265)
    assert sum(stop - start for start, stop in ranges) == 65265
    assert all(previous[1] == following[0] for previous, following in zip(ranges, ranges[1:]))
    with pytest.raises(ValueError):
        shard_ranges(0, 256)


def test_committed_shard_roundtrip_preserves_signed_labels(tmp_path: Path) -> None:
    arrays = synthetic_shard()
    commit_shard(tmp_path, 0, 8, "contract", arrays)
    loaded = read_shard(tmp_path, 0, 8, "contract", arrays["window_index"])
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded[key], value)
    assert (loaded["utility"] < 0).any()
    with pytest.raises(FileExistsError):
        commit_shard(tmp_path, 0, 8, "contract", arrays)


def test_corrupt_committed_shard_fails_instead_of_silent_resume(tmp_path: Path) -> None:
    arrays = synthetic_shard()
    commit_shard(tmp_path, 0, 8, "contract", arrays)
    with (tmp_path / "shard_000000_000008.npz").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        read_shard(tmp_path, 0, 8, "contract", arrays["window_index"])


def test_unpublished_shard_can_be_recomputed_after_interruption(tmp_path: Path) -> None:
    arrays = synthetic_shard()
    (tmp_path / "shard_000000_000008.npz").write_bytes(b"incomplete")
    assert read_shard(tmp_path, 0, 8, "contract", arrays["window_index"]) is None
    commit_shard(tmp_path, 0, 8, "contract", arrays)
    assert read_shard(tmp_path, 0, 8, "contract", arrays["window_index"]) is not None


def test_resume_rejects_wrong_contract_and_source_indices(tmp_path: Path) -> None:
    arrays = synthetic_shard()
    commit_shard(tmp_path, 0, 8, "contract", arrays)
    with pytest.raises(ValueError, match="different contract"):
        read_shard(tmp_path, 0, 8, "other", arrays["window_index"])
    changed = arrays["window_index"].copy()
    changed["timestep"][0] += 1
    with pytest.raises(AssertionError):
        read_shard(tmp_path, 0, 8, "contract", changed)
