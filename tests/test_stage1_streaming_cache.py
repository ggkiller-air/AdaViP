"""CPU checks for verified shared caching before any GPU benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def make_cache(root: Path) -> tuple:
    from adavip.manifeel.decoded_sensor_cache import fingerprint

    values = np.arange(20 * 3 * 2 * 3, dtype=np.float32).reshape(20, 3, 2, 3)
    path = root / "task0_front.npy"
    np.save(path, values)
    record = {"filename": path.name, "task_index": 0, "key": "front", "shape": list(values.shape),
              "sha256": hashlib.sha256(memoryview(values).cast("B")).hexdigest()}
    manifest = {"contract": fingerprint({"test": True}), "nbytes": values.nbytes, "arrays": [record]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return path, values


def test_block_index_validates_full_source_and_does_not_overwrite(tmp_path) -> None:
    pytest.importorskip("torch")
    from adavip.manifeel.stage1_streaming_cache import build_block_index

    path, values = make_cache(tmp_path)
    output = tmp_path / "blocks.json"
    index = build_block_index(tmp_path, output, workers=2, block_bytes=values[0].nbytes * 3)
    assert len(index["arrays"][0]["block_sha256"]) == 7
    with pytest.raises(FileExistsError):
        build_block_index(tmp_path, output)
    changed = np.load(path, mmap_mode="r+")
    changed[-1] += 1
    changed.flush()
    with pytest.raises(ValueError, match="Full cache checksum"):
        build_block_index(tmp_path, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def _read_shared(array, result) -> None:
    array.ensure_rows(np.asarray([9, 3, 9]))
    result.put(float(array[[9, 3, 9]].sum()))


def test_shared_blocks_are_lazy_bitwise_readonly_and_visible_across_fork(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from adavip.manifeel.stage1_streaming_cache import SharedDecodedArray, build_block_index

    path, values = make_cache(tmp_path)
    index = build_block_index(tmp_path, tmp_path / "blocks.json", block_bytes=values[0].nbytes * 3)
    array = SharedDecodedArray(path, index["arrays"][0], len(values))
    assert sum(array.ready) == 0
    assert not array.values.flags.writeable
    context = torch.multiprocessing.get_context("fork")
    result = context.Queue()
    process = context.Process(target=_read_shared, args=(array, result))
    process.start()
    process.join(timeout=20)
    assert process.exitcode == 0
    assert result.get(timeout=2) == float(values[[9, 3, 9]].sum())
    assert list(array.ready) == [0, 1, 0, 1, 0, 0, 0]
    np.testing.assert_array_equal(array[[9, 3, 9]], values[[9, 3, 9]])
    array.warm()
    assert all(array.ready)
    np.testing.assert_array_equal(array.values, values)
    np.testing.assert_array_equal(np.load(path), values)


def test_shared_block_rejects_source_identity_and_post_index_corruption(tmp_path) -> None:
    pytest.importorskip("torch")
    from adavip.manifeel.stage1_streaming_cache import SharedDecodedArray, build_block_index

    path, values = make_cache(tmp_path)
    index = build_block_index(tmp_path, tmp_path / "blocks.json", block_bytes=values[0].nbytes * 3)
    array = SharedDecodedArray(path, index["arrays"][0], len(values))
    changed = np.load(path, mmap_mode="r+")
    changed[0] += 1
    changed.flush()
    with pytest.raises(ValueError, match="identity"):
        SharedDecodedArray(path, index["arrays"][0], len(values))
    with pytest.raises(ValueError, match="block checksum"):
        array.ensure_rows(np.asarray([0]))
    assert array.ready[0] == -1
    with pytest.raises(ValueError, match="previously failed"):
        array.ensure_rows(np.asarray([0]))


def test_balanced_stream_supports_unbounded_cache_warmup_without_order_change() -> None:
    pytest.importorskip("torch")
    from itertools import islice
    from types import SimpleNamespace
    from adavip.manifeel.stage1_benchmark_data import BalancedStreamBatches

    dataset = SimpleNamespace(task_lengths=[5] * 9)
    expected = list(BalancedStreamBatches(dataset, 42, 32, 2, 4, 7))
    infinite = BalancedStreamBatches(dataset, 42, 32, 2, 4, None)
    assert list(islice(infinite, 7)) == expected
    with pytest.raises(TypeError, match="no length"):
        len(infinite)
