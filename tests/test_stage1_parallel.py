"""CPU regressions for protocol-preserving parallel data access."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_global_shards_reconstruct_every_original_batch(world_size: int) -> None:
    pytest.importorskip("torch")
    from adavip.manifeel.stage1_parallel import GlobalBatchShard, mean_loss_scale

    original = list(range(89))
    shards = [list(GlobalBatchShard(original, 32, rank, world_size))
              for rank in range(world_size)]
    reconstructed = []
    for batch_index in range(len(shards[0])):
        batch = [index for rank in range(world_size) for index in shards[rank][batch_index]]
        assert batch == original[batch_index * 32:(batch_index + 1) * 32]
        reconstructed.extend(batch)
        scales = [mean_loss_scale(len(shards[rank][batch_index]), len(batch), world_size)
                  for rank in range(world_size)]
        assert sum(scales) / world_size == pytest.approx(1.0)
    assert reconstructed == original


def test_global_shards_reject_empty_final_rank() -> None:
    pytest.importorskip("torch")
    from adavip.manifeel.stage1_parallel import GlobalBatchShard

    with pytest.raises(ValueError, match="empty rank"):
        GlobalBatchShard(list(range(33)), 32, 0, 4)


@pytest.mark.parametrize("indices", [np.arange(2, 8), np.asarray([1, 3, 4, 7])])
def test_fast_cache_reads_are_bitwise_identical(indices: np.ndarray) -> None:
    pytest.importorskip("torch")
    from adavip.manifeel.decoded_sensor_cache import SourceIndexedArray
    from adavip.manifeel.stage1_parallel_data import CachedStage1Dataset
    from adavip.manifeel.stage1_v2_dataset import Stage1V2ManiFeelDataset

    dataset = CachedStage1Dataset.__new__(CachedStage1Dataset)
    dataset.shape_meta = {"obs": {"front": {"shape": [3, 5, 6]}}}
    values = np.random.default_rng(42).random((12, 5, 6, 3), dtype=np.float32)
    array = SourceIndexedArray(values, np.arange(12), 12)
    expected = Stage1V2ManiFeelDataset._read_images(dataset, array, indices, "front")
    actual = dataset._read_images(array, indices, "front")
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert not array.values.flags.writeable


def _compare_distributed_gradient(rank: int, rendezvous: str) -> None:
    import torch
    import torch.distributed as distributed
    from torch.nn.parallel import DistributedDataParallel
    from adavip.manifeel.stage1_parallel import mean_loss_scale

    distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        reference = torch.nn.Linear(3, 2)
        parallel = DistributedDataParallel(torch.nn.Linear(3, 2))
        parallel.module.load_state_dict(reference.state_dict())
        inputs = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 21
        reference(inputs).square().mean().backward()
        local = inputs[7 * rank // 2:7 * (rank + 1) // 2]
        loss = parallel(local).square().mean() * mean_loss_scale(len(local), 7, 2)
        loss.backward()
        for expected, actual in zip(reference.parameters(), parallel.module.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad)
    finally:
        distributed.destroy_process_group()


def test_uneven_ddp_gradients_equal_serial_global_mean(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    torch.multiprocessing.spawn(_compare_distributed_gradient, args=(rendezvous,), nprocs=2)
