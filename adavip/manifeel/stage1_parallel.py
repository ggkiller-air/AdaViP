"""DDP helpers that preserve global batches and the balanced sample sequence."""

from __future__ import annotations

import itertools
import math
from typing import Iterator, List

import torch
from torch import Tensor
from torch.utils.data import Sampler


class GlobalBatchShard(Sampler[List[int]]):
    """Split each original global batch without padding or repeating any samples."""

    def __init__(self, sampler: Sampler[int], batch_size: int, rank: int, world_size: int) -> None:
        if batch_size < world_size or not 0 <= rank < world_size:
            raise ValueError("Every rank needs a nonempty shard of each full batch")
        remainder = len(sampler) % batch_size
        if remainder and remainder < world_size:
            raise ValueError("The final global batch cannot leave an empty rank")
        self.sampler = sampler
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        iterator = iter(self.sampler)
        while True:
            batch = list(itertools.islice(iterator, self.batch_size))
            if not batch:
                return
            start = len(batch) * self.rank // self.world_size
            stop = len(batch) * (self.rank + 1) // self.world_size
            yield batch[start:stop]


class LossModule(torch.nn.Module):
    """Expose the policy loss through DDP's forward/backward hooks."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict) -> dict[str, Tensor]:
        """Return scalar objectives only, excluding unused diagnostic embeddings."""
        return {name: value for name, value in self.policy.compute_loss_with_metrics(batch).items()
                if name != "z"}


def mean_loss_scale(local_size: int, global_size: int, world_size: int) -> float:
    """Correct DDP's rank-average to the true sample-average for uneven shards."""
    return world_size * local_size / global_size
