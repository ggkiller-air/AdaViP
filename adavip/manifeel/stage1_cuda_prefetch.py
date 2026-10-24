"""Explicit copy-stream prefetching without changing batch order or contents."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from diffusion_policy.common.pytorch_util import dict_apply


def backward_with_prefetch(loss: torch.Tensor, preload_next: Callable[[], None] | None = None) -> None:
    """Enqueue next H2D before gradient-clipping finite checks synchronize CPU."""
    loss.backward()
    if preload_next is not None:
        preload_next()


@dataclass
class BatchTransfer:
    """Keep original identities, pinned storage and independent timing events."""

    host: dict
    batch: dict
    host_wait_seconds: float
    copy_start: Any
    copy_end: Any
    wait_start: Any = None
    wait_end: Any = None


class CudaBatchPrefetch:
    """Stage N+1 after enqueuing N, on a dedicated rank-local CUDA stream."""

    def __init__(self, iterator: Iterator, device: torch.device) -> None:
        self.iterator = iterator
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.pending: BatchTransfer | None = None
        self.inflight: list[tuple] = []

    def preload(self) -> None:
        """Fetch the next pinned CPU batch and enqueue its asynchronous H2D."""
        if self.pending is not None:
            raise RuntimeError("A prefetched batch has not been consumed")
        self.inflight = [(event, host) for event, host in self.inflight if not event.query()]
        started = time.perf_counter()
        host = next(self.iterator)
        waited = time.perf_counter() - started
        if not host["action"].is_pinned():
            raise ValueError("CUDA prefetch requires pinned CPU batches")
        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            copy_start.record(self.stream)
            batch = dict_apply(host, lambda value: value.to(self.device, non_blocking=True))
            copy_end.record(self.stream)
        self.pending = BatchTransfer(host, batch, waited, copy_start, copy_end)
        self.inflight.append((copy_end, host))

    def acquire(self) -> BatchTransfer:
        """Make compute wait only for this batch, with allocator-safe lifetimes."""
        if self.pending is None:
            self.preload()
        transfer = self.pending
        self.pending = None
        current = torch.cuda.current_stream(self.device)
        transfer.wait_start = torch.cuda.Event(enable_timing=True)
        transfer.wait_end = torch.cuda.Event(enable_timing=True)
        transfer.wait_start.record(current)
        current.wait_event(transfer.copy_end)
        transfer.wait_end.record(current)
        dict_apply(transfer.batch, lambda value: value.record_stream(current))
        return transfer

    def close(self) -> None:
        """Release pinned host references only after outstanding copies finish."""
        self.stream.synchronize()
        self.pending = None
        self.inflight.clear()


def transfer_on_current_stream(iterator: Iterator, device: torch.device) -> BatchTransfer:
    """Measure the original pinned-loader path with identical nonblocking copies."""
    started = time.perf_counter()
    host = next(iterator)
    waited = time.perf_counter() - started
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    copy_start.record()
    batch = dict_apply(host, lambda value: value.to(device, non_blocking=True))
    copy_end.record()
    return BatchTransfer(host, batch, waited, copy_start, copy_end)
