"""CPU scheduling regressions; actual overlap still requires the H200 job."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    log = []

    class FakeEvent:
        def __init__(self, **kwargs):
            self.complete = False

        def record(self, stream=None):
            log.append(("event", stream))

        def query(self):
            return self.complete

    class FakeStream:
        def __init__(self, **kwargs):
            self.events = []

        def wait_event(self, event):
            self.events.append(event)
            log.append(("wait", event))

        def synchronize(self):
            log.append(("synchronize", self))

    current = FakeStream()
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext(stream))
    return SimpleNamespace(current=current, log=log)


class FakeTensor:
    """Expose only transfer and lifetime methods used by the prefetcher."""

    def __init__(self, value, pinned=True):
        self.value = value
        self.pinned = pinned
        self.recorded = []

    def is_pinned(self):
        return self.pinned

    def to(self, device, non_blocking):
        assert non_blocking is True
        return FakeTensor(self.value)

    def record_stream(self, stream):
        self.recorded.append(stream)


def test_copy_stream_preserves_order_waits_and_retains_host_lifetimes(fake_cuda):
    from adavip.manifeel.stage1_cuda_prefetch import CudaBatchPrefetch

    hosts = [{"action": FakeTensor(index), "obs": {"front": FakeTensor(index + 10)}}
             for index in range(3)]
    prefetch = CudaBatchPrefetch(iter(hosts), "cuda:0")
    prefetch.preload()
    with pytest.raises(RuntimeError, match="not been consumed"):
        prefetch.preload()
    first = prefetch.acquire()
    assert first.host is hosts[0]
    assert first.batch["action"].value == 0
    assert first.batch["obs"]["front"].recorded == [fake_cuda.current]
    assert fake_cuda.current.events == [first.copy_end]
    prefetch.preload()
    assert len(prefetch.inflight) == 2
    second = prefetch.acquire()
    assert second.host is hosts[1]
    assert second.batch["action"].value == 1
    first.copy_end.complete = True
    prefetch.preload()
    assert len(prefetch.inflight) == 2
    assert prefetch.acquire().host is hosts[2]
    prefetch.close()
    assert not prefetch.inflight
    assert prefetch.pending is None
    assert fake_cuda.log[-1] == ("synchronize", prefetch.stream)


def test_copy_stream_rejects_unpinned_inputs(fake_cuda):
    from adavip.manifeel.stage1_cuda_prefetch import CudaBatchPrefetch

    prefetch = CudaBatchPrefetch(iter([{"action": FakeTensor(1, pinned=False)}]), "cuda:0")
    with pytest.raises(ValueError, match="pinned"):
        prefetch.preload()


def test_prefetch_runs_after_backward_enqueue_before_clipping_sync():
    pytest.importorskip("torch")
    from adavip.manifeel.stage1_cuda_prefetch import backward_with_prefetch

    order = []

    class Loss:
        def backward(self):
            order.append("backward_enqueued")

    backward_with_prefetch(Loss(), lambda: order.append("copy_next_enqueued"))
    order.append("clip_finite_check_sync")
    assert order == ["backward_enqueued", "copy_next_enqueued", "clip_finite_check_sync"]
    order.clear()
    backward_with_prefetch(Loss())
    assert order == ["backward_enqueued"]
