"""Bounded checks for multiprocessing transport before allocating model memory."""

from __future__ import annotations

import os
import tempfile
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any


def check_dataloader_ipc(timeout_seconds: float = 30) -> dict[str, Any]:
    """Verify the requested temporary directory, Unix sockets and tensor transfer."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    selected = tempfile.gettempdir()
    requested = os.environ.get("TMPDIR")
    if requested and Path(requested).resolve() != Path(selected).resolve():
        raise RuntimeError(f"TMPDIR is not usable: requested {requested}, selected {selected}")
    estimated_socket = os.path.join(selected, "pymp-xxxxxxxx", "listener-xxxxxxxx")
    if len(os.fsencode(estimated_socket)) >= 108:
        raise RuntimeError("Multiprocessing Unix socket path is too long; set a shorter writable TMPDIR")
    with Listener() as listener:
        socket_bytes = len(os.fsencode(listener.address))
    loader = DataLoader(
        TensorDataset(torch.arange(4)), batch_size=2, num_workers=2,
        timeout=timeout_seconds, generator=torch.Generator().manual_seed(42),
    )
    received = torch.cat([batch[0] for batch in loader])
    torch.testing.assert_close(received, torch.arange(4), rtol=0, atol=0)
    return {"event": "ipc_preflight_ok", "tmpdir": selected, "socket_path_bytes": socket_bytes, "workers": 2}
