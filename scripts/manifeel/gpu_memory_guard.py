#!/usr/bin/env python3
"""Hold a small CUDA allocation so scheduler memory-utilization checks pass."""

from __future__ import annotations

import os
import time

import torch


def main() -> None:
    """Reserve and touch a bounded fraction of the visible GPU memory."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the GPU memory guard")
    fraction = float(os.environ.get("MANIFEEL_GPU_GUARD_FRACTION", "0.12"))
    if not 0.10 < fraction < 0.25:
        raise ValueError("MANIFEEL_GPU_GUARD_FRACTION must be between 0.10 and 0.25")

    device = torch.device("cuda:0")
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    n_bytes = int(total_bytes * fraction)
    n_elements = n_bytes // torch.tensor([], dtype=torch.float32).element_size()
    allocation = torch.empty(n_elements, dtype=torch.float32, device=device)
    allocation.zero_()
    print(
        f"reserved_bytes={n_elements * allocation.element_size()} "
        f"total_bytes={total_bytes} fraction={fraction}",
        flush=True,
    )
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
