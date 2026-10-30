"""Explicit, recorded CUDA runtime profiles for language-conditioning runs."""

from __future__ import annotations

import torch


PROFILES = {
    "strict": {"deterministic": True, "cudnn_tf32": False},
    "nondeterministic": {"deterministic": False, "cudnn_tf32": False},
    "tf32_deterministic": {"deterministic": True, "cudnn_tf32": True},
    "stage0": {"deterministic": False, "cudnn_tf32": True},
}


def configure_profile(profile: str) -> dict:
    """Set FP32 compute flags without enabling BF16 or matmul TF32."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown language runtime profile: {profile}")
    flags = PROFILES[profile]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = flags["deterministic"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = False
    return dict(flags, matmul_tf32=False, cudnn_benchmark=False)
