"""Keep default LC precision unchanged and explicit runtime choices auditable."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.manifeel.language_runtime import configure_profile


def test_language_runtime_profiles_restore_explicit_precision_flags() -> None:
    """Each profile must reset inherited global flags, including matmul TF32."""
    original = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        configure_profile("stage0")
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        configure_profile("strict")
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.allow_tf32
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.benchmark
        configure_profile("tf32_deterministic")
        assert torch.backends.cudnn.deterministic and torch.backends.cudnn.allow_tf32
        configure_profile("nondeterministic")
        assert not torch.backends.cudnn.deterministic and not torch.backends.cudnn.allow_tf32
        with pytest.raises(ValueError, match="Unknown"):
            configure_profile("implicit")
    finally:
        (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
         torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) = original
