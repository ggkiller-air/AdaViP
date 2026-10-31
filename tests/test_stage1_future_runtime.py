"""Benchmark isolation and image-layout controls without a GPU workload."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.manifeel.benchmark_stage1_future_runtime import (
    PROFILES, apply_image_layout, check_production_isolation, float_output_hook,
    image_layout_hook, summarize,
)


def test_benchmark_rejects_production_output(tmp_path: Path) -> None:
    production = tmp_path / "formal"
    for target in (production, production / "B", production / "B/checkpoints"):
        with pytest.raises(ValueError, match="outside"):
            check_production_isolation(target, production)
    check_production_isolation(tmp_path / "benchmark", production)


def test_layout_hook_preserves_nonimage_inputs_and_gradients() -> None:
    value = torch.randn(2, 3, 8, 8, requires_grad=True)
    state = torch.randn(2, 256)
    transformed, unchanged = image_layout_hook(None, (value, state))
    assert transformed.is_contiguous(memory_format=torch.channels_last)
    assert unchanged is state
    torch.testing.assert_close(transformed, value, rtol=0, atol=0)
    transformed.square().sum().backward()
    torch.testing.assert_close(value.grad, 2 * value)


def test_loss_output_hook_retains_gradient_and_uses_float32() -> None:
    value = torch.ones(2, 512, dtype=torch.bfloat16, requires_grad=True)
    result = float_output_hook(None, (), value)
    assert result.dtype == torch.float32
    result.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))


def test_summary_compares_each_experiment_separately() -> None:
    rows = []
    for experiment, baseline in (("B", 100), ("E", 50)):
        for profile, speeds in (("baseline", [baseline, baseline]), ("bf16", [baseline * 2, baseline * 3])):
            rows.extend(dict(experiment=experiment, profile=profile, samples_per_second=speed,
                             warmup_seconds_including_compile=1, passed_finite_checks=True) for speed in speeds)
    result = summarize(rows)
    assert result["B"]["bf16"]["speedup"] == result["E"]["bf16"]["speedup"] == 2.5
    assert result["B"]["baseline"]["median_samples_per_second"] == 100
    assert result["E"]["baseline"]["median_samples_per_second"] == 50
