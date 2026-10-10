"""CPU checks for non-intervening fusion attention recording and summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from adavip.manifeel.attention_trace import AttentionTraceRecorder


class StubFusionEncoder(nn.Module):
    """Small vision-query fusion layer with a direct tactile path."""

    use_fusion = True
    state_key = "state"

    def __init__(self) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(4, 2, batch_first=True, dropout=0)
        self.fusion_gate = nn.Parameter(torch.tensor(0.8))

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply a vision-query attention to two modality tokens."""
        vision = observation["vision"].unsqueeze(1)
        tactile = observation["tactile"].unsqueeze(1)
        attended, _ = self.cross_attention(vision, torch.cat((vision, tactile), dim=1),
                                           torch.cat((vision, tactile), dim=1),
                                           need_weights=False)
        return vision.squeeze(1) + torch.tanh(self.fusion_gate) * attended.squeeze(1)


class StubFusionPolicy:
    """Flatten two observation frames in the same order as Diffusion Policy."""

    def __init__(self) -> None:
        self.obs_encoder = StubFusionEncoder().eval()

    def predict_action(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the encoded observations reshaped by environment and time."""
        batch_size, frames = observation["state"].shape[:2]
        feature = self.obs_encoder({key: value.reshape(-1, value.shape[-1])
                                    for key, value in observation.items()})
        return {"action": feature.reshape(batch_size, frames, -1)}


def test_recording_preserves_action_and_captures_per_head_weights(tmp_path: Path) -> None:
    """Diagnostic recomputation must not intervene in the actual policy."""
    torch.manual_seed(7)
    policy = StubFusionPolicy()
    observation = {"vision": torch.randn(2, 2, 4), "tactile": torch.randn(2, 2, 4),
                   "state": torch.zeros(2, 2, 7)}
    original = policy.predict_action(observation)["action"]
    recorder = AttentionTraceRecorder(policy, tmp_path)
    recorded = policy.predict_action(observation)["action"]
    summary = recorder.save()
    recorder.close()
    assert torch.equal(original, recorded)
    assert summary["chunks"] == 1
    with np.load(tmp_path / "attention_trace.npz") as trace:
        assert trace["weights"].shape == (1, 2, 2, 2, 2)
        np.testing.assert_allclose(trace["weights"].sum(axis=-1), 1, atol=1e-6)
        assert trace["fusion_to_vision_norm"].shape == (1, 2, 2)


def test_attention_summary_requires_labeled_stage_boundaries() -> None:
    """Stages must be supplied rather than inferred from arbitrary elapsed time."""
    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/plot_attention_trace.py"
    spec = importlib.util.spec_from_file_location("plot_attention_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights = np.full((6, 2, 2, 4, 2), 0.5)
    weights[:2, ..., 0] = 0.8
    weights[:2, ..., 1] = 0.2
    summary = module.summarize_attention(weights, (2, 4))
    assert summary["stage_tactile_weight"]["approach"] == pytest.approx(0.2)
    assert summary["stage_tactile_weight"]["contact"] == pytest.approx(0.5)
    assert "stage_tactile_weight" not in module.summarize_attention(weights, None)
    with pytest.raises(ValueError, match="boundaries"):
        module.summarize_attention(weights, (0, 4))


def test_plot_writes_heatmap_and_first_success(tmp_path: Path, monkeypatch) -> None:
    """The offline plotter should consume real trace shapes and preserve time units."""
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/plot_attention_trace.py"
    spec = importlib.util.spec_from_file_location("plot_attention_trace_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights = np.full((6, 2, 2, 4, 2), 0.5)
    np.savez_compressed(tmp_path / "attention_trace.npz", weights=weights,
                        fusion_to_vision_norm=np.ones((6, 2, 2)) * 0.1,
                        fusion_gate=np.asarray(0.2))
    success = np.zeros((12, 2), dtype=bool)
    success[7, 0] = True
    np.savez_compressed(tmp_path / "rollout_trace.npz", success_by_step=success)
    monkeypatch.setattr(sys, "argv", ["plot_attention_trace", "--run", str(tmp_path),
                                      "--approach-end", "2", "--contact-end", "4"])
    module.main()
    assert (tmp_path / "attention_env0.png").is_file()
    summary = __import__("json").loads((tmp_path / "attention_env0.json").read_text())
    assert summary["first_success_sim_step_env"] == 7
