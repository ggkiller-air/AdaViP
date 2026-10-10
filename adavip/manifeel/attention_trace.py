"""Read-only, per-head fusion attention traces for evaluation rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class AttentionTraceRecorder:
    """Capture vision-query weights without changing a checkpoint's forward pass."""

    def __init__(self, policy: Any, output_dir: Path) -> None:
        encoder = policy.obs_encoder
        if not getattr(encoder, "use_fusion", False):
            raise ValueError("attention traces require a fusion-enabled HyperResNet encoder")
        if encoder.training or encoder.cross_attention.training:
            raise ValueError("attention traces require an evaluation-mode encoder")
        self.policy = policy
        self.output_dir = output_dir
        self.weights: list[np.ndarray] = []
        self.fusion_ratios: list[np.ndarray] = []
        self._batch_size: int | None = None
        self._in_recompute = False
        self._original_predict_action = policy.predict_action
        self._hook = encoder.cross_attention.register_forward_hook(self._record)
        self.fusion_gate = float(torch.tanh(encoder.fusion_gate).detach().cpu())

        def traced_predict_action(observation: dict[str, torch.Tensor]) -> Any:
            self._batch_size = observation[encoder.state_key].shape[0]
            before = len(self.weights)
            try:
                result = self._original_predict_action(observation)
                if len(self.weights) != before + 1:
                    raise ValueError("expected exactly one fusion attention call per action chunk")
                return result
            finally:
                self._batch_size = None

        policy.predict_action = traced_predict_action

    def _record(self, module: Any, inputs: tuple, output: tuple) -> None:
        """Recompute only diagnostic weights with dropout disabled."""
        if self._in_recompute or self._batch_size is None:
            return
        query, key, value = inputs[:3]
        self._in_recompute = True
        try:
            with torch.no_grad():
                _, weights = module(query, key, value, need_weights=True,
                                    average_attn_weights=False)
        finally:
            self._in_recompute = False
        if weights is None or weights.shape[-2:] != (1, 2):
            raise ValueError("expected one vision query and two modality keys")
        batch_size = self._batch_size
        if weights.shape[0] % batch_size:
            raise ValueError("attention batch does not match policy observation batch")
        observations = weights.shape[0] // batch_size
        self.weights.append(weights[:, :, 0, :].detach().float().cpu().numpy().reshape(
            batch_size, observations, weights.shape[1], 2))
        attended = output[0]
        ratio = (abs(self.fusion_gate) * attended.detach().float().norm(dim=-1)
                 / query.detach().float().norm(dim=-1).clamp_min(1e-8))
        self.fusion_ratios.append(ratio.cpu().numpy().reshape(batch_size, observations))

    def close(self) -> None:
        """Remove the hook and restore the previous inference entry point."""
        self._hook.remove()
        self.policy.predict_action = self._original_predict_action

    def save(self) -> dict[str, float | int]:
        """Persist [chunk, environment, observation, head, vision/touch] weights."""
        if not self.weights:
            raise ValueError("no attention calls were recorded")
        weights = np.stack(self.weights)
        ratios = np.stack(self.fusion_ratios)
        np.savez_compressed(self.output_dir / "attention_trace.npz",
                            weights=weights, fusion_to_vision_norm=ratios,
                            fusion_gate=np.asarray(self.fusion_gate))
        return {"chunks": len(weights), "fusion_gate": self.fusion_gate,
                "mean_tactile_weight": float(weights[..., 1].mean()),
                "mean_fusion_to_vision_norm": float(ratios.mean())}
