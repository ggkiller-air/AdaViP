"""Read-only feature modulation probes for trained HyperResNet encoders."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional


class HyperNetProbeRecorder:
    """Record per-frame modulation without changing the encoder's outputs."""

    def __init__(self, encoder: Any) -> None:
        if encoder.training or any(module.training for module in encoder.modules()):
            raise ValueError("HyperNet probes require an evaluation-mode encoder")
        self.encoder = encoder
        self.batches: list[dict[str, np.ndarray]] = []
        self._features: dict[str, torch.Tensor] = {}
        self._current: dict[str, torch.Tensor] = {}
        self._recomputing = False
        self._hooks = [encoder.register_forward_pre_hook(self._start)]
        for key in (encoder.vision_key, encoder.tactile_key):
            self._hooks.append(encoder.key_model_map[key].register_forward_hook(
                self._feature_hook(key)))
        self._hooks.append(encoder.hypernet.register_forward_hook(self._modulation))
        if encoder.use_fusion:
            self._hooks.append(encoder.cross_attention.register_forward_hook(self._attention))
        if encoder.use_fusion_hypernet:
            self._hooks.append(encoder.fusion_hypernet.register_forward_hook(self._fusion_modulation))
        self._hooks.append(encoder.register_forward_hook(self._finish))

    def _start(self, module: Any, inputs: tuple) -> None:
        self._features = {}
        self._current = {}

    def _feature_hook(self, key: str) -> Any:
        def capture(module: Any, inputs: tuple, output: torch.Tensor) -> None:
            self._features[key] = output.detach()

        return capture

    def _modulation(self, module: Any, inputs: tuple, output: torch.Tensor) -> None:
        gamma_vision, beta_vision, gamma_tactile, beta_tactile = output.detach().tanh().chunk(4, dim=-1)
        features = torch.stack((self._features[self.encoder.vision_key],
                                self._features[self.encoder.tactile_key]), dim=1)
        gamma = torch.stack((gamma_vision, gamma_tactile), dim=1)
        beta = torch.stack((beta_vision, beta_tactile), dim=1)
        residual = self.encoder.alpha.detach() * (gamma * features + beta)
        modulated = features + residual
        self._current.update({
            "base_features": features,
            "gamma": gamma,
            "beta": beta,
            "modulated_features": modulated,
            "residual_ratio": residual.norm(dim=-1) / features.norm(dim=-1).clamp_min(1e-8),
            "feature_cosine": functional.cosine_similarity(features, modulated, dim=-1),
        })

    def _attention(self, module: Any, inputs: tuple, output: tuple) -> None:
        if self._recomputing:
            return
        self._recomputing = True
        try:
            with torch.no_grad():
                _, weights = module(*inputs[:3], need_weights=True, average_attn_weights=False)
        finally:
            self._recomputing = False
        if weights is None or weights.shape[-2:] != (1, 2):
            raise ValueError("expected one vision query and two modality keys")
        self._current["attention"] = weights[:, :, 0, :].detach()
        gate = self.encoder.fusion_gate.detach().tanh().abs()
        self._current["fusion_ratio"] = (
            gate * output[0].detach().norm(dim=-1).squeeze(1)
            / inputs[0].detach().norm(dim=-1).squeeze(1).clamp_min(1e-8))
        if self.encoder.use_fusion_hypernet:
            self._current["pre_fusion_hypernet_features"] = (
                inputs[0].detach().squeeze(1)
                + self.encoder.fusion_gate.detach().tanh() * output[0].detach().squeeze(1))

    def _fusion_modulation(self, module: Any, inputs: tuple, output: torch.Tensor) -> None:
        gamma, beta = output.detach().tanh().chunk(2, dim=-1)
        features = self._current["pre_fusion_hypernet_features"]
        residual = self.encoder.fusion_alpha.detach() * (gamma * features + beta)
        self._current.update({"fusion_gamma": gamma, "fusion_beta": beta,
                              "fusion_hypernet_residual_ratio": (
                                  residual.norm(dim=-1) / features.norm(dim=-1).clamp_min(1e-8))})

    def _finish(self, module: Any, inputs: tuple, output: torch.Tensor) -> None:
        if "residual_ratio" not in self._current:
            raise ValueError("encoder forward did not produce HyperNet modulation")
        self._current["encoder_output"] = output.detach()
        self._current["normalized_state"] = inputs[0][self.encoder.state_key].detach()
        self.batches.append({key: value.float().cpu().numpy().copy()
                             for key, value in self._current.items()})

    def arrays(self) -> dict[str, np.ndarray]:
        """Return frame-major arrays; modality order is vision, then tactile."""
        if not self.batches:
            raise ValueError("no encoder frames were recorded")
        return {key: np.concatenate([batch[key] for batch in self.batches], axis=0)
                for key in self.batches[0]}

    def close(self) -> None:
        """Remove all diagnostic hooks."""
        for hook in self._hooks:
            hook.remove()


class HyperNetRolloutRecorder:
    """Save encoder probes with explicit action-chunk, environment and history axes."""

    def __init__(self, policy: Any, output_dir: Path) -> None:
        self.policy = policy
        self.output_dir = output_dir
        self.probe = HyperNetProbeRecorder(policy.obs_encoder)

    def save(self) -> dict[str, object]:
        """Persist exact rollout layout and compatibility fusion attention traces."""
        arrays = self.probe.arrays()
        chunks = len(self.probe.batches)
        observations = self.policy.n_obs_steps
        frame_counts = [len(batch["gamma"]) for batch in self.probe.batches]
        if len(set(frame_counts)) != 1 or frame_counts[0] % observations:
            raise ValueError("rollout encoder frames do not have a constant environment/history layout")
        environments = frame_counts[0] // observations
        shaped = {key: value.reshape(chunks, environments, observations, *value.shape[1:])
                  for key, value in arrays.items()}
        shaped["chunk_start_sim_step"] = np.arange(chunks) * self.policy.n_action_steps
        shaped["alpha"] = np.asarray(float(self.policy.obs_encoder.alpha.detach().cpu()))
        np.savez_compressed(self.output_dir / "hypernet_rollout_trace.npz", **shaped)
        if "attention" in shaped:
            gate = float(self.policy.obs_encoder.fusion_gate.detach().tanh().cpu())
            np.savez_compressed(self.output_dir / "attention_trace.npz", weights=shaped["attention"],
                                fusion_to_vision_norm=shaped["fusion_ratio"], fusion_gate=np.asarray(gate))
        return {"chunks": chunks, "environments": environments, "observation_steps": observations,
                "summary": summarize_probe(arrays)}

    def close(self) -> None:
        """Remove diagnostic hooks without changing the policy inference entry point."""
        self.probe.close()


def select_episode_frames(episode_ends: np.ndarray, episodes: list[int],
                          frames_per_episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample deterministic, evenly spaced frames without assigning task phases."""
    ends = np.asarray(episode_ends)
    if (ends.ndim != 1 or len(ends) == 0 or not np.issubdtype(ends.dtype, np.integer)
            or ends[0] <= 0 or np.any(np.diff(ends) <= 0)):
        raise ValueError("episode ends must be strictly increasing positive integers")
    if frames_per_episode <= 0 or not episodes or len(set(episodes)) != len(episodes):
        raise ValueError("provide unique episodes and a positive frame count")
    indices, identifiers, steps = [], [], []
    for episode in episodes:
        if not 0 <= episode < len(ends):
            raise ValueError(f"episode {episode} is outside the dataset")
        start = int(ends[episode - 1]) if episode else 0
        length = int(ends[episode]) - start
        selected = np.linspace(0, length - 1, min(length, frames_per_episode), dtype=np.int64)
        indices.extend((selected + start).tolist())
        identifiers.extend([episode] * len(selected))
        steps.extend(selected.tolist())
    return np.asarray(indices), np.asarray(identifiers), np.asarray(steps)


def summarize_probe(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Describe modulation and temporal variation, not causal modality reliance."""
    ratios = arrays["residual_ratio"]
    if ratios.ndim != 2 or ratios.shape[1] != 2 or not np.isfinite(ratios).all():
        raise ValueError("residual ratios must be finite [frame, vision/tactile] values")
    result: dict[str, object] = {"frames": len(ratios), "modalities": {}}
    for modality_index, modality in enumerate(("vision", "tactile")):
        values = ratios[:, modality_index]
        result["modalities"][modality] = {
            "residual_ratio_mean": float(values.mean()),
            "residual_ratio_min": float(values.min()),
            "residual_ratio_max": float(values.max()),
            "gamma_channel_temporal_std_mean": float(
                arrays["gamma"][:, modality_index].std(axis=0).mean()),
            "beta_channel_temporal_std_mean": float(
                arrays["beta"][:, modality_index].std(axis=0).mean()),
            "gamma_saturated_fraction_abs_ge_095": float(
                (np.abs(arrays["gamma"][:, modality_index]) >= 0.95).mean()),
            "beta_saturated_fraction_abs_ge_095": float(
                (np.abs(arrays["beta"][:, modality_index]) >= 0.95).mean()),
            "base_feature_norm_mean": float(np.linalg.norm(
                arrays["base_features"][:, modality_index], axis=-1).mean()),
            "residual_norm_mean": float(np.linalg.norm(
                arrays["modulated_features"][:, modality_index]
                - arrays["base_features"][:, modality_index], axis=-1).mean()),
            "feature_cosine_mean": float(arrays["feature_cosine"][:, modality_index].mean()),
        }
        if "dynamic_residual_ratio" in arrays:
            result["modalities"][modality]["dynamic_residual_ratio_mean"] = float(
                arrays["dynamic_residual_ratio"][:, modality_index].mean())
    if "attention" in arrays:
        tactile = arrays["attention"][..., 1]
        result["attention"] = {
            "tactile_mean": float(tactile.mean()),
            "tactile_min": float(tactile.min()),
            "tactile_max": float(tactile.max()),
            "tactile_mean_by_head": tactile.mean(axis=0).tolist(),
            "fusion_ratio_mean": float(arrays["fusion_ratio"].mean()),
        }
    if "fusion_gamma" in arrays:
        result["fusion_hypernet"] = {
            "residual_ratio_mean": float(arrays["fusion_hypernet_residual_ratio"].mean()),
            "gamma_saturated_fraction_abs_ge_095": float((np.abs(arrays["fusion_gamma"]) >= 0.95).mean()),
            "beta_saturated_fraction_abs_ge_095": float((np.abs(arrays["fusion_beta"]) >= 0.95).mean()),
        }
    return result


def temporal_modulation_ratio(arrays: dict[str, np.ndarray], alpha: float) -> np.ndarray:
    """Measure feature change versus each episode's mean gamma/beta, holding features fixed."""
    features = arrays["base_features"]
    dynamic = np.zeros(features.shape[:2], dtype=np.float32)
    for episode in np.unique(arrays["episode"]):
        selected = arrays["episode"] == episode
        gamma = arrays["gamma"][selected]
        beta = arrays["beta"][selected]
        difference = alpha * ((gamma - gamma.mean(axis=0)) * features[selected]
                              + beta - beta.mean(axis=0))
        dynamic[selected] = np.linalg.norm(difference, axis=-1) / np.maximum(
            np.linalg.norm(features[selected], axis=-1), 1e-8)
    return dynamic
