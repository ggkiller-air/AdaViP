"""Load Stage0 inference weights and adapt complete simulator observations."""

from __future__ import annotations

from pathlib import Path

import dill
import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from adavip.manifeel.action_adapter import truncate_action_for_task
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, validate_shape_meta


def load_evaluation_policy(
    checkpoint: Path, device: str = "cpu", weights: str = "ema",
) -> tuple[torch.nn.Module, DictConfig, dict]:
    """Restore a trusted checkpoint's policy and normalizer, without an optimizer."""
    if weights not in ("ema", "raw"):
        raise ValueError("weights must be ema or raw")
    payload = torch.load(str(checkpoint), map_location="cpu", pickle_module=dill, mmap=True)
    cfg = payload["cfg"]
    validate_shape_meta(cfg.shape_meta)
    state_key = "ema_model" if weights == "ema" else "model"
    if state_key not in payload["state_dicts"]:
        raise ValueError(f"Checkpoint has no {state_key}; refusing a silent weights fallback")
    state = payload["state_dicts"][state_key]
    if not any(key.startswith("normalizer.") for key in state):
        raise ValueError("Checkpoint policy has no saved normalizer")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state, strict=True)
    policy.to(device)
    policy.requires_grad_(False)
    policy.eval()
    metadata = {key: dill.loads(payload["pickles"][key])
                for key in ("epoch", "global_step", "batch_in_epoch")
                if key in payload.get("pickles", {})}
    metadata["weights"] = weights
    return policy, cfg, metadata


class FullSensorEvaluationPolicy:
    """Inject constant language and truncate actions without re-normalizing inputs."""

    def __init__(self, policy: torch.nn.Module, shape_meta: dict,
                 task_embedding: np.ndarray, action_dim: int) -> None:
        validate_shape_meta(shape_meta)
        self.policy = policy
        self.shape_meta = shape_meta
        self.action_dim = action_dim
        self.task_embedding = torch.as_tensor(task_embedding, device=policy.device, dtype=policy.dtype)
        if tuple(self.task_embedding.shape) != tuple(shape_meta["obs"]["task_embedding"]["shape"]):
            raise ValueError("Task embedding does not match checkpoint language shape")

    @property
    def device(self) -> torch.device:
        """Expose the policy's inference device to the simulator runner."""
        return self.policy.device

    @property
    def dtype(self) -> torch.dtype:
        """Expose the policy's floating point dtype to the simulator runner."""
        return self.policy.dtype

    def reset(self) -> None:
        """Reset the underlying action policy before a rollout."""
        self.policy.reset()

    def predict_action(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Require every trained sensor, preserving image ranges and history order."""
        observations = dict(observations)
        state = observations["state"]
        batch_size = state.shape[0]
        history = self.policy.n_obs_steps
        for key in (*IMAGE_KEYS, "state"):
            expected = (batch_size, history, *self.shape_meta["obs"][key]["shape"])
            if key not in observations or tuple(observations[key].shape) != expected:
                actual = tuple(observations[key].shape) if key in observations else "missing"
                raise ValueError(f"{key}: expected {expected}, got {actual}")
        observations["task_embedding"] = self.task_embedding.view(1, 1, -1).expand(batch_size, history, -1)
        result = self.policy.predict_action(observations)
        result["action"] = truncate_action_for_task(result["action"], self.action_dim)
        return result
