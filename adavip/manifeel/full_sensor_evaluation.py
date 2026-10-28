"""Load Stage0/Stage2 inference weights and adapt simulator observations."""

from __future__ import annotations

import inspect
from pathlib import Path

import dill
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from adavip.manifeel.action_adapter import truncate_action_for_task
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, validate_shape_meta


STAGE2_CONFIG = Path(__file__).resolve().parents[2] / "configs/manifeel/eval_stage2_t2.yaml"


def stage2_variant(state: dict, contract: dict) -> str:
    """Identify legacy checkpoints by saved fusion tensors and validate metadata."""
    mlp = "final_fusion.0.weight" in state
    transformer = "cross_attention.in_proj_weight" in state
    if mlp == transformer:
        raise ValueError("Stage2 checkpoint must contain exactly one known fusion architecture")
    variant = "stage2_mlp_fusion" if mlp else "stage2_transformer"
    if contract.get("variant", variant) != variant:
        raise ValueError("Stage2 variant metadata disagrees with fusion tensors")
    return variant


def load_evaluation_policy(
    checkpoint: Path, device: str = "cpu", weights: str = "ema",
    config_path: Path | None = None,
) -> tuple[torch.nn.Module, DictConfig, dict]:
    """Restore a trusted checkpoint's policy and normalizer, without an optimizer."""
    if weights not in ("ema", "raw"):
        raise ValueError("weights must be ema or raw")
    load_options = {"map_location": "cpu", "pickle_module": dill}
    if "mmap" in inspect.signature(torch.load).parameters:
        load_options["mmap"] = True
    payload = torch.load(str(checkpoint), **load_options)
    is_stage2 = "contract" in payload and "model" in payload and "state_dicts" not in payload
    if is_stage2:
        state_key = "ema" if weights == "ema" else "model"
        if state_key not in payload:
            raise ValueError(f"Checkpoint has no {state_key}; refusing a silent weights fallback")
        state = payload[state_key]
        variant = stage2_variant(state, payload["contract"])
        if config_path:
            cfg = OmegaConf.load(config_path)
        elif "cfg" in payload:
            cfg = OmegaConf.create(payload["cfg"])
        else:
            cfg = OmegaConf.load(STAGE2_CONFIG)
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.policy._target_ = (
            "adavip.manifeel.stage2_mlp_policy.Stage2MlpPolicy" if variant == "stage2_mlp_fusion"
            else "adavip.manifeel.stage2_policy.Stage2Policy")
        strategy = payload["contract"].get("encoder_strategy", "s0")
        if variant == "stage2_transformer" and strategy != "s0":
            if cfg.policy.get("encoder_strategy", strategy) != strategy:
                raise ValueError("Checkpoint encoder strategy disagrees with architecture")
            cfg.policy.encoder_strategy = strategy
        metadata = {key: payload[key] for key in ("epoch", "global_step", "batch_cursor", "stage")
                    if key in payload}
        metadata.update(variant=variant, contract_hash=payload.get("contract_hash"))
        metadata["encoder_strategy"] = strategy
    else:
        if config_path:
            raise ValueError("Stage0 checkpoints already contain their configuration")
        cfg = payload["cfg"]
        state_key = "ema_model" if weights == "ema" else "model"
        if state_key not in payload["state_dicts"]:
            raise ValueError(f"Checkpoint has no {state_key}; refusing a silent weights fallback")
        state = payload["state_dicts"][state_key]
        metadata = {key: dill.loads(payload["pickles"][key])
                    for key in ("epoch", "global_step", "batch_in_epoch")
                    if key in payload.get("pickles", {})}
    validate_shape_meta(cfg.shape_meta)
    if not any(key.startswith("normalizer.") for key in state):
        raise ValueError("Checkpoint policy has no saved normalizer")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(state, strict=True)
    policy.to(device)
    policy.requires_grad_(False)
    policy.eval()
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
