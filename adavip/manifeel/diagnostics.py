"""Record ManiFeel rollout inputs, control signals, and fusion activations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ABLATIONS = ("none", "no_film", "no_attention", "no_tactile")


def _array(value: Any) -> np.ndarray:
    """Move an observation or simulator tensor to host memory."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_image(image: np.ndarray, path: Path) -> None:
    """Write a channel-first, float-[0,1] RGB image for visual inspection."""
    rgb = np.moveaxis(image, 0, -1)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot write diagnostic image: {path}")


class RolloutRecorder:
    """Wrap the underlying per-step simulator without changing runner behavior."""

    def __init__(self, env: Any, output_dir: Path) -> None:
        self.env = env
        self.output_dir = output_dir
        self.actions: list[np.ndarray] = []
        self.success: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.task_states: dict[str, list[np.ndarray]] = {
            key: [] for key in ("socket_pos", "plug_pos", "fingertip_midpoint_pos")
        }
        self.initial_state: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, **kwargs: Any) -> Any:
        """Capture the rollout initial state, not the final cleanup reset."""
        observation = self.env.reset(**kwargs)
        if self.initial_state is None and "state" in observation:
            self.initial_state = _array(observation["state"]).copy()
        return observation

    def step(self, action: np.ndarray) -> tuple[Any, Any, Any, Any]:
        """Record each action as actually executed by the simulator."""
        observation, reward, done, info = self.env.step(action)
        self.actions.append(_array(action).copy())
        if "success" in info:
            self.success.append(_array(info["success"]).copy())
        if "state" in observation:
            self.states.append(_array(observation["state"]).copy())
        task = getattr(self.env, "envs", None)
        if task is not None:
            for key, values in self.task_states.items():
                if hasattr(task, key):
                    values.append(_array(getattr(task, key)).copy())
        return observation, reward, done, info

    def save(self) -> dict[str, Any]:
        """Persist compact per-step traces and return per-environment outcomes."""
        arrays: dict[str, np.ndarray] = {
            "executed_actions": np.stack(self.actions),
        }
        if self.initial_state is not None:
            arrays["initial_state"] = self.initial_state
        if self.states:
            arrays["ee_pose"] = np.stack(self.states)
        if self.success:
            arrays["success_by_step"] = np.stack(self.success)
        for key, values in self.task_states.items():
            if values:
                arrays[key] = np.stack(values)
        np.savez_compressed(self.output_dir / "rollout_trace.npz", **arrays)
        result: dict[str, Any] = {"executed_steps": len(self.actions)}
        if self.success:
            per_env = np.any(arrays["success_by_step"].reshape(len(self.success),
                arrays["success_by_step"].shape[1], -1) > 0.5, axis=(0, 2))
            result["success_by_env"] = per_env.tolist()
            result["success_rate"] = float(np.mean(per_env))
        if "plug_pos" in arrays and "socket_pos" in arrays:
            distances = np.linalg.norm(arrays["plug_pos"] - arrays["socket_pos"], axis=-1)
            result["minimum_plug_socket_distance_by_env"] = distances.min(axis=0).tolist()
            result["final_plug_socket_distance_by_env"] = distances[-1].tolist()
        return result


class PolicyRecorder:
    """Capture policy-side inputs, intermediate features, and predicted chunks."""

    def __init__(self, policy: Any, output_dir: Path, ablation: str = "none",
                 canonicalize_quaternion: bool = False) -> None:
        if ablation not in ABLATIONS:
            raise ValueError(f"unsupported ablation: {ablation}")
        self.policy = policy
        self.output_dir = output_dir
        self.chunks: list[np.ndarray] = []
        self.input_stats: list[dict[str, Any]] = []
        self.feature_stats: dict[str, list[dict[str, float]]] = {}
        self.quaternion_sign_corrections = 0
        self.canonicalize_quaternion = canonicalize_quaternion
        self._hooks: list[Any] = []
        encoder = policy.obs_encoder
        if ablation != "none" and not hasattr(encoder, "film"):
            raise ValueError("FiLM ablations require FilmFusionObsEncoder")
        if ablation == "no_film":
            self._hooks.append(encoder.film.register_forward_hook(
                lambda module, inputs, output: inputs[0]))
        elif ablation == "no_attention":
            self._hooks.append(encoder.cross_attention.register_forward_hook(
                lambda module, inputs, output: (torch.zeros_like(output[0]), output[1])))
        elif ablation == "no_tactile":
            self._hooks.append(encoder.sparsh_encoder.register_forward_hook(
                lambda module, inputs, output: torch.zeros_like(output)))

        modules = {"condition": encoder}
        if hasattr(encoder, "clip_encoder"):
            modules.update({
                "clip": encoder.clip_encoder,
                "sparsh": encoder.sparsh_encoder,
                "vision_projection": encoder.vision_projection,
                "tactile_projection": encoder.tactile_projection,
                "cross_attention": encoder.cross_attention,
            })
            if hasattr(encoder, "hypernet"):
                modules["hypernet"] = encoder.hypernet
        else:
            modules.update({f"resnet_{key}": module
                for key, module in encoder.key_model_map.items()})
        for name, module in modules.items():
            self.feature_stats[name] = []
            self._hooks.append(module.register_forward_hook(self._feature_hook(name)))

        original_predict_action = policy.predict_action

        def traced_predict_action(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            index = len(self.chunks)
            if self.canonicalize_quaternion:
                state = observation["state"].clone()
                negative = state[..., 3] < 0
                self.quaternion_sign_corrections += int(negative.sum().item())
                state[..., 3:7] = torch.where(negative.unsqueeze(-1),
                    -state[..., 3:7], state[..., 3:7])
                observation = {**observation, "state": state}
            self._record_input(observation, index)
            result = original_predict_action(observation)
            self.chunks.append(_array(result["action"]).copy())
            return result

        policy.predict_action = traced_predict_action
        self._original_predict_action = original_predict_action
        self.ablation = ablation

    def _feature_hook(self, name: str):
        """Build a lightweight, per-inference activation summary hook."""
        def record(module: Any, inputs: Any, output: Any) -> None:
            if name == "hypernet":
                output = torch.cat((output.gamma_vision, output.beta_vision,
                    output.gamma_tactile, output.beta_tactile), dim=-1)
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                return
            flat = value.detach().float().flatten(start_dim=1)
            self.feature_stats[name].append({
                "mean": float(flat.mean().item()),
                "std": float(flat.std(unbiased=False).item()),
                "norm": float(flat.norm(dim=1).mean().item()),
                "nonfinite": int((~torch.isfinite(flat)).sum().item()),
            })
        return record

    def _record_input(self, observation: dict[str, torch.Tensor], index: int) -> None:
        record: dict[str, Any] = {"chunk": index}
        for key, value in observation.items():
            record[key] = {
                "shape": list(value.shape),
                "min": float(value.min().item()),
                "max": float(value.max().item()),
                "mean": float(value.float().mean().item()),
            }
            if "pair" in key and value.shape[2] == 6:
                record[key]["mean_absolute_frame_difference"] = float(
                    (value[:, :, :3] - value[:, :, 3:]).abs().mean().item())
            if index in (0, 12, 60) and value.ndim == 5 and value.shape[2] in (3, 6):
                self.output_dir.joinpath("images").mkdir(exist_ok=True)
                image = _array(value[0, -1])
                for channel in range(0, image.shape[0], 3):
                    _save_image(image[channel : channel + 3],
                        self.output_dir / "images" / f"chunk{index:03d}_{key}_{channel // 3}.png")
        self.input_stats.append(record)

    def save(self) -> dict[str, Any]:
        """Save the policy trace and remove any temporary observation hooks."""
        self.policy.predict_action = self._original_predict_action
        for hook in self._hooks:
            hook.remove()
        np.savez_compressed(self.output_dir / "policy_chunks.npz", action_chunks=np.stack(self.chunks))
        (self.output_dir / "input_stats.json").write_text(json.dumps(self.input_stats, indent=2) + "\n")
        (self.output_dir / "feature_stats.json").write_text(json.dumps(self.feature_stats, indent=2) + "\n")
        result: dict[str, Any] = {"ablation": self.ablation, "predicted_chunks": len(self.chunks)}
        result["quaternion_sign_corrections"] = self.quaternion_sign_corrections
        if hasattr(self.policy.obs_encoder, "film"):
            result["film_alpha"] = float(self.policy.obs_encoder.film.alpha.detach().cpu().item())
        result["action_training_stats"] = {
            name: _array(value).tolist()
            for name, value in self.policy.normalizer["action"].get_input_stats().items()
        }
        action = np.concatenate(self.chunks, axis=0)
        result["mean_absolute_action_by_dim"] = np.mean(np.abs(action), axis=(0, 1)).tolist()
        result["nonfinite_actions"] = int((~np.isfinite(action)).sum())
        return result


def attach_diagnostics(runner: Any, policy: Any, output_dir: Path,
                       ablation: str = "none", canonicalize_quaternion: bool = False
                       ) -> tuple[RolloutRecorder, PolicyRecorder]:
    """Install recording at the simulator step and policy inference boundaries."""
    underlying = getattr(runner, "_runner", runner)
    multistep = underlying.env.env if hasattr(runner, "_runner") else underlying.env
    video = multistep.env
    recorder = RolloutRecorder(video.env, output_dir)
    video.env = recorder
    return recorder, PolicyRecorder(policy, output_dir, ablation=ablation,
        canonicalize_quaternion=canonicalize_quaternion)
