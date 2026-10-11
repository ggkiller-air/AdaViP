"""Controlled open-loop action sensitivity probes for ManiFeel policies."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur
import torch.nn.functional as functional

from adavip.manifeel.hypernet_probe import select_episode_frames


def observation_windows(episode_ends: np.ndarray, episodes: list[int], samples: int,
                         n_obs_steps: int, lag: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build oldest-to-newest histories with episode-local initial-frame padding."""
    if n_obs_steps <= 0 or lag < 0:
        raise ValueError("observation steps must be positive and lag nonnegative")
    indices, identifiers, steps = select_episode_frames(episode_ends, episodes, samples)
    starts = indices - steps
    offsets = np.arange(n_obs_steps - 1, -1, -1)
    windows = starts[:, None] + np.maximum(steps[:, None] - lag - offsets, 0)
    return windows, identifiers, steps


def perturb_observations(observations: dict[str, torch.Tensor],
                         delayed: dict[str, torch.Tensor], key: str,
                         mode: str) -> dict[str, torch.Tensor]:
    """Change one modality while preserving all other observation histories."""
    if key not in observations:
        raise ValueError(f"unknown observation key: {key}")
    changed = dict(observations)
    if mode == "delay":
        changed[key] = delayed[key]
    elif mode == "blur":
        value = observations[key]
        if value.ndim != 5:
            raise ValueError("blur requires [batch, history, channel, height, width] RGB")
        changed[key] = gaussian_blur(value.flatten(0, 1), [15, 15], [5.0, 5.0]).reshape_as(value)
    else:
        raise ValueError(f"unsupported perturbation: {mode}")
    return changed


def seeded_prediction(policy: Any, observations: dict[str, torch.Tensor],
                      seed: int) -> torch.Tensor:
    """Replay all diffusion sampling randomness for a paired observation comparison."""
    device = next(iter(observations.values())).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.inference_mode():
            action = policy.predict_action(observations)["action"]
        if not torch.isfinite(action).all():
            raise ValueError("policy predicted nonfinite actions")
        return action.detach().clone()


def capture_hypernet_output(policy: Any, observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """Read the generated pre-tanh modulation for the supplied observation history."""
    outputs = []

    def capture(module: Any, inputs: tuple, output: torch.Tensor) -> None:
        outputs.append(output.detach().clone())

    handle = policy.obs_encoder.hypernet.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            normalized = policy.normalizer.normalize(observations)
            flattened = {key: value.reshape(-1, *value.shape[2:])
                         for key, value in normalized.items()}
            policy.obs_encoder(flattened)
    finally:
        handle.remove()
    if len(outputs) != 1:
        raise ValueError("expected one HyperNet call per observation history")
    return outputs[0]


@contextmanager
def override_hypernet(encoder: Any, replacement: torch.Tensor | None,
                      environment_selector: Callable[[], np.ndarray] | None = None,
                      n_obs_steps: int = 1) -> Iterator[None]:
    """Override only generated modulation; keep current features, alpha and fusion intact."""
    if replacement is None:
        yield
        return

    def replace(module: Any, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if replacement.shape[-1] != output.shape[-1] or replacement.shape[0] not in (1, output.shape[0]):
            raise ValueError("replacement modulation does not match encoder frames")
        fixed = replacement.to(output).expand_as(output)
        if environment_selector is None:
            return fixed
        mask = selected_history_frames(output, environment_selector, n_obs_steps)
        return torch.where(mask[:, None], fixed, output)

    handle = encoder.hypernet.register_forward_hook(replace)
    try:
        yield
    finally:
        handle.remove()


def selected_history_frames(value: torch.Tensor, selector: Callable[[], np.ndarray],
                            n_obs_steps: int) -> torch.Tensor:
    """Map causal per-environment selections to the encoder's flattened history axis."""
    selected = selector()
    if selected.ndim != 1 or selected.dtype != np.bool_ or n_obs_steps <= 0:
        raise ValueError("environment selector must return a one-dimensional boolean mask")
    if len(selected) * n_obs_steps != value.shape[0]:
        raise ValueError("selected environments do not match encoder history layout")
    return torch.as_tensor(selected, device=value.device).repeat_interleave(n_obs_steps)


@contextmanager
def override_hypernet_context(encoder: Any, reference_context: torch.Tensor, modality: str,
                              environment_selector: Callable[[], np.ndarray] | None = None,
                              n_obs_steps: int = 1) -> Iterator[None]:
    """Freeze one HyperNet context slice while preserving the real modality and state inputs."""
    width = encoder.hypernet[0].in_features
    feature_width = (width - 7) // 2
    slices = {"vision": (0, feature_width), "tactile": (feature_width, 2 * feature_width),
              "state": (2 * feature_width, width)}
    if modality not in slices or reference_context.shape != (1, width) or width != 2 * feature_width + 7:
        raise ValueError("context intervention requires one matching vision/tactile/state reference")
    start, stop = slices[modality]

    def replace(module: Any, inputs: tuple) -> tuple:
        context = inputs[0]
        mask = selected_history_frames(context, environment_selector, n_obs_steps) if environment_selector else None
        changed = context.clone()
        fixed = reference_context.to(context).expand_as(context)[:, start:stop]
        changed[:, start:stop] = fixed if mask is None else torch.where(mask[:, None], fixed, context[:, start:stop])
        return (changed, *inputs[1:])

    handle = encoder.hypernet.register_forward_pre_hook(replace)
    try:
        yield
    finally:
        handle.remove()


def normalized_action_change(original: torch.Tensor, changed: torch.Tensor,
                              action_scale: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-history RMS action changes in checkpoint-normalized coordinates."""
    difference = (changed - original) * action_scale
    if difference.ndim != 3 or difference.shape[-1] < 6:
        raise ValueError("expected action chunks with at least six action dimensions")
    return {"all": difference.square().mean(dim=(1, 2)).sqrt(),
            "translation": difference[..., :3].square().mean(dim=(1, 2)).sqrt(),
            "rotation": difference[..., 3:6].square().mean(dim=(1, 2)).sqrt()}


def patch_blur_observations(observations: dict[str, torch.Tensor], key: str,
                            boxes: list[tuple[int, int, int, int]]) -> dict[str, torch.Tensor]:
    """Blur one spatial region in both history frames for each paired batch item."""
    value = observations[key]
    if value.ndim != 5 or value.shape[0] != 1 or not boxes:
        raise ValueError("patch probes require one RGB observation history and nonempty boxes")
    changed = {name: tensor.expand(len(boxes), *tensor.shape[1:])
               for name, tensor in observations.items()}
    image = value.expand(len(boxes), *value.shape[1:]).clone()
    blurred = gaussian_blur(value.flatten(0, 1), [31, 31], [10.0, 10.0]).reshape_as(value)
    for box_index, (top, bottom, left, right) in enumerate(boxes):
        if not 0 <= top < bottom <= value.shape[-2] or not 0 <= left < right <= value.shape[-1]:
            raise ValueError("patch box is outside the image")
        image[box_index, ..., top:bottom, left:right] = blurred[0, ..., top:bottom, left:right]
    changed[key] = image
    return changed


def reconstruct_condition(encoder: Any, features: torch.Tensor, gamma: torch.Tensor,
                           beta: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Rebuild the exact encoder output from cached raw features and chosen modulation."""
    modulated = features + encoder.alpha * (gamma * features + beta)
    vision, tactile = modulated[:, 0], modulated[:, 1]
    if encoder.use_fusion:
        query = vision.unsqueeze(1)
        tokens = torch.stack((vision, tactile), dim=1)
        attended, _ = encoder.cross_attention(query, tokens, tokens, need_weights=False)
        vision = vision + encoder.fusion_gate.tanh() * attended.squeeze(1)
        if encoder.use_fusion_hypernet:
            width = vision.shape[-1]
            context = torch.cat((functional.layer_norm(vision, (width,)),
                                 functional.layer_norm(tactile, (width,)), state), dim=-1)
            fusion_gamma, fusion_beta = encoder.fusion_hypernet(context).tanh().chunk(2, dim=-1)
            vision = vision + encoder.fusion_alpha * (fusion_gamma * vision + fusion_beta)
    encoded = {encoder.vision_key: vision, encoder.tactile_key: tactile}
    return torch.cat([encoded[key] for key in encoder.rgb_keys] + [state], dim=-1)


def seeded_condition_prediction(policy: Any, encoded: torch.Tensor, seed: int) -> torch.Tensor:
    """Replay the saved action sampler on exact cached encoder conditions."""
    if not policy.obs_as_global_cond or encoded.shape[1] != policy.n_obs_steps:
        raise ValueError("cached action probes require the saved global-condition history layout")

    class ConditionPolicy:
        """Delegate the original scheduler and action slicing without image re-encoding."""

        def predict_action(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Follow the upstream global-condition sampling path exactly."""
            condition = observations["encoded"]
            batch_size = condition.shape[0]
            data = torch.zeros((batch_size, policy.horizon, policy.action_dim),
                                dtype=condition.dtype, device=condition.device)
            mask = torch.zeros_like(data, dtype=torch.bool)
            sampled = policy.conditional_sample(data, mask, local_cond=None,
                global_cond=condition.reshape(batch_size, -1), **policy.kwargs)
            actions = policy.normalizer["action"].unnormalize(sampled[..., :policy.action_dim])
            start = policy.n_obs_steps - 1
            return {"action": actions[:, start:start + policy.n_action_steps]}

    return seeded_prediction(ConditionPolicy(), {"encoded": encoded}, seed)
