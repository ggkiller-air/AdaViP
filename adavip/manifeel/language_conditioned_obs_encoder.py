"""Six-sensor state compression followed by late language conditioning."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, SENSOR_KEYS, validate_shape_meta
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.model_getter import get_resnet


class LanguageConditionedObsEncoder(nn.Module):
    """Keep language outside z_t and return one window-level DP condition."""

    def __init__(self, shape_meta: dict, variant: str = "l1", n_obs_steps: int = 2,
                 token_dim: int = 256, image_feature_dim: int = 512,
                 use_group_norm: bool = True, head_seed: int = 9042,
                 rgb_model: nn.Module | None = None) -> None:
        super().__init__()
        validate_shape_meta(shape_meta)
        if variant not in ("l1", "l2") or min(n_obs_steps, token_dim, image_feature_dim) < 1:
            raise ValueError("Expected l1/l2 and positive dimensions")
        self.shape_meta = copy.deepcopy(shape_meta)
        self.variant, self.n_obs_steps, self.token_dim = variant, n_obs_steps, token_dim
        self.image_feature_dim = image_feature_dim
        backbone = get_resnet("resnet18", weights=None) if rgb_model is None else copy.deepcopy(rgb_model)
        if use_group_norm:
            backbone = replace_submodules(
                backbone, lambda module: isinstance(module, nn.BatchNorm2d),
                lambda module: nn.GroupNorm(math.gcd(32, module.num_features), module.num_features),
            )
        self.image_encoders = nn.ModuleDict({key: copy.deepcopy(backbone) for key in IMAGE_KEYS})
        self.temporal_projections = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(n_obs_steps * image_feature_dim, token_dim),
                               nn.LayerNorm(token_dim), nn.SiLU()) for key in IMAGE_KEYS
        })
        self.state_projection = nn.Sequential(
            nn.Linear(n_obs_steps * 7, token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
        )
        self.sensor_identity = nn.Parameter(torch.empty(6, token_dim))
        nn.init.normal_(self.sensor_identity, std=0.02)
        self.fusion = nn.Sequential(
            nn.Linear(6 * token_dim + 6, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim),
        )
        language_dim = shape_meta["obs"]["task_embedding"]["shape"][0]
        # Restore the common initialization stream before the U-Net is constructed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.language_head = nn.Sequential(
                nn.Linear(token_dim + language_dim if variant == "l1" else language_dim, token_dim),
                nn.SiLU(), nn.Linear(token_dim, token_dim if variant == "l1" else 2 * token_dim),
            )
            if variant == "l2":
                nn.init.zeros_(self.language_head[-1].weight)
                nn.init.zeros_(self.language_head[-1].bias)

    def encode_sensor_tokens(self, observations: dict[str, Tensor]) -> Tensor:
        """Encode only V1,V2,V3,T1,T2,P as [B,6,d], without language."""
        batch_size = observations["state"].shape[0]
        tokens = []
        for key in IMAGE_KEYS:
            images = observations[key]
            if tuple(images.shape) != (batch_size, self.n_obs_steps, *self.shape_meta["obs"][key]["shape"]):
                raise ValueError(f"Invalid {key} observation shape")
            features = self.image_encoders[key](images.flatten(0, 1))
            if features.shape != (batch_size * self.n_obs_steps, self.image_feature_dim):
                raise ValueError("Image encoder output dimension changed")
            tokens.append(self.temporal_projections[key](features.reshape(batch_size, -1)))
        state = observations["state"]
        if state.shape != (batch_size, self.n_obs_steps, 7):
            raise ValueError("state must be [B,To,7]")
        tokens.append(self.state_projection(state.flatten(1)))
        return torch.stack(tokens, dim=1) + self.sensor_identity.unsqueeze(0)

    def fuse_state_tokens(self, tokens: Tensor) -> Tensor:
        """Produce z_t from six sensor tokens and six full-input flags."""
        if tokens.ndim != 3 or tokens.shape[1:] != (6, self.token_dim):
            raise ValueError("State tokens must be [B,6,d]")
        flags = tokens.new_ones((len(tokens), 6))
        return self.fusion(torch.cat((tokens.flatten(1), flags), dim=-1))

    def language_embedding(self, observations: dict[str, Tensor]) -> Tensor:
        """Read the unchanged frozen embedding once per observation window."""
        language = observations["task_embedding"]
        expected = (len(observations["state"]), self.n_obs_steps,
                    self.shape_meta["obs"]["task_embedding"]["shape"][0])
        if language.shape != expected or not torch.equal(language, language[:, :1].expand_as(language)):
            raise ValueError("Language must be constant [B,To,language_dim]")
        return language[:, 0]

    def condition_state(self, state: Tensor, language: Tensor) -> Tensor:
        """Apply the sole architecture difference between L1 and L2."""
        if self.variant == "l1":
            return self.language_head(torch.cat((state, language), dim=-1))
        gamma, beta = self.language_head(language).chunk(2, dim=-1)
        return (1 + gamma) * state + beta

    def forward(self, observations: dict[str, Tensor], presence: Tensor | None = None) -> Tensor:
        """Return [B,d] through the identical training and inference path."""
        if set(observations) != set(SENSOR_KEYS):
            raise ValueError("Expected all six sensors and task_embedding")
        if presence is not None and (presence.dtype != torch.bool
                                     or presence.shape != (len(observations["state"]), 7)
                                     or not presence.all()):
            raise ValueError("Language comparison requires full input; dropout is disabled")
        state = self.fuse_state_tokens(self.encode_sensor_tokens(observations))
        return self.condition_state(state, self.language_embedding(observations))
