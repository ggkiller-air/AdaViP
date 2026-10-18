"""Window-level seven-token encoding and simple masked Stage-0 fusion."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, SENSOR_KEYS, validate_shape_meta
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.model_getter import get_resnet


class FullSensorObsEncoder(nn.Module):
    """Compress each sensor's history before any cross-sensor information mixing.

    Images arrive already normalized once to [-1,1]. No ImageNet normalization,
    random crop, HyperNet, attention or auxiliary representation loss is applied.
    """

    def __init__(
        self,
        shape_meta: dict,
        n_obs_steps: int = 2,
        token_dim: int = 256,
        image_feature_dim: int = 512,
        rgb_model: nn.Module | None = None,
        use_group_norm: bool = True,
    ) -> None:
        super().__init__()
        validate_shape_meta(shape_meta)
        if min(n_obs_steps, token_dim, image_feature_dim) < 1:
            raise ValueError("Observation history and feature dimensions must be positive")
        self.shape_meta = copy.deepcopy(shape_meta)
        self.n_obs_steps = n_obs_steps
        self.token_dim = token_dim
        self.image_feature_dim = image_feature_dim
        backbone = get_resnet("resnet18", weights=None) if rgb_model is None else copy.deepcopy(rgb_model)
        if use_group_norm:
            backbone = replace_submodules(
                root_module=backbone,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(math.gcd(32, module.num_features), module.num_features),
            )
        self.image_encoders = nn.ModuleDict({key: copy.deepcopy(backbone) for key in IMAGE_KEYS})
        self.temporal_projections = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(n_obs_steps * image_feature_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
            for key in IMAGE_KEYS
        })
        self.state_projection = nn.Sequential(nn.Linear(n_obs_steps * 7, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
        language_dim = shape_meta["obs"]["task_embedding"]["shape"][0]
        self.language_projection = nn.Sequential(nn.Linear(language_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
        self.sensor_identity = nn.Parameter(torch.empty(7, token_dim))
        nn.init.normal_(self.sensor_identity, std=0.02)
        self.missing_tokens = nn.Parameter(torch.zeros(7, token_dim))
        self.fusion = nn.Sequential(
            nn.Linear(7 * token_dim + 7, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim),
        )

    def encode_tokens(self, observations: dict[str, Tensor]) -> Tensor:
        """Return [B,7,d] in the canonical V1,V2,V3,T1,T2,P,L order."""
        if set(observations) != set(SENSOR_KEYS):
            raise ValueError(f"Expected exactly seven observation keys: {SENSOR_KEYS}")
        batch_size = observations["state"].shape[0]
        tokens = []
        for key in IMAGE_KEYS:
            images = observations[key]
            expected = (batch_size, self.n_obs_steps, *self.shape_meta["obs"][key]["shape"])
            if tuple(images.shape) != expected:
                raise ValueError(f"{key}: expected {expected}, got {tuple(images.shape)}")
            features = self.image_encoders[key](images.flatten(0, 1))
            if tuple(features.shape) != (batch_size * self.n_obs_steps, self.image_feature_dim):
                raise ValueError(f"{key}: image backbone must output [B*To,{self.image_feature_dim}]")
            tokens.append(self.temporal_projections[key](features.reshape(batch_size, -1)))
        state = observations["state"]
        if tuple(state.shape) != (batch_size, self.n_obs_steps, 7):
            raise ValueError("state must have shape [B,To,7]")
        tokens.append(self.state_projection(state.flatten(1)))
        language = observations["task_embedding"]
        language_dim = self.shape_meta["obs"]["task_embedding"]["shape"][0]
        if tuple(language.shape) != (batch_size, self.n_obs_steps, language_dim):
            raise ValueError("task_embedding must have shape [B,To,language_dim]")
        if not torch.equal(language, language[:, :1].expand_as(language)):
            raise ValueError("The task instruction must stay constant within an observation window")
        tokens.append(self.language_projection(language[:, 0]))
        return torch.stack(tokens, dim=1) + self.sensor_identity.unsqueeze(0)

    def fuse_tokens(self, tokens: Tensor, presence: Tensor | None = None) -> Tensor:
        """Replace removed sensor tokens before fusion and append explicit presence flags."""
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (7, self.token_dim):
            raise ValueError("tokens must have shape [B,7,d]")
        if presence is None:
            presence = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if presence.dtype != torch.bool or tuple(presence.shape) != tuple(tokens.shape[:2]):
            raise ValueError("presence must be a boolean [B,7] tensor")
        presence = presence.to(device=tokens.device)
        missing = (self.missing_tokens + self.sensor_identity).unsqueeze(0)
        masked = torch.where(presence.unsqueeze(-1), tokens, missing)
        return self.fusion(torch.cat((masked.flatten(1), presence.to(tokens.dtype)), dim=-1))

    def forward(self, observations: dict[str, Tensor], presence: Tensor | None = None) -> Tensor:
        """Return one [B,d] conditioning vector for the complete history window."""
        return self.fuse_tokens(self.encode_tokens(observations), presence)
