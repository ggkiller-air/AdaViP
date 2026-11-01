"""Independent four-image, proprio and language encoder with explicit masks."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from adavip.openvitac.schema import IMAGE_KEYS, SENSOR_KEYS, STATE_DIM, validate_shape_meta
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.model_getter import get_resnet


class OpenViTacObsEncoder(nn.Module):
    """Preserve the UViP Stage-0 temporal compression and Concat+MLP design."""

    def __init__(self, shape_meta, n_obs_steps=2, token_dim=256, image_feature_dim=512,
                 rgb_model=None, use_group_norm=True):
        super().__init__()
        validate_shape_meta(shape_meta)
        self.shape_meta = copy.deepcopy(shape_meta)
        self.n_obs_steps, self.token_dim, self.image_feature_dim = n_obs_steps, token_dim, image_feature_dim
        backbone = get_resnet("resnet18", weights=None) if rgb_model is None else copy.deepcopy(rgb_model)
        if use_group_norm:
            backbone = replace_submodules(backbone, lambda m: isinstance(m, nn.BatchNorm2d),
                                          lambda m: nn.GroupNorm(math.gcd(32, m.num_features), m.num_features))
        self.image_encoders = nn.ModuleDict({key: copy.deepcopy(backbone) for key in IMAGE_KEYS})
        self.temporal_projections = nn.ModuleDict({
            key: nn.Sequential(nn.Linear(n_obs_steps * image_feature_dim, token_dim),
                               nn.LayerNorm(token_dim), nn.SiLU()) for key in IMAGE_KEYS})
        self.state_projection = nn.Sequential(nn.Linear(n_obs_steps * STATE_DIM, token_dim),
                                             nn.LayerNorm(token_dim), nn.SiLU())
        language_dim = shape_meta["obs"]["task_embedding"]["shape"][0]
        self.language_projection = nn.Sequential(nn.Linear(language_dim, token_dim),
                                                nn.LayerNorm(token_dim), nn.SiLU())
        self.sensor_identity = nn.Parameter(torch.empty(len(SENSOR_KEYS), token_dim))
        nn.init.normal_(self.sensor_identity, std=.02)
        self.missing_tokens = nn.Parameter(torch.zeros(len(SENSOR_KEYS), token_dim))
        self.fusion = nn.Sequential(nn.Linear(len(SENSOR_KEYS) * (token_dim + 1), token_dim),
                                    nn.SiLU(), nn.Linear(token_dim, token_dim))

    def encode_tokens(self, observations: dict) -> torch.Tensor:
        """Return [B,6,d] in the fixed V1,V2,T1,T2,P,L order."""
        if set(observations) != set(SENSOR_KEYS):
            raise ValueError("Expected exactly two cameras, two tactile streams, P and L")
        batch = len(observations["state"])
        tokens = []
        for key in IMAGE_KEYS:
            value = observations[key]
            if tuple(value.shape) != (batch, self.n_obs_steps, *self.shape_meta["obs"][key]["shape"]):
                raise ValueError("Invalid image shape: " + key)
            features = self.image_encoders[key](value.flatten(0, 1))
            if features.shape != (batch * self.n_obs_steps, self.image_feature_dim):
                raise ValueError("Unexpected backbone feature dimension")
            tokens.append(self.temporal_projections[key](features.reshape(batch, -1)))
        state, language = observations["state"], observations["task_embedding"]
        if state.shape != (batch, self.n_obs_steps, STATE_DIM):
            raise ValueError("Expected [B,To,8] measured proprio")
        if tuple(language.shape) != (batch, self.n_obs_steps, self.shape_meta["obs"]["task_embedding"]["shape"][0]):
            raise ValueError("Unexpected language shape")
        if not torch.equal(language, language[:, :1].expand_as(language)):
            raise ValueError("Instruction must be constant within a causal window")
        tokens.extend((self.state_projection(state.flatten(1)), self.language_projection(language[:, 0])))
        return torch.stack(tokens, dim=1) + self.sensor_identity[None]

    def masked_tokens(self, tokens: torch.Tensor, presence=None) -> tuple:
        """Remove information before the first fusion, including sensor identity consistently."""
        if tuple(tokens.shape[1:]) != (len(SENSOR_KEYS), self.token_dim):
            raise ValueError("Expected six sensor tokens")
        if presence is None:
            presence = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if presence.dtype != torch.bool or presence.shape != tokens.shape[:2]:
            raise ValueError("Expected boolean [B,6] presence")
        missing = (self.missing_tokens + self.sensor_identity)[None]
        return torch.where(presence[..., None], tokens, missing), presence

    def fuse_tokens(self, tokens: torch.Tensor, presence=None) -> torch.Tensor:
        """Keep the 256D DP conditioning width while changing only input cardinality."""
        masked, flags = self.masked_tokens(tokens, presence)
        return self.fusion(torch.cat((masked.flatten(1), flags.to(tokens.dtype)), dim=-1))

    def forward(self, observations: dict, presence=None) -> torch.Tensor:
        return self.fuse_tokens(self.encode_tokens(observations), presence)
