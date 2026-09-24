"""Piper-style FiLM fusion observation encoder for ManiFeel Power Plug."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


VISION_DIM = 1024
TACTILE_DIM = 768
STATE_DIM = 7
HYPERNET_INPUT_DIM = VISION_DIM + TACTILE_DIM + STATE_DIM
HYPERNET_OUTPUT_DIM = 2 * VISION_DIM + 2 * TACTILE_DIM
OUTPUT_DIM = 512 + STATE_DIM


@dataclass(frozen=True)
class FilmParameters:
    """Per-sample affine deltas generated for both representation streams."""

    gamma_vision: Tensor
    beta_vision: Tensor
    gamma_tactile: Tensor
    beta_tactile: Tensor


class LightweightHyperNet(nn.Module):
    """Generate CLIP and Sparsh FiLM parameters from the full observation."""

    output_dim = HYPERNET_OUTPUT_DIM

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HYPERNET_INPUT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, observation: Tensor) -> FilmParameters:
        if observation.ndim != 2 or observation.shape[-1] != HYPERNET_INPUT_DIM:
            raise ValueError(
                f"observation must be [N,{HYPERNET_INPUT_DIM}], got {observation.shape}"
            )
        output = self.network(observation)
        return FilmParameters(
            *torch.split(output, (VISION_DIM, VISION_DIM, TACTILE_DIM, TACTILE_DIM), dim=-1)
        )


class LearnableResidualFiLM(nn.Module):
    """Apply FiLM deltas through Piper's positive learnable residual scale."""

    def __init__(self, alpha_init: float = 0.01) -> None:
        super().__init__()
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive")
        alpha = torch.tensor(float(alpha_init))
        self.residual_alpha = nn.Parameter(torch.log(torch.expm1(alpha)))

    @property
    def alpha(self) -> Tensor:
        """Return the positive residual scale."""
        return F.softplus(self.residual_alpha)

    def forward(self, features: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        if features.ndim != 3 or gamma.ndim != 2 or beta.shape != gamma.shape:
            raise ValueError("Residual FiLM expects [N,K,D] features and [N,D] parameters")
        if features.shape[0] != gamma.shape[0] or features.shape[-1] != gamma.shape[-1]:
            raise ValueError("Residual FiLM feature and parameter dimensions differ")
        return features + self.alpha * (
            gamma.unsqueeze(1) * features + beta.unsqueeze(1)
        )


class FilmFusionObsEncoder(nn.Module):
    """Frozen CLIP/Sparsh features with Piper FiLM and vision-led fusion."""

    def __init__(
        self,
        clip_encoder: nn.Module,
        sparsh_encoder: nn.Module,
        alpha_init: float = 0.01,
        hidden_dim: int = 256,
        token_dim: int = 256,
        num_heads: int = 4,
        vision_key: str = "wrist",
        tactile_key: str = "right_tactile_camera_taxim_pair",
        state_key: str = "state",
    ) -> None:
        super().__init__()
        if token_dim != 256:
            raise ValueError("film_fusion requires token_dim=256")
        self.clip_encoder = clip_encoder
        self.sparsh_encoder = sparsh_encoder
        self.clip_encoder.requires_grad_(False)
        self.sparsh_encoder.requires_grad_(False)
        self.clip_encoder.eval()
        self.sparsh_encoder.eval()
        self.vision_key = vision_key
        self.tactile_key = tactile_key
        self.state_key = state_key

        self.hypernet = LightweightHyperNet(hidden_dim=hidden_dim)
        self.film = LearnableResidualFiLM(alpha_init=alpha_init)
        self.vision_projection = nn.Linear(VISION_DIM, token_dim)
        self.tactile_projection = nn.Linear(TACTILE_DIM, token_dim)
        self.cross_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.condition_projection = nn.Linear(2 * token_dim, 512)

    def train(self, mode: bool = True) -> "FilmFusionObsEncoder":
        """Train fusion layers while keeping both representation models frozen."""
        super().train(mode)
        self.clip_encoder.eval()
        self.sparsh_encoder.eval()
        return self

    def output_shape(self) -> tuple[int]:
        """Return the per-observation Diffusion Policy feature shape."""
        return (OUTPUT_DIM,)

    def fusion_features(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        """Expose intermediate tensors for validation and ablations."""
        required = {self.vision_key, self.tactile_key, self.state_key}
        missing = sorted(required - observations.keys())
        if missing:
            raise KeyError(f"missing film_fusion observations: {missing}")
        state = observations[self.state_key].float()
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError(f"state must be [N,{STATE_DIM}], got {state.shape}")

        vision = self.clip_encoder(observations[self.vision_key])
        tactile = self.sparsh_encoder(observations[self.tactile_key])
        if vision.shape != (state.shape[0], VISION_DIM):
            raise ValueError(f"CLIP features must be [N,{VISION_DIM}], got {vision.shape}")
        if tactile.shape != (state.shape[0], TACTILE_DIM):
            raise ValueError(f"Sparsh features must be [N,{TACTILE_DIM}], got {tactile.shape}")

        vision = F.layer_norm(vision.float(), (VISION_DIM,)).unsqueeze(1)
        tactile = F.layer_norm(tactile.float(), (TACTILE_DIM,)).unsqueeze(1)
        parameters = self.hypernet(
            torch.cat((vision.squeeze(1), tactile.squeeze(1), state), dim=-1)
        )
        adapted_vision = self.film(
            vision,
            parameters.gamma_vision,
            parameters.beta_vision,
        )
        adapted_tactile = self.film(
            tactile,
            parameters.gamma_tactile,
            parameters.beta_tactile,
        )

        vision_tokens = self.vision_projection(adapted_vision)
        tactile_tokens = self.tactile_projection(adapted_tactile)
        key_value = torch.cat((vision_tokens, tactile_tokens), dim=1)
        attended, _ = self.cross_attention(
            vision_tokens,
            key_value,
            key_value,
            need_weights=False,
        )
        fused_vision = (vision_tokens + attended).flatten(1)
        direct_tactile = tactile_tokens.flatten(1)
        fusion = self.condition_projection(
            torch.cat((fused_vision, direct_tactile), dim=-1)
        )
        condition = torch.cat((fusion, state), dim=-1)
        return {
            "hypernet_output": torch.cat(
                (
                    parameters.gamma_vision,
                    parameters.beta_vision,
                    parameters.gamma_tactile,
                    parameters.beta_tactile,
                ),
                dim=-1,
            ),
            "vision_tokens": vision_tokens,
            "tactile_tokens": tactile_tokens,
            "attention_output": attended,
            "fusion_output": fusion,
            "condition": condition,
        }

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        return self.fusion_features(observations)["condition"]
