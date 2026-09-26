"""Frozen CLIP/Sparsh fusion encoder without FiLM or HyperNet modulation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .film_fusion_obs_encoder import (
    OUTPUT_DIM,
    STATE_DIM,
    TACTILE_DIM,
    VISION_DIM,
)


class CSFusionObsEncoder(nn.Module):
    """Fuse frozen CLIP and Sparsh embeddings with trainable attention layers."""

    def __init__(
        self,
        clip_encoder: nn.Module,
        sparsh_encoder: nn.Module,
        token_dim: int = 256,
        num_heads: int = 4,
        vision_key: str = "wrist",
        tactile_key: str = "right_tactile_camera_taxim_pair",
        state_key: str = "state",
    ) -> None:
        super().__init__()
        if token_dim != 256:
            raise ValueError("CS fusion requires token_dim=256")
        self.clip_encoder = clip_encoder
        self.sparsh_encoder = sparsh_encoder
        self.clip_encoder.requires_grad_(False)
        self.sparsh_encoder.requires_grad_(False)
        self.clip_encoder.eval()
        self.sparsh_encoder.eval()
        self.vision_key = vision_key
        self.tactile_key = tactile_key
        self.state_key = state_key

        self.vision_projection = nn.Linear(VISION_DIM, token_dim)
        self.tactile_projection = nn.Linear(TACTILE_DIM, token_dim)
        self.cross_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.condition_projection = nn.Linear(2 * token_dim, 512)

    def train(self, mode: bool = True) -> "CSFusionObsEncoder":
        """Train only projection and fusion layers."""
        super().train(mode)
        self.clip_encoder.eval()
        self.sparsh_encoder.eval()
        return self

    def output_shape(self) -> tuple[int]:
        """Return the per-observation feature shape expected by DP."""
        return (OUTPUT_DIM,)

    def fusion_features(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        """Expose intermediate tensors for diagnostics and ablations."""
        required = {self.vision_key, self.tactile_key, self.state_key}
        missing = sorted(required - observations.keys())
        if missing:
            raise KeyError(f"missing CS fusion observations: {missing}")
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
        vision_tokens = self.vision_projection(vision)
        tactile_tokens = self.tactile_projection(tactile)
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
            "vision_tokens": vision_tokens,
            "tactile_tokens": tactile_tokens,
            "attention_output": attended,
            "fusion_output": fusion,
            "condition": condition,
        }

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        return self.fusion_features(observations)["condition"]
