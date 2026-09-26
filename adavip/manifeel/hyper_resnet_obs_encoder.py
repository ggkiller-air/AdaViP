"""Trainable Power Plug ResNet encoder with HyperNet modulation."""

from __future__ import annotations

import math
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


FEATURE_DIM = 512
STATE_DIM = 7
CONDITION_DIM = 2 * FEATURE_DIM + STATE_DIM


class HyperResNetObsEncoder(MultiImageObsEncoder):
    """Modulate trainable ResNet18 streams and optionally their fusion."""

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_rgb_model: bool = False,
        imagenet_norm: bool = False,
        use_fusion: bool = False,
        use_fusion_hypernet: bool = False,
        hypernet_hidden_dim: int = 256,
        alpha_init: float = 0.01,
        alpha_max: float = 0.1,
        num_heads: int = 4,
        vision_key: str = "wrist",
        tactile_key: str = "right_tactile_camera_taxim",
        state_key: str = "state",
    ) -> None:
        super().__init__(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            resize_shape=resize_shape,
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            imagenet_norm=imagenet_norm,
        )
        if share_rgb_model:
            raise ValueError("Power Plug HyperResNet requires independent RGB models")
        if set(self.rgb_keys) != {vision_key, tactile_key} or self.low_dim_keys != [state_key]:
            raise ValueError("Power Plug HyperResNet requires wrist, right TacRGB, and state")
        if tuple(self.key_shape_map[state_key]) != (STATE_DIM,):
            raise ValueError(f"{state_key} must have shape [{STATE_DIM}]")
        if hypernet_hidden_dim <= 0 or not 0 < alpha_init < alpha_max:
            raise ValueError("HyperNet width must be positive and 0 < alpha_init < alpha_max")
        if use_fusion_hypernet and not use_fusion:
            raise ValueError("Fusion HyperNet requires use_fusion=True")

        self.vision_key = vision_key
        self.tactile_key = tactile_key
        self.state_key = state_key
        self.use_fusion = use_fusion
        self.use_fusion_hypernet = use_fusion_hypernet
        self.hypernet = nn.Sequential(
            nn.Linear(CONDITION_DIM, hypernet_hidden_dim),
            nn.SiLU(),
            nn.Linear(hypernet_hidden_dim, 4 * FEATURE_DIM),
        )
        nn.init.zeros_(self.hypernet[-1].weight)
        nn.init.zeros_(self.hypernet[-1].bias)
        self.alpha_max = alpha_max
        alpha_fraction = alpha_init / alpha_max
        self.residual_alpha = nn.Parameter(
            torch.tensor(math.log(alpha_fraction / (1 - alpha_fraction)))
        )
        self.cross_attention = nn.MultiheadAttention(
            FEATURE_DIM,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))
        self.fusion_hypernet: nn.Module | None = None
        self.fusion_residual_alpha: nn.Parameter | None = None
        if use_fusion_hypernet:
            self.fusion_hypernet = nn.Sequential(
                nn.Linear(CONDITION_DIM, hypernet_hidden_dim),
                nn.SiLU(),
                nn.Linear(hypernet_hidden_dim, 2 * FEATURE_DIM),
            )
            nn.init.zeros_(self.fusion_hypernet[-1].weight)
            nn.init.zeros_(self.fusion_hypernet[-1].bias)
            self.fusion_residual_alpha = nn.Parameter(
                torch.tensor(math.log(alpha_fraction / (1 - alpha_fraction)))
            )

    @property
    def alpha(self) -> Tensor:
        """Return the positive scale of the HyperNet residual."""
        return self.alpha_max * torch.sigmoid(self.residual_alpha)

    @property
    def fusion_alpha(self) -> Tensor:
        """Return the positive scale of the fusion HyperNet residual."""
        if self.fusion_residual_alpha is None:
            raise RuntimeError("Fusion HyperNet is disabled")
        return self.alpha_max * torch.sigmoid(self.fusion_residual_alpha)

    def output_shape(self) -> tuple[int]:
        """Return the baseline-compatible observation feature width."""
        return (CONDITION_DIM,)

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the same observations as the upstream TacRGB baseline."""
        encoded = {}
        for key in self.rgb_keys:
            image = obs_dict[key]
            if tuple(image.shape[1:]) != tuple(self.key_shape_map[key]):
                raise ValueError(f"unexpected {key} image shape: {tuple(image.shape)}")
            feature = self.key_model_map[key](self.key_transform_map[key](image))
            if feature.ndim != 2 or feature.shape[-1] != FEATURE_DIM:
                raise ValueError(f"{key} ResNet feature must be [N,{FEATURE_DIM}]")
            encoded[key] = feature

        state = obs_dict[self.state_key]
        if state.ndim != 2 or state.shape[-1] != STATE_DIM:
            raise ValueError(f"state must be [N,{STATE_DIM}], got {tuple(state.shape)}")
        vision = encoded[self.vision_key]
        tactile = encoded[self.tactile_key]
        context = torch.cat(
            (
                F.layer_norm(vision, (FEATURE_DIM,)),
                F.layer_norm(tactile, (FEATURE_DIM,)),
                state,
            ),
            dim=-1,
        )
        gamma_vision, beta_vision, gamma_tactile, beta_tactile = torch.tanh(
            self.hypernet(context)
        ).chunk(4, dim=-1)
        encoded[self.vision_key] = vision + self.alpha * (
            gamma_vision * vision + beta_vision
        )
        encoded[self.tactile_key] = tactile + self.alpha * (
            gamma_tactile * tactile + beta_tactile
        )

        fused_vision = encoded[self.vision_key]
        if self.use_fusion:
            vision_token = fused_vision.unsqueeze(1)
            tactile_token = encoded[self.tactile_key].unsqueeze(1)
            tokens = torch.cat((vision_token, tactile_token), dim=1)
            attended, _ = self.cross_attention(
                vision_token, tokens, tokens, need_weights=False
            )
            fused_vision = fused_vision + torch.tanh(self.fusion_gate) * attended.squeeze(1)

            if self.use_fusion_hypernet:
                if self.fusion_hypernet is None:
                    raise RuntimeError("Fusion HyperNet was not initialized")
                fusion_context = torch.cat(
                    (
                        F.layer_norm(fused_vision, (FEATURE_DIM,)),
                        F.layer_norm(encoded[self.tactile_key], (FEATURE_DIM,)),
                        state,
                    ),
                    dim=-1,
                )
                gamma_fusion, beta_fusion = torch.tanh(
                    self.fusion_hypernet(fusion_context)
                ).chunk(2, dim=-1)
                fused_vision = fused_vision + self.fusion_alpha * (
                    gamma_fusion * fused_vision + beta_fusion
                )

        encoded[self.vision_key] = fused_vision

        return torch.cat(
            [encoded[key] for key in self.rgb_keys] + [state], dim=-1
        )
