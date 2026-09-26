"""Power Plug TacRGB baseline encoder with attention fusion only."""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


FEATURE_DIM = 512
STATE_DIM = 7
CONDITION_DIM = 2 * FEATURE_DIM + STATE_DIM


class ResNetFusionObsEncoder(MultiImageObsEncoder):
    """Fuse trainable wrist and TacRGB ResNet18 features without modulation."""

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
            raise ValueError("ResNet fusion requires independent RGB models")
        if set(self.rgb_keys) != {vision_key, tactile_key} or self.low_dim_keys != [state_key]:
            raise ValueError("ResNet fusion requires wrist, right TacRGB, and state")
        if tuple(self.key_shape_map[state_key]) != (STATE_DIM,):
            raise ValueError(f"{state_key} must have shape [{STATE_DIM}]")
        self.vision_key = vision_key
        self.tactile_key = tactile_key
        self.state_key = state_key
        self.cross_attention = nn.MultiheadAttention(
            FEATURE_DIM,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.fusion_gate = nn.Parameter(torch.tensor(0.0))

    def output_shape(self) -> tuple[int]:
        """Return the baseline-compatible observation feature width."""
        return (CONDITION_DIM,)

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Apply vision-led fusion while preserving baseline feature ordering."""
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
        vision_token = encoded[self.vision_key].unsqueeze(1)
        tactile_token = encoded[self.tactile_key].unsqueeze(1)
        tokens = torch.cat((vision_token, tactile_token), dim=1)
        attended, _ = self.cross_attention(
            vision_token, tokens, tokens, need_weights=False
        )
        encoded[self.vision_key] = encoded[self.vision_key] + torch.tanh(
            self.fusion_gate
        ) * attended.squeeze(1)
        return torch.cat([encoded[key] for key in self.rgb_keys] + [state], dim=-1)
