"""AdaViP observation encoder for ManiFeel diffusion-policy experiments."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Union

import torch
from torch import Tensor, nn
import torchvision

from adavip.model import AdaViPPerception
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer


class AdaViPObsEncoder(ModuleAttrMixin):
    """Encode ManiFeel observations, adapt modalities with AdaViP, and fuse them.

    The encoder keeps the Diffusion Policy interface: it receives a flattened
    observation dict with shape [B*T, ...] and returns one feature vector per
    flattened timestep. RGB streams are encoded with ordinary ResNet backbones;
    AdaViP is applied after per-key encoding and before the final DP condition
    vector is assembled.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Mapping[str, nn.Module]],
        visual_keys: Sequence[str] = ("front", "wrist"),
        tactile_keys: Sequence[str] = (
            "left_tactile_camera_taxim",
            "right_tactile_camera_taxim",
        ),
        proprio_key: str = "state",
        task_key: str = "task_embedding",
        state_feature_dim: int = 128,
        latent_dim: int = 512,
        resize_shape: Union[tuple[int, int], Mapping[str, tuple[int, int]], None] = None,
        crop_shape: Union[tuple[int, int], Mapping[str, tuple[int, int]], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        imagenet_norm: bool = False,
        hypernet_hidden_dim: int = 512,
        adaptive_rank: int = 4,
        fusion_heads: int = 8,
        fusion_layers: int = 1,
        dropout: float = 0.0,
        append_task_embedding: bool = True,
        freeze_rgb_model: bool = False,
    ) -> None:
        super().__init__()
        self.shape_meta = shape_meta
        self.visual_keys = tuple(visual_keys)
        self.tactile_keys = tuple(tactile_keys)
        self.proprio_key = proprio_key
        self.task_key = task_key
        self.latent_dim = latent_dim
        self.append_task_embedding = append_task_embedding
        self.freeze_rgb_model = freeze_rgb_model

        obs_meta = shape_meta["obs"]
        self._validate_keys(obs_meta)
        self.key_shape_map = {key: tuple(value["shape"]) for key, value in obs_meta.items()}

        self.rgb_keys = self.visual_keys + self.tactile_keys
        self.key_model_map = nn.ModuleDict()
        self.key_transform_map = nn.ModuleDict()
        for key in self.rgb_keys:
            this_model = rgb_model[key] if isinstance(rgb_model, Mapping) else copy.deepcopy(rgb_model)
            if use_group_norm:
                this_model = replace_submodules(
                    root_module=this_model,
                    predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                    func=lambda module: nn.GroupNorm(
                        num_groups=max(module.num_features // 16, 1),
                        num_channels=module.num_features,
                    ),
                )
            if freeze_rgb_model:
                self._freeze_module(this_model)
            self.key_model_map[key] = this_model
            self.key_transform_map[key] = self._make_image_transform(
                key=key,
                resize_shape=resize_shape,
                crop_shape=crop_shape,
                random_crop=random_crop,
                imagenet_norm=imagenet_norm,
            )

        with torch.no_grad():
            rgb_feature_dim = self._infer_rgb_feature_dim()

        state_dim = self.key_shape_map[proprio_key][0]
        task_dim = self.key_shape_map[task_key][0]
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim),
            nn.GELU(),
            nn.LayerNorm(state_feature_dim),
        )
        self.perception = AdaViPPerception(
            modality_dims={
                "vision": rgb_feature_dim * len(self.visual_keys),
                "tactile": rgb_feature_dim * len(self.tactile_keys),
                "proprio": state_feature_dim,
            },
            task_dim=task_dim,
            latent_dim=latent_dim,
            progress_dim=0,
            hypernet_hidden_dim=hypernet_hidden_dim,
            adaptive_rank=adaptive_rank,
            fusion_heads=fusion_heads,
            fusion_layers=fusion_layers,
            dropout=dropout,
        )
        self._output_dim = latent_dim + (task_dim if append_task_embedding else 0)

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "AdaViPObsEncoder":
        super().train(mode)
        if self.freeze_rgb_model:
            for model in self.key_model_map.values():
                self._freeze_module(model)
        return self

    def _validate_keys(self, obs_meta: dict) -> None:
        required = set(self.visual_keys + self.tactile_keys + (self.proprio_key, self.task_key))
        missing = sorted(key for key in required if key not in obs_meta)
        if missing:
            raise KeyError(f"shape_meta['obs'] is missing required keys: {missing}")
        for key in self.visual_keys + self.tactile_keys:
            if obs_meta[key].get("type") != "rgb":
                raise ValueError(f"{key!r} must be declared as type: rgb")
        for key in (self.proprio_key, self.task_key):
            if obs_meta[key].get("type", "low_dim") != "low_dim":
                raise ValueError(f"{key!r} must be declared as type: low_dim")

    def _make_image_transform(
        self,
        key: str,
        resize_shape: Union[tuple[int, int], Mapping[str, tuple[int, int]], None],
        crop_shape: Union[tuple[int, int], Mapping[str, tuple[int, int]], None],
        random_crop: bool,
        imagenet_norm: bool,
    ) -> nn.Module:
        input_shape = self.key_shape_map[key]
        transforms: list[nn.Module] = []
        if resize_shape is not None:
            height, width = resize_shape[key] if isinstance(resize_shape, Mapping) else resize_shape
            transforms.append(torchvision.transforms.Resize(size=(height, width)))
            input_shape = (input_shape[0], height, width)

        if crop_shape is not None:
            height, width = crop_shape[key] if isinstance(crop_shape, Mapping) else crop_shape
            if random_crop:
                transforms.append(
                    CropRandomizer(
                        input_shape=input_shape,
                        crop_height=height,
                        crop_width=width,
                        num_crops=1,
                        pos_enc=False,
                    )
                )
            else:
                transforms.append(torchvision.transforms.CenterCrop(size=(height, width)))

        if imagenet_norm:
            transforms.append(
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        if not transforms:
            return nn.Identity()
        return nn.Sequential(*transforms)

    def _infer_rgb_feature_dim(self) -> int:
        key = self.rgb_keys[0]
        shape = self.key_shape_map[key]
        example = torch.zeros((1,) + shape, dtype=self.dtype, device=self.device)
        feature = self.key_model_map[key](self.key_transform_map[key](example))
        if feature.ndim != 2:
            feature = feature.flatten(start_dim=1)
        return int(feature.shape[-1])

    def _encode_rgb_group(self, obs_dict: Mapping[str, Tensor], keys: Sequence[str]) -> Tensor:
        features = []
        for key in keys:
            image = obs_dict[key]
            if image.shape[1:] != self.key_shape_map[key]:
                raise AssertionError(
                    f"{key} expects {self.key_shape_map[key]}, got {tuple(image.shape[1:])}"
                )
            transformed = self.key_transform_map[key](image)
            if self.freeze_rgb_model:
                with torch.no_grad():
                    feature = self.key_model_map[key](transformed)
            else:
                feature = self.key_model_map[key](transformed)
            if feature.ndim != 2:
                feature = feature.flatten(start_dim=1)
            features.append(feature)
        return torch.cat(features, dim=-1).unsqueeze(1)

    def forward(self, obs_dict: Mapping[str, Tensor]) -> Tensor:
        vision = self._encode_rgb_group(obs_dict, self.visual_keys)
        tactile = self._encode_rgb_group(obs_dict, self.tactile_keys)
        proprio = self.state_encoder(obs_dict[self.proprio_key]).unsqueeze(1)
        task_embedding = obs_dict[self.task_key]

        perception = self.perception(
            features={
                "vision": vision,
                "tactile": tactile,
                "proprio": proprio,
            },
            task_embedding=task_embedding,
            progress=None,
        )
        fused = perception.fused.squeeze(1)
        if self.append_task_embedding:
            fused = torch.cat([fused, task_embedding], dim=-1)
        return fused

    @torch.no_grad()
    def output_shape(self) -> tuple[int]:
        return (self._output_dim,)
