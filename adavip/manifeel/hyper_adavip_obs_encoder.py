"""Feature-map AdaViP encoder initialized around a frozen FM observation path."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Union

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from adavip.model.adaptation import HyperNetwork
from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer


class HyperAdaViPObsEncoder(ModuleAttrMixin):
    """Adapt four frozen ResNet feature maps with per-modality dynamic encoders.

    Each HyperNet sees a shared context made from global-average-pooled features,
    robot state, and task embedding. It generates all weights and biases for a
    three-stage convolutional encoder. Static trainable projections and decoders
    map the residual branch back to the baseline feature-map shape.
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Mapping[str, nn.Module]],
        rgb_keys: Sequence[str] = (
            "front",
            "wrist",
            "left_tactile_camera_taxim",
            "right_tactile_camera_taxim",
        ),
        state_key: str = "state",
        task_key: str = "task_embedding",
        projection_dim: int = 64,
        dynamic_channels: Sequence[int] = (32, 16, 16),
        hypernet_hidden_dim: int = 256,
        residual_alpha_init: float = 1e-2,
        resize_shape: Union[tuple[int, int], Mapping[str, tuple[int, int]], None] = None,
        crop_shape: Union[tuple[int, int], Mapping[str, tuple[int, int]], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        imagenet_norm: bool = False,
        norm_groups: int = 8,
        freeze_rgb_model: bool = True,
    ) -> None:
        super().__init__()
        if len(rgb_keys) != 4:
            raise ValueError("HyperAdaViPObsEncoder requires exactly four RGB keys")
        if len(dynamic_channels) != 3 or any(width <= 0 for width in dynamic_channels):
            raise ValueError("dynamic_channels must contain three positive widths")
        if projection_dim <= 0 or norm_groups <= 0:
            raise ValueError("projection_dim and norm_groups must be positive")

        self.shape_meta = shape_meta
        self.rgb_keys = tuple(rgb_keys)
        self.sorted_rgb_keys = tuple(sorted(rgb_keys))
        self.state_key = state_key
        self.task_key = task_key
        self.projection_dim = projection_dim
        self.dynamic_channels = tuple(dynamic_channels)
        self.norm_groups = norm_groups
        self.freeze_rgb_model = freeze_rgb_model

        obs_meta = shape_meta["obs"]
        self._validate_keys(obs_meta)
        self.key_shape_map = {key: tuple(value["shape"]) for key, value in obs_meta.items()}
        self.low_dim_keys = tuple(
            sorted(key for key, value in obs_meta.items() if value.get("type", "low_dim") == "low_dim")
        )

        self.key_model_map = nn.ModuleDict()
        self.key_transform_map = nn.ModuleDict()
        for key in self.rgb_keys:
            model = rgb_model[key] if isinstance(rgb_model, Mapping) else copy.deepcopy(rgb_model)
            if use_group_norm:
                model = replace_submodules(
                    root_module=model,
                    predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                    func=lambda module: nn.GroupNorm(
                        num_groups=max(module.num_features // 16, 1),
                        num_channels=module.num_features,
                    ),
                )
            if freeze_rgb_model:
                self._freeze_module(model)
            self.key_model_map[key] = model
            self.key_transform_map[key] = self._make_image_transform(
                key, resize_shape, crop_shape, random_crop, imagenet_norm
            )

        with torch.no_grad():
            feature_channels, feature_height, feature_width = self._infer_feature_map_shape()
        if (feature_height, feature_width) != (8, 8):
            raise ValueError(
                "The three-stage adapter expects 8x8 backbone feature maps, got "
                f"{feature_height}x{feature_width}"
            )
        self.feature_channels = feature_channels

        context_dim = feature_channels * len(self.rgb_keys)
        context_dim += self.key_shape_map[state_key][0] + self.key_shape_map[task_key][0]
        parameter_dim = self._dynamic_parameter_dim()
        self.hypernets = nn.ModuleDict(
            {
                key: HyperNetwork(context_dim, parameter_dim, hypernet_hidden_dim)
                for key in self.rgb_keys
            }
        )
        self.channel_reductions = nn.ModuleDict(
            {key: nn.Conv2d(feature_channels, projection_dim, 1) for key in self.rgb_keys}
        )
        bottleneck_dim = self.dynamic_channels[-1]
        self.decoders = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(bottleneck_dim, 32, 3, padding=1),
                    nn.GELU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(32, projection_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(projection_dim, projection_dim, 3, padding=1),
                    nn.GELU(),
                )
                for key in self.rgb_keys
            }
        )
        self.channel_restorations = nn.ModuleDict(
            {key: nn.Conv2d(projection_dim, feature_channels, 1) for key in self.rgb_keys}
        )
        self.residual_alphas = nn.ParameterDict(
            {
                key: nn.Parameter(torch.tensor(float(residual_alpha_init)))
                for key in self.rgb_keys
            }
        )
        self._output_dim = feature_channels * len(self.rgb_keys) + sum(
            self.key_shape_map[key][0] for key in self.low_dim_keys
        )

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        module.eval()
        module.requires_grad_(False)

    def train(self, mode: bool = True) -> "HyperAdaViPObsEncoder":
        super().train(mode)
        if self.freeze_rgb_model:
            for model in self.key_model_map.values():
                self._freeze_module(model)
        return self

    def _validate_keys(self, obs_meta: dict) -> None:
        required = set(self.rgb_keys + (self.state_key, self.task_key))
        missing = sorted(required.difference(obs_meta))
        if missing:
            raise KeyError(f"shape_meta['obs'] is missing required keys: {missing}")
        for key in self.rgb_keys:
            if obs_meta[key].get("type") != "rgb":
                raise ValueError(f"{key!r} must be declared as type: rgb")

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
            transforms.append(torchvision.transforms.Resize((height, width)))
            input_shape = (input_shape[0], height, width)
        if crop_shape is not None:
            height, width = crop_shape[key] if isinstance(crop_shape, Mapping) else crop_shape
            transforms.append(
                CropRandomizer(input_shape, height, width, num_crops=1, pos_enc=False)
                if random_crop
                else torchvision.transforms.CenterCrop((height, width))
            )
        if imagenet_norm:
            transforms.append(
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )
        return nn.Sequential(*transforms) if transforms else nn.Identity()

    @staticmethod
    def _forward_feature_map(model: nn.Module, image: Tensor) -> Tensor:
        if hasattr(model, "forward_feature_map"):
            return model.forward_feature_map(image)
        required = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
        if not all(hasattr(model, name) for name in required):
            raise TypeError("rgb_model must be a ResNet or implement forward_feature_map(image)")
        feature = model.maxpool(model.relu(model.bn1(model.conv1(image))))
        for name in ("layer1", "layer2", "layer3", "layer4"):
            feature = getattr(model, name)(feature)
        return feature

    def _infer_feature_map_shape(self) -> tuple[int, int, int]:
        key = self.rgb_keys[0]
        example = torch.zeros((1,) + self.key_shape_map[key], dtype=self.dtype, device=self.device)
        feature = self._forward_feature_map(
            self.key_model_map[key], self.key_transform_map[key](example)
        )
        if feature.ndim != 4:
            raise ValueError(f"Backbone feature map must be BCHW, got {tuple(feature.shape)}")
        return int(feature.shape[1]), int(feature.shape[2]), int(feature.shape[3])

    def _dynamic_parameter_dim(self) -> int:
        widths = (self.projection_dim,) + self.dynamic_channels
        return sum(out_width * in_width * 9 + out_width for in_width, out_width in zip(widths, widths[1:]))

    @staticmethod
    def _dynamic_conv2d(feature: Tensor, weight: Tensor, bias: Tensor, stride: int) -> Tensor:
        batch_size, in_channels, height, width = feature.shape
        out_channels = weight.shape[1]
        grouped_feature = feature.reshape(1, batch_size * in_channels, height, width)
        grouped_weight = weight.reshape(batch_size * out_channels, in_channels, 3, 3)
        output = F.conv2d(
            grouped_feature,
            grouped_weight,
            bias.reshape(-1),
            stride=stride,
            padding=1,
            groups=batch_size,
        )
        return output.reshape(batch_size, out_channels, output.shape[-2], output.shape[-1])

    def _dynamic_encode(self, feature: Tensor, parameters: Tensor) -> Tensor:
        offset = 0
        widths = (self.projection_dim,) + self.dynamic_channels
        for stage, (in_width, out_width) in enumerate(zip(widths, widths[1:])):
            weight_width = out_width * in_width * 9
            weight = parameters[:, offset : offset + weight_width].reshape(
                -1, out_width, in_width, 3, 3
            )
            offset += weight_width
            bias = parameters[:, offset : offset + out_width]
            offset += out_width
            feature = self._dynamic_conv2d(feature, weight, bias, stride=2 if stage < 2 else 1)
            groups = min(self.norm_groups, out_width)
            while out_width % groups:
                groups -= 1
            feature = F.gelu(F.group_norm(feature, groups))
        return feature

    def _encode_feature_maps(self, obs_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
        features = {}
        for key in self.rgb_keys:
            image = obs_dict[key]
            if image.shape[1:] != self.key_shape_map[key]:
                raise AssertionError(
                    f"{key} expects {self.key_shape_map[key]}, got {tuple(image.shape[1:])}"
                )
            transformed = self.key_transform_map[key](image)
            if self.freeze_rgb_model:
                with torch.no_grad():
                    features[key] = self._forward_feature_map(self.key_model_map[key], transformed)
            else:
                features[key] = self._forward_feature_map(self.key_model_map[key], transformed)
        return features

    def forward(self, obs_dict: Mapping[str, Tensor]) -> Tensor:
        features = self._encode_feature_maps(obs_dict)
        context = torch.cat(
            [F.adaptive_avg_pool2d(features[key], 1).flatten(1) for key in self.rgb_keys]
            + [obs_dict[self.state_key], obs_dict[self.task_key]],
            dim=-1,
        )
        adapted = {}
        for key in self.rgb_keys:
            reduced = self.channel_reductions[key](features[key])
            bottleneck = self._dynamic_encode(reduced, self.hypernets[key](context))
            delta = self.channel_restorations[key](self.decoders[key](bottleneck))
            adapted[key] = features[key] + self.residual_alphas[key] * delta

        output_parts = [F.adaptive_avg_pool2d(adapted[key], 1).flatten(1) for key in self.sorted_rgb_keys]
        output_parts.extend(obs_dict[key] for key in self.low_dim_keys)
        return torch.cat(output_parts, dim=-1)

    @torch.no_grad()
    def output_shape(self) -> tuple[int]:
        return (self._output_dim,)
