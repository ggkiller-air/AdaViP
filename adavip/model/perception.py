"""Task-conditioned adaptive visuotactile perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .adaptation import (
    AdaptiveModalityTransform,
    HyperNetwork,
    split_generated_parameters,
)


@dataclass
class PerceptionOutput:
    """Features passed from AdaViP perception to a policy backbone."""

    fused: Tensor
    modalities: Dict[str, Tensor]
    context: Tensor


class AdaViPPerception(nn.Module):
    """Generate task- and context-conditioned multimodal representations.

    Inputs are already modality-specific features. Raw image, tactile, and
    proprioception encoders remain injectable upstream, which keeps this class
    independent of the data collection format and baseline implementation.
    """

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        task_dim: int,
        latent_dim: int = 128,
        progress_dim: int = 0,
        hypernet_hidden_dim: int = 256,
        adaptive_rank: int = 4,
        fusion_heads: int = 4,
        fusion_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not modality_dims:
            raise ValueError("At least one modality is required")
        if task_dim <= 0 or latent_dim <= 0:
            raise ValueError("task_dim and latent_dim must be positive")
        if progress_dim < 0:
            raise ValueError("progress_dim cannot be negative")
        if fusion_heads <= 0:
            raise ValueError("fusion_heads must be positive")
        if latent_dim % fusion_heads != 0:
            raise ValueError("latent_dim must be divisible by fusion_heads")
        if fusion_layers <= 0:
            raise ValueError("fusion_layers must be positive")
        if adaptive_rank <= 0:
            raise ValueError("adaptive_rank must be positive")

        self.modality_dims = dict(modality_dims)
        self.modality_names: Tuple[str, ...] = tuple(modality_dims.keys())
        self.task_dim = task_dim
        self.progress_dim = progress_dim
        self.latent_dim = latent_dim

        self.modality_transforms = nn.ModuleDict(
            {
                name: AdaptiveModalityTransform(
                    input_dim, latent_dim, rank=adaptive_rank
                )
                for name, input_dim in self.modality_dims.items()
            }
        )

        self.fusion_transform = AdaptiveModalityTransform(
            latent_dim, latent_dim, rank=adaptive_rank
        )

        # Context is derived from the current base representations. Base
        # encoders may expose different widths, so retain their true widths
        # here rather than assuming every modality already has latent_dim.
        context_dim = task_dim + progress_dim + sum(self.modality_dims.values())
        generated_width = sum(
            transform.parameter_dim
            for transform in self.modality_transforms.values()
        ) + self.fusion_transform.parameter_dim
        self.hypernetwork = HyperNetwork(
            input_dim=context_dim,
            output_dim=generated_width,
            hidden_dim=hypernet_hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=fusion_heads,
            dim_feedforward=4 * latent_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_modal_fusion = nn.TransformerEncoder(
            encoder_layer, num_layers=fusion_layers
        )
        self.fusion_norm = nn.LayerNorm(latent_dim)

    def _as_sequence(self, name: str, features: Tensor) -> Tensor:
        expected_dim = self.modality_dims[name]
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError(
                f"Modality {name!r} must have [batch, time, feature] shape, "
                f"got {tuple(features.shape)}"
            )
        if features.shape[-1] != expected_dim:
            raise ValueError(
                f"Modality {name!r} expects width {expected_dim}, "
                f"got {features.shape[-1]}"
            )
        return features

    def _context(
        self,
        features: Mapping[str, Tensor],
        task_embedding: Tensor,
        progress: Optional[Tensor],
    ) -> Tensor:
        if task_embedding.ndim != 2 or task_embedding.shape[-1] != self.task_dim:
            raise ValueError(
                f"task_embedding must be [batch, {self.task_dim}], "
                f"got {tuple(task_embedding.shape)}"
            )

        context_parts = [task_embedding]
        batch_size = task_embedding.shape[0]
        if self.progress_dim:
            if progress is None:
                progress = task_embedding.new_zeros(batch_size, self.progress_dim)
            elif progress.ndim == 3:
                progress = progress.mean(dim=1)
            if progress.ndim != 2 or progress.shape != (batch_size, self.progress_dim):
                raise ValueError(
                    f"progress must be [batch, {self.progress_dim}], "
                    f"got {tuple(progress.shape)}"
                )
            context_parts.append(progress)
        elif progress is not None:
            if progress.ndim not in (2, 3):
                raise ValueError("progress must have batch or sequence dimensions")

        context_parts.extend(
            features[name].mean(dim=1) for name in self.modality_names
        )
        return torch.cat(context_parts, dim=-1)

    def forward(
        self,
        features: Mapping[str, Tensor],
        task_embedding: Tensor,
        progress: Optional[Tensor] = None,
    ) -> PerceptionOutput:
        missing = [name for name in self.modality_names if name not in features]
        if missing:
            raise KeyError(f"Missing modality features: {missing}")

        sequence_features = {
            name: self._as_sequence(name, features[name])
            for name in self.modality_names
        }
        batch_sizes = {value.shape[0] for value in sequence_features.values()}
        time_sizes = {value.shape[1] for value in sequence_features.values()}
        if len(batch_sizes) != 1 or len(time_sizes) != 1:
            raise ValueError("All modality features must share batch and time dimensions")
        feature_batch_size = next(iter(batch_sizes))
        if task_embedding.shape[0] != feature_batch_size:
            raise ValueError("Task and modality feature batch dimensions must match")

        context = self._context(sequence_features, task_embedding, progress)
        generated = self.hypernetwork(context)
        parameter_widths = tuple(
            self.modality_transforms[name].parameter_dim
            for name in self.modality_names
        ) + (self.fusion_transform.parameter_dim,)
        modality_parameters, fusion_parameters = split_generated_parameters(
            generated, parameter_widths
        )

        transformed = {
            name: self.modality_transforms[name](sequence_features[name], parameters)
            for name, parameters in zip(self.modality_names, modality_parameters)
        }

        # Attention runs across modalities independently at each timestep.
        stacked = torch.stack(
            [transformed[name] for name in self.modality_names], dim=2
        )
        batch_size, time_steps, modality_count, _ = stacked.shape
        tokens = stacked.reshape(batch_size * time_steps, modality_count, self.latent_dim)
        attended = self.cross_modal_fusion(tokens)
        fused = attended.mean(dim=1).reshape(batch_size, time_steps, self.latent_dim)
        fused = self.fusion_transform(self.fusion_norm(fused), fusion_parameters)

        return PerceptionOutput(fused=fused, modalities=transformed, context=context)
