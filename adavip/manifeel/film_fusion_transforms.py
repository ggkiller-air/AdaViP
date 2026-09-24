"""Tensor preprocessing shared by the frozen CLIP and Sparsh backbones."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def to_float01(images: Tensor) -> Tensor:
    """Convert uint8 images to float while preserving existing float values."""
    if images.dtype == torch.uint8:
        return images.float().mul_(1.0 / 255.0)
    return images.float()


def preprocess_clip_default(images: Tensor, size: int = 224) -> Tensor:
    """Apply OpenAI CLIP's aspect-preserving resize and center crop."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"CLIP expects [batch, 3, height, width], got {images.shape}")
    images = to_float01(images)
    images = TF.resize(
        images,
        size,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    images = TF.center_crop(images, [size, size])
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def preprocess_sparsh(images: Tensor, size: tuple[int, int] = (320, 240)) -> Tensor:
    """Rotate landscape GelSight pairs and preserve Sparsh's [0, 1] range."""
    if images.ndim != 4 or images.shape[1] != 6:
        raise ValueError(f"Sparsh expects [batch, 6, height, width], got {images.shape}")
    images = to_float01(images)
    if images.shape[-2] < images.shape[-1]:
        images = torch.rot90(images, k=-1, dims=(-2, -1))
    if tuple(images.shape[-2:]) != size:
        images = F.interpolate(
            images,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return images
