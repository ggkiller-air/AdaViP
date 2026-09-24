"""Strict frozen CLIP RN50 and Sparsh DINO ViT-B inference wrappers."""

from __future__ import annotations

import importlib.util
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from .film_fusion_transforms import preprocess_clip_default, preprocess_sparsh


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import source module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenClipEncoder(nn.Module):
    """Restore the full OpenAI CLIP RN50 image tower and keep it frozen."""

    output_dim = 1024

    def __init__(self, checkpoint_path: str | Path, source_root: str | Path) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        model_source = Path(source_root) / "clip" / "model.py"
        for path in (checkpoint_path, model_source):
            if not path.is_file():
                raise FileNotFoundError(f"required CLIP file is unavailable: {path}")

        model_module = _load_source_module("_adavip_openai_clip_model", model_source)
        scripted = torch.jit.load(str(checkpoint_path), map_location="cpu")
        state_dict = dict(scripted.state_dict())
        del scripted
        required = {
            "visual.attnpool.c_proj.weight",
            "token_embedding.weight",
            "positional_embedding",
            "text_projection",
        }
        missing = sorted(required - state_dict.keys())
        if missing:
            raise KeyError(f"CLIP checkpoint is incomplete: missing {missing}")
        model = model_module.build_model(state_dict).float()
        if int(model.visual.output_dim) != self.output_dim:
            raise ValueError(f"expected RN50 image width 1024, got {model.visual.output_dim}")
        self.model = model
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenClipEncoder":
        """Keep the frozen image tower in inference mode."""
        super().train(False)
        self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        images = preprocess_clip_default(images, size=224)
        with torch.no_grad():
            features = self.model.encode_image(images.to(dtype=self.model.dtype))
        if features.ndim != 2 or features.shape[-1] != self.output_dim:
            raise RuntimeError(f"unexpected CLIP image output shape: {tuple(features.shape)}")
        return features.float()


@contextmanager
def _inference_logging_shim() -> Iterator[None]:
    """Avoid importing Sparsh's training-only OpenCV logger."""
    module_name = "tactile_ssl.utils.logging"
    previous = sys.modules.get(module_name)
    shim = ModuleType(module_name)
    shim.get_pylogger = logging.getLogger  # type: ignore[attr-defined]
    sys.modules[module_name] = shim
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


class FrozenSparshEncoder(nn.Module):
    """Restore the six-channel Sparsh teacher ViT and mean-pool patch tokens."""

    output_dim = 768
    checkpoint_prefix = "teacher_encoder.backbone."

    def __init__(
        self,
        checkpoint_path: str | Path,
        source_root: str | Path,
        image_size: tuple[int, int] = (320, 240),
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        source_root = Path(source_root)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Sparsh checkpoint is unavailable: {checkpoint_path}")
        if not (source_root / "tactile_ssl").is_dir():
            raise FileNotFoundError(f"Sparsh source is unavailable: {source_root}")
        if len(image_size) != 2 or min(image_size) <= 0:
            raise ValueError(f"invalid Sparsh image size: {image_size}")

        source_string = str(source_root.resolve())
        if source_string not in sys.path:
            sys.path.insert(0, source_string)
        with _inference_logging_shim():
            from tactile_ssl.model import vit_base

        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        if "model" not in checkpoint:
            raise KeyError("Sparsh checkpoint has no 'model' state dictionary")
        state = {
            key[len(self.checkpoint_prefix) :]: value
            for key, value in checkpoint["model"].items()
            if key.startswith(self.checkpoint_prefix)
        }
        patch_weight = state.get("patch_embed.proj.weight")
        register_tokens = state.get("register_tokens")
        if patch_weight is None or tuple(patch_weight.shape[:2]) != (self.output_dim, 6):
            shape = None if patch_weight is None else tuple(patch_weight.shape)
            raise ValueError(f"expected Sparsh patch weight [768, 6, P, P], got {shape}")
        patch_size = tuple(int(value) for value in patch_weight.shape[-2:])
        if patch_size != (16, 16):
            raise ValueError(f"expected Sparsh patch size 16, got {patch_size}")
        if register_tokens is None or tuple(register_tokens.shape) != (1, 1, self.output_dim):
            shape = None if register_tokens is None else tuple(register_tokens.shape)
            raise ValueError(f"expected one Sparsh register token, got {shape}")
        if image_size[0] % patch_size[0] or image_size[1] % patch_size[1]:
            raise ValueError("Sparsh image dimensions must be divisible by patch size")

        backbone = vit_base(
            img_size=image_size,
            in_chans=6,
            patch_size=patch_size[0],
            num_register_tokens=1,
            pos_embed_fn="sinusoidal",
        )
        incompatible = backbone.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"non-strict Sparsh restore: {incompatible}")
        self.backbone = backbone
        self.image_size = tuple(image_size)
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenSparshEncoder":
        """Keep the frozen tactile backbone in inference mode."""
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        images = preprocess_sparsh(images, size=self.image_size)
        with torch.no_grad():
            tokens = self.backbone(images.to(dtype=next(self.backbone.parameters()).dtype))
        expected_tokens = self.grid_size[0] * self.grid_size[1]
        if tokens.ndim != 3 or tokens.shape[1:] != (expected_tokens, self.output_dim):
            raise RuntimeError(
                f"expected Sparsh patch output [batch, {expected_tokens}, 768], "
                f"got {tokens.shape}"
            )
        return tokens.float().mean(dim=1)
