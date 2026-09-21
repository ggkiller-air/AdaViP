"""FM workspace integration hooks for checkpoint retention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dill
import torch
from omegaconf import OmegaConf

from manifeel.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace as _BaseFMWorkspace,
)


class TrainDiffusionUnetImageWorkspace(_BaseFMWorkspace):
    """Use the existing FM workspace with configurable checkpoint pruning."""

    def resume_training(self, cfg: OmegaConf) -> bool:
        """Resume locally, or initialize a new policy from a baseline checkpoint."""
        resumed = super().resume_training(cfg)
        if resumed:
            return True

        checkpoint = OmegaConf.select(cfg, "initialization.checkpoint", default=None)
        if checkpoint:
            state_dict_name = str(
                OmegaConf.select(cfg, "initialization.state_dict", default="ema_model")
            )
            self._initialize_policy(Path(checkpoint), state_dict_name)
        return False

    def _initialize_policy(self, checkpoint: Path, state_dict_name: str) -> None:
        """Load matching baseline policy weights while retaining new adapter weights."""
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {checkpoint}")
        print(f"Initializing from {state_dict_name} in {checkpoint}")
        payload = torch.load(checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
        source = payload["state_dicts"][state_dict_name]
        normalizer_prefix = "normalizer."
        normalizer_state = {
            key[len(normalizer_prefix) :]: value
            for key, value in source.items()
            if key.startswith(normalizer_prefix)
        }
        self.model.normalizer.load_state_dict(normalizer_state, strict=True)
        target = self.model.state_dict()
        transferred = {
            key: value
            for key, value in source.items()
            if key in target and target[key].shape == value.shape
        }
        required_prefixes = (
            "obs_encoder.key_model_map.",
            "model.",
            "normalizer.",
        )
        missing_prefixes = [
            prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in transferred)
        ]
        if missing_prefixes:
            raise RuntimeError(
                "Baseline checkpoint did not provide compatible weights for: "
                + ", ".join(missing_prefixes)
            )
        result = self.model.load_state_dict(transferred, strict=False)
        if self.ema_model is not None:
            self.ema_model.load_state_dict(self.model.state_dict(), strict=True)
        del payload, source, target
        print(
            f"Transferred {len(transferred)} tensors; "
            f"initialized {len(result.missing_keys)} adapter tensors separately."
        )

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> str:
        """Save a checkpoint while honoring ``checkpoint.prune``."""
        kwargs.setdefault(
            "prune",
            bool(OmegaConf.select(self.cfg, "checkpoint.prune", default=True)),
        )
        return super().save_checkpoint(*args, **kwargs)
