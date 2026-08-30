"""FM workspace integration hooks for checkpoint retention."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from manifeel.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace as _BaseFMWorkspace,
)


class TrainDiffusionUnetImageWorkspace(_BaseFMWorkspace):
    """Use the existing FM workspace with configurable checkpoint pruning."""

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> str:
        """Save a checkpoint while honoring ``checkpoint.prune``."""
        kwargs.setdefault(
            "prune",
            bool(OmegaConf.select(self.cfg, "checkpoint.prune", default=True)),
        )
        return super().save_checkpoint(*args, **kwargs)
