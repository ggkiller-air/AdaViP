"""Upstream single-task DP workspace with retained epoch checkpoints."""

from __future__ import annotations

from typing import Any

from manifeel.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


class RetainedDiffusionUnetImageWorkspace(TrainDiffusionUnetImageWorkspace):
    """Keep every scheduled checkpoint and the completed final epoch."""

    def save_checkpoint(self, *args: Any, **kwargs: Any) -> str:
        """Disable periodic checkpoint pruning without changing their format."""
        kwargs["prune"] = False
        return super().save_checkpoint(*args, **kwargs)

    def run(self) -> None:
        """Save the completed target epoch after the upstream training loop."""
        super().run()
        if self.epoch == self.cfg.training.num_epochs:
            self.save_checkpoint(epoch=self.epoch, use_thread=False)
