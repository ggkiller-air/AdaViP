"""AdaViP policy wrapper around an unchanged action-policy backbone."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from adavip.model.perception import AdaViPPerception, PerceptionOutput


class AdaViPPolicy(nn.Module):
    """Adapt multimodal perception before forwarding to a shared backbone.

    When ``base_encoders`` is omitted, ``modalities`` must already contain
    feature tensors. When encoders are supplied, the policy freezes them and
    accepts raw modality inputs with the same keys. This keeps the policy
    independent of image/tactile storage and encoder implementation details.
    """

    def __init__(
        self,
        perception: AdaViPPerception,
        backbone: nn.Module,
        base_encoders: Optional[Mapping[str, nn.Module]] = None,
    ):
        super().__init__()
        self.perception = perception
        self.backbone = backbone

        if base_encoders is None:
            self.base_encoders = None
        else:
            missing = [
                name
                for name in perception.modality_names
                if name not in base_encoders
            ]
            if missing:
                raise ValueError(f"Missing base encoders for modalities: {missing}")
            self.base_encoders = nn.ModuleDict(
                {name: base_encoders[name] for name in perception.modality_names}
            )
            self._freeze_base_encoders()

    def _freeze_base_encoders(self) -> None:
        if self.base_encoders is None:
            return
        for encoder in self.base_encoders.values():
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "AdaViPPolicy":
        super().train(mode)
        # Keep frozen encoders in inference mode even when the policy trains.
        self._freeze_base_encoders()
        return self

    def _encode_modalities(self, modalities: Mapping[str, Tensor]):
        if self.base_encoders is None:
            return modalities
        missing = [
            name for name in self.perception.modality_names if name not in modalities
        ]
        if missing:
            raise KeyError(f"Missing modality inputs: {missing}")
        with torch.no_grad():
            return {
                name: self.base_encoders[name](modalities[name])
                for name in self.perception.modality_names
            }

    def forward(
        self,
        modalities: Mapping[str, Tensor],
        task_embedding: Tensor,
        progress: Optional[Tensor] = None,
        return_perception: bool = False,
        **backbone_kwargs,
    ):
        features = self._encode_modalities(modalities)
        perception_output: PerceptionOutput = self.perception(
            features=features,
            task_embedding=task_embedding,
            progress=progress,
        )
        action = self.backbone(perception_output.fused, **backbone_kwargs)
        if return_perception:
            return {"action": action, "perception": perception_output}
        return action
