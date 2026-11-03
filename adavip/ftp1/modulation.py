"""Observation-only UVIP modulation; native action queries remain with FTP-1."""

from __future__ import annotations

import torch
from torch import Tensor, nn


GROUPS = ("V1", "V2", "T1", "T2", "P", "L")
WIDTHS = (2048, 2048, 512, 512, 1024, 2048)
UTILITY_NAMES = (*GROUPS, "vision", "tactile")


class FTP1UtilityModulation(nn.Module):
    """Predict eight utilities and FiLM four native observation token blocks.

    Inputs are pre-fusion native tokens, excluding action tokens and flow time.
    The caller retains native masks, positional encoding, state/time adaRMS,
    action projections and all expert attention. This module has no action path.
    """

    def __init__(self, context_dim: int = 256) -> None:
        super().__init__()
        self.projections = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(width, context_dim), nn.LayerNorm(context_dim), nn.SiLU(),
        ) for name, width in zip(GROUPS, WIDTHS)})
        self.context = nn.Sequential(nn.Linear(6 * context_dim + 6, context_dim),
                                     nn.SiLU(), nn.Linear(context_dim, context_dim))
        self.utility_predictor = nn.Sequential(nn.Linear(context_dim, context_dim),
                                              nn.SiLU(), nn.Linear(context_dim, 8))
        self.context_condition = nn.Sequential(nn.Linear(context_dim, 64), nn.SiLU())
        self.utility_condition = nn.Sequential(nn.Linear(8, 64), nn.SiLU())
        self.film = nn.ModuleDict({name: nn.Linear(128, 2 * width)
                                  for name, width in zip(GROUPS[:4], WIDTHS[:4])})
        for head in self.film.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, tokens: dict[str, Tensor], masks: dict[str, Tensor]) -> dict:
        """Return modulated tokens, detached-predictor utilities and context.

        Utility regression consumes ``utility`` directly. Action training uses
        only ``tokens``; its condition explicitly stops gradients at utility.
        Missing tokens are zeroed after FiLM so its bias cannot resurrect them.
        """
        if set(tokens) != set(GROUPS) or set(masks) != set(GROUPS):
            raise ValueError(f"Expected exactly {GROUPS}")
        batch = tokens["V1"].shape[0]
        pooled, present, visible = [], [], {}
        for name, width in zip(GROUPS, WIDTHS):
            value, valid = tokens[name], masks[name]
            if value.ndim != 3 or value.shape[0] != batch or value.shape[-1] != width:
                raise ValueError(f"{name} must be [B,N,{width}]")
            if valid.shape != value.shape[:2] or valid.dtype != torch.bool:
                raise ValueError(f"{name} mask must be boolean [B,N]")
            has_value = valid.any(dim=1)
            # masked_fill also excludes NaNs in absent/padded source slots.
            visible[name] = value.masked_fill(~valid.unsqueeze(-1), 0)
            mean = visible[name].sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
            pooled.append(self.projections[name](mean) * has_value[:, None])
            present.append(has_value)
        flags = torch.stack(present, dim=1).to(dtype=pooled[0].dtype)
        context = self.context(torch.cat((*pooled, flags), dim=-1))
        utility = self.utility_predictor(context.detach())
        condition = torch.cat((self.context_condition(context),
                               self.utility_condition(utility.detach())), dim=-1)
        result = dict(visible)
        for name in GROUPS[:4]:
            gamma, beta = self.film[name](condition).chunk(2, dim=-1)
            result[name] = ((1 + gamma[:, None]) * visible[name] + beta[:, None]).masked_fill(
                ~masks[name].unsqueeze(-1), 0)
        return {"tokens": result, "utility": utility, "context": context}


def per_sample_flow_loss(prediction: Tensor, target: Tensor, action_mask: Tensor) -> Tensor:
    """Native masked flow MSE per sample, respecting both time and action slots.

    FTP-1's scalar reduction uses the entire batch mask denominator. Utility
    needs separate denominators; its batch mean equals that native scalar only
    when every sample has the same number of valid elements.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Flow predictions and targets must both be [B,H,D]")
    mask = torch.broadcast_to(action_mask.to(device=prediction.device), prediction.shape)
    if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask.sum(dim=(1, 2)) > 0):
        raise ValueError("Every sample must have a nonempty binary action mask")
    error = (prediction.float() - target.float()).square().masked_fill(mask == 0, 0)
    return error.sum(dim=(1, 2)) / mask.sum(dim=(1, 2))
