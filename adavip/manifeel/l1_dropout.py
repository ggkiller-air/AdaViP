"""L1 input-dropout ablation with unchanged parameters and full-input behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

from adavip.manifeel.full_sensor_schema import SENSOR_KEYS, condition_probabilities
from adavip.manifeel.language_conditioned_obs_encoder import LanguageConditionedObsEncoder
from adavip.manifeel.language_conditioning_workspace import LanguageConditioningWorkspace, source_fingerprint


def fuse_masked_l1_tokens(encoder: LanguageConditionedObsEncoder, tokens: Tensor, presence: Tensor) -> Tensor:
    """Share the exact token intervention between training and checkpoint diagnostics."""
    if tokens.ndim != 3 or tokens.shape[1:] != (6, encoder.token_dim):
        raise ValueError("State tokens must be [B,6,d]")
    if presence.dtype != torch.bool or presence.shape != tokens.shape[:2]:
        raise ValueError("Sensor presence must be boolean [B,6]")
    presence = presence.to(tokens.device)
    masked = torch.where(presence.unsqueeze(-1), tokens, torch.zeros_like(tokens))
    return encoder.fusion(torch.cat((masked.flatten(1), presence.to(tokens.dtype)), dim=-1))


class L1DropoutObsEncoder(LanguageConditionedObsEncoder):
    """Mask projected sensor tokens and late language without adding parameters."""

    def fuse_state_tokens(self, tokens: Tensor, presence: Tensor | None = None) -> Tensor:
        """Zero removed tokens after projection/identity and append six presence flags."""
        if presence is None:
            return super().fuse_state_tokens(tokens)
        return fuse_masked_l1_tokens(self, tokens, presence)

    def forward(self, observations: dict[str, Tensor], presence: Tensor | None = None) -> Tensor:
        """Apply one sensor-removal condition to the entire observation window."""
        if self.variant != "l1" or set(observations) != set(SENSOR_KEYS):
            raise ValueError("L1 dropout requires L1 and all seven observation keys")
        tokens = self.encode_sensor_tokens(observations)
        language = self.language_embedding(observations)
        if presence is None:
            return self.condition_state(self.fuse_state_tokens(tokens), language)
        if presence.dtype != torch.bool or presence.shape != (len(tokens), 7):
            raise ValueError("Input presence must be boolean [B,7]")
        presence = presence.to(tokens.device)
        state = self.fuse_state_tokens(tokens, presence[:, :6])
        language = torch.where(presence[:, 6:7], language, torch.zeros_like(language))
        return self.condition_state(state, language)


class L1DropoutWorkspace(LanguageConditioningWorkspace):
    """Train action-only L1 with an independent, exactly resumable mask RNG."""

    def __init__(self, cfg: object, output_dir: str | None = None) -> None:
        super().__init__(cfg, output_dir)
        self._dropout_generator = torch.Generator().manual_seed(int(cfg.training.seed) + 400000)
        self._dropout_probabilities = torch.as_tensor(
            condition_probabilities(float(cfg.policy.full_input_probability), list(cfg.policy.mask_weights)),
            dtype=torch.float64,
        )

    def _validate_input_protocol(self) -> None:
        if (self.cfg.variant != "l1" or not isinstance(self.model.obs_encoder, L1DropoutObsEncoder)
                or float(self.cfg.policy.full_input_probability) != 0.9
                or list(self.cfg.policy.mask_weights) != [1] * 9):
            raise ValueError("L1 dropout requires 90% full and 10% uniformly across nine removals")

    def _source_fingerprint(self) -> str:
        digest = hashlib.sha256(source_fingerprint().encode())
        digest.update(Path(__file__).read_bytes())
        digest.update((Path(__file__).resolve().parents[2] /
                       "scripts/manifeel/train_l1_dropout.py").read_bytes())
        return digest.hexdigest()

    def _sample_training_conditions(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.multinomial(self._dropout_probabilities, batch_size, replacement=True,
                                 generator=self._dropout_generator).to(device)

    def _capture_rng(self) -> None:
        super()._capture_rng()
        self.rng_state["input_dropout"] = self._dropout_generator.get_state()

    def _restore_rng(self) -> None:
        super()._restore_rng()
        self._dropout_generator.set_state(self.rng_state["input_dropout"])
