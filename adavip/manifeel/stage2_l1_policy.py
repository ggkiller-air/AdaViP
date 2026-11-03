"""S0 Student initialized from the six-sensor, late-language L1 Teacher."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from adavip.manifeel.stage2_policy import Stage2Policy


class Stage2L1Policy(Stage2Policy):
    """Preserve the L1 context path and adapt its tokens to the approved S0 head."""

    def __init__(self, obs_encoder, noise_scheduler, encoder_strategy="s0", **kwargs):
        if obs_encoder.variant != "l1" or encoder_strategy != "s0":
            raise ValueError("The L1 Student requires an L1 encoder and S0 strategy")
        super().__init__(obs_encoder, noise_scheduler, encoder_strategy=encoder_strategy, **kwargs)
        language_dim = obs_encoder.shape_meta["obs"]["task_embedding"]["shape"][0]
        # Outside the inherited encoder: strict Teacher loading remains exact.
        self.language_token_projection = nn.Sequential(
            nn.Linear(language_dim, 256), nn.LayerNorm(256), nn.SiLU())

    def configure_encoders(self, stage: str) -> None:
        """Keep S0's A/B freeze and C unfreeze, including the new action token."""
        super().configure_encoders(stage)
        if hasattr(self, "language_token_projection"):
            self.language_token_projection.requires_grad_(stage != "A")

    def encode_stage2(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        """Use the exact late-language Teacher context for utility and S0 fusion."""
        normalized = self.normalizer.normalize(observations)
        sensors = self.obs_encoder.encode_sensor_tokens(normalized)
        state = self.obs_encoder.fuse_state_tokens(sensors)
        language = self.obs_encoder.language_embedding(normalized)
        context_state = self.obs_encoder.condition_state(state, language)
        tokens = torch.cat((sensors, self.language_token_projection(language)[:, None]), dim=1)
        utility = self.utility_predictor(context_state.detach())
        action_utility = utility.detach()
        context = torch.cat((self.z_projection(context_state),
                             self.utility_projection(action_utility)), dim=-1)
        modulated = torch.stack(
            [net(tokens[:, i], context) for i, net in enumerate(self.film_hypernets)]
            + [tokens[:, 5], tokens[:, 6]], dim=1)
        query = context_state + self.query_projection(action_utility)
        attended, _ = self.cross_attention(query[:, None], modulated, modulated, need_weights=False)
        fused = self.fusion_norm(attended[:, 0])
        conditioning = self.fusion_norm(fused + self.fusion_ff(fused))
        return dict(z_t=context_state, pre_language_z=state, tokens=tokens,
                    u_hat=utility, conditioning=conditioning)
