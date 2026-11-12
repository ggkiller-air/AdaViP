"""S0 Student initialized from the six-sensor, late-language L1 Teacher."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from adavip.manifeel.stage2_policy import Stage2Policy


class Stage2L1Policy(Stage2Policy):
    """Preserve the L1 context path and adapt its tokens to the approved S0 head."""

    def __init__(self, obs_encoder, noise_scheduler, encoder_strategy="s0", ablation="Full", **kwargs):
        if obs_encoder.variant != "l1" or encoder_strategy != "s0":
            raise ValueError("The L1 Student requires an L1 encoder and S0 strategy")
        super().__init__(obs_encoder, noise_scheduler, encoder_strategy=encoder_strategy, **kwargs)
        language_dim = obs_encoder.shape_meta["obs"]["task_embedding"]["shape"][0]
        # Outside the inherited encoder: strict Teacher loading remains exact.
        self.language_token_projection = nn.Sequential(
            nn.Linear(language_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.ablation = "Full"
        self.apply_ablation(ablation)

    def apply_ablation(self, ablation: str) -> None:
        """Prune a canonical Full initialization without consuming random numbers."""
        if ablation not in ("Full", "B", "C", "D", "E", "F"):
            raise ValueError(f"Unknown main ablation: {ablation}")
        if self.ablation != "Full":
            raise ValueError("Ablations must be derived from the canonical Full initialization")
        self.ablation = ablation
        if ablation in ("B", "E"):
            self.utility_predictor = None
        if ablation in ("B", "D", "E"):
            self.query_projection = None
        if ablation in ("C", "E"):
            self.film_hypernets = None
            self.z_projection = None
        if ablation in ("B", "C", "E"):
            self.utility_projection = None
        if ablation == "B":
            # Full uses [z64, u64]. Keep exactly its z columns and bias;
            # remove utility columns instead of retaining permanently dead weights.
            for film in self.film_hypernets:
                linear = film.network[0]
                linear.weight = nn.Parameter(linear.weight[:, :64].detach().clone())
                linear.in_features = 64

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
        utility = (self.utility_predictor(context_state.detach()) if self.utility_predictor is not None
                   else context_state.new_empty((len(context_state), 0)))
        action_utility = utility.detach()
        if self.film_hypernets is None:
            modulated = tokens  # Strict identity, including after optimizer updates.
        else:
            context = self.z_projection(context_state)
            if self.utility_projection is not None:
                context = torch.cat((context, self.utility_projection(action_utility)), dim=-1)
            modulated = torch.stack(
                [net(tokens[:, i], context) for i, net in enumerate(self.film_hypernets)]
                + [tokens[:, 5], tokens[:, 6]], dim=1)
        query = context_state
        if self.query_projection is not None:
            query = query + self.query_projection(action_utility)
        attended, _ = self.cross_attention(query[:, None], modulated, modulated, need_weights=False)
        fused = self.fusion_norm(attended[:, 0])
        conditioning = self.fusion_norm(fused + self.fusion_ff(fused))
        return dict(z_t=context_state, pre_language_z=state, tokens=tokens,
                    u_hat=utility, conditioning=conditioning, modulated=modulated, query=query)

    def training_losses(self, batch, stage, denoising_sample=None):
        """B/E have no Utility computation or supervision, including no label dependency."""
        if self.utility_predictor is not None:
            return super().training_losses(batch, stage, denoising_sample)
        if stage not in ("B", "C"):
            raise ValueError("B/E skip Utility-only Stage A")
        encoded = self.encode_stage2(batch["obs"])
        sample = self.prepare_denoising_sample(batch) if denoising_sample is None else denoising_sample
        per_sample = self.denoising_loss_per_sample(sample, encoded["conditioning"])
        return dict(action_loss=per_sample.mean(), action_loss_per_sample=per_sample,
                    utility_loss=per_sample.new_zeros(()),
                    utility_loss_per_sample=torch.zeros_like(per_sample),
                    u_hat=encoded["u_hat"], utility_target=encoded["u_hat"],
                    z_t=encoded["z_t"], conditioning=encoded["conditioning"])
