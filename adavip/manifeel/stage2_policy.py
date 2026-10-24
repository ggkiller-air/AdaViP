"""T=2 utility-conditioned FiLM and cross-attention diffusion policy."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from adavip.manifeel.full_sensor_obs_encoder import FullSensorObsEncoder
from adavip.manifeel.reference_diffusion_policy import DenoisingSample, FullSensorReferencePolicy


class FilmHyperNet(nn.Module):
    """Produce identity-initialized per-channel FiLM parameters for one sensor."""

    def __init__(self, token_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2 * token_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, token: Tensor, context: Tensor) -> Tensor:
        """Apply x'=(1+delta_gamma)*x+beta."""
        delta_gamma, beta = self.network(context).chunk(2, dim=-1)
        return (1 + delta_gamma) * token + beta


class Stage2Policy(FullSensorReferencePolicy):
    """Predict signed utility from detached z_t and actions from detached utility."""

    def __init__(self, obs_encoder: FullSensorObsEncoder, noise_scheduler: object,
                 utility_rms: Tensor | None = None, **kwargs: object) -> None:
        super().__init__(obs_encoder=obs_encoder, noise_scheduler=noise_scheduler, **kwargs)
        if obs_encoder.token_dim != 256:
            raise ValueError("Stage 2 requires seven 256D tokens")
        rms = torch.ones(9) if utility_rms is None else torch.as_tensor(utility_rms, dtype=torch.float32)
        if rms.shape != (9,) or not torch.isfinite(rms).all() or (rms <= 0).any():
            raise ValueError("Utility RMS must be nine finite positive values")
        self.register_buffer("utility_rms", rms.clone())
        self.utility_predictor = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 9),
        )
        self.z_projection = nn.Sequential(nn.Linear(256, 64), nn.SiLU())
        self.utility_projection = nn.Sequential(nn.Linear(9, 64), nn.SiLU())
        self.film_hypernets = nn.ModuleList([FilmHyperNet() for _ in range(5)])
        self.query_projection = nn.Sequential(
            nn.Linear(9, 256), nn.SiLU(), nn.Linear(256, 256),
        )
        nn.init.zeros_(self.query_projection[-1].weight)
        nn.init.zeros_(self.query_projection[-1].bias)
        self.cross_attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(256)
        self.fusion_ff = nn.Sequential(nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 256))

    def initialize_from_teacher(self, teacher: FullSensorReferencePolicy) -> None:
        """Strictly inherit the EMA encoder and normalizer; retain fresh DP weights."""
        if teacher.obs_encoder.n_obs_steps != 2 or teacher.obs_encoder.token_dim != 256:
            raise ValueError("Teacher must use the T=2, 256D encoder")
        self.obs_encoder.load_state_dict(teacher.obs_encoder.state_dict(), strict=True)
        self.set_normalizer(teacher.normalizer)

    def encode_stage2(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return z_t, normalized utility and the direct Transformer conditioning."""
        tokens = self.obs_encoder.encode_tokens(self.normalizer.normalize(observations))
        z_t = self.obs_encoder.fuse_tokens(tokens)
        u_hat = self.utility_predictor(z_t.detach())
        action_utility = u_hat.detach()
        context = torch.cat((self.z_projection(z_t), self.utility_projection(action_utility)), dim=-1)
        modulated = torch.stack(
            [self.film_hypernets[index](tokens[:, index], context) for index in range(5)]
            + [tokens[:, 5], tokens[:, 6]], dim=1,
        )
        query = z_t + self.query_projection(action_utility)
        attended, _ = self.cross_attention(query[:, None], modulated, modulated, need_weights=False)
        fused = self.fusion_norm(attended[:, 0])
        conditioning = self.fusion_norm(fused + self.fusion_ff(fused))
        return {"z_t": z_t, "tokens": tokens, "u_hat": u_hat, "conditioning": conditioning}

    def training_losses(self, batch: dict, stage: str,
                        denoising_sample: DenoisingSample | None = None) -> dict[str, Tensor]:
        """Compute independent Utility and Action losses with the requested stop-gradients."""
        if stage not in ("A", "B", "C"):
            raise ValueError("Unknown training stage")
        encoded = self.encode_stage2(batch["obs"])
        target = batch["utility"].to(encoded["u_hat"].dtype) / self.utility_rms
        utility_per_dimension = nn.functional.smooth_l1_loss(
            encoded["u_hat"], target, reduction="none", beta=1.0,
        )
        utility_loss = utility_per_dimension.mean()
        if stage == "A":
            action_loss = utility_loss.detach().new_zeros(())
        else:
            sample = self.prepare_denoising_sample(batch) if denoising_sample is None else denoising_sample
            action_loss = self.denoising_loss_per_sample(sample, encoded["conditioning"]).mean()
        return {"action_loss": action_loss, "utility_loss": utility_loss,
                "u_hat": encoded["u_hat"], "utility_target": target,
                "z_t": encoded["z_t"], "conditioning": encoded["conditioning"]}

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor], presence: Tensor | None = None,
                       generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Sample the standard DP action chunk using Stage 2 conditioning."""
        if presence is not None and not bool(presence.all()):
            raise ValueError("Stage 2 inference expects all seven sensors")
        conditioning = self.encode_stage2(obs_dict)["conditioning"]
        trajectory = torch.randn((conditioning.shape[0], self.horizon, 7),
                                 device=conditioning.device, dtype=conditioning.dtype, generator=generator)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, global_cond=conditioning)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory,
                                                   generator=generator).prev_sample
        actions = self.normalizer["action"].unnormalize(trajectory)
        start = self.n_obs_steps - 1
        return {"action": actions[:, start:start + self.n_action_steps], "action_pred": actions}
