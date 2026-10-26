"""T=2 utility-conditioned FiLM and independent Teacher MLP fusion policy."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from adavip.manifeel.full_sensor_obs_encoder import FullSensorObsEncoder
from adavip.manifeel.stage2_policy import FilmHyperNet
from adavip.manifeel.reference_diffusion_policy import DenoisingSample, FullSensorReferencePolicy


class Stage2MlpPolicy(FullSensorReferencePolicy):
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
        self.final_fusion = nn.Sequential(
            nn.Linear(7 * 256 + 7, 256), nn.SiLU(), nn.Linear(256, 256),
        )

    def initialize_from_teacher(self, teacher: FullSensorReferencePolicy) -> None:
        """Inherit two independent Teacher fusion copies and retain a fresh DP."""
        if teacher.obs_encoder.n_obs_steps != 2 or teacher.obs_encoder.token_dim != 256:
            raise ValueError("Teacher must use the T=2, 256D encoder")
        self.obs_encoder.load_state_dict(teacher.obs_encoder.state_dict(), strict=True)
        self.final_fusion.load_state_dict(teacher.obs_encoder.fusion.state_dict(), strict=True)
        if self.final_fusion[0].weight.data_ptr() == self.obs_encoder.fusion[0].weight.data_ptr():
            raise RuntimeError("The two Teacher fusion copies unexpectedly share parameters")
        self.set_normalizer(teacher.normalizer)

    def condition_from_tokens(self, tokens: Tensor, presence: Tensor | None = None,
                              action_utility: Tensor | None = None) -> dict[str, Tensor]:
        """Fuse cached tokens under one presence mask and optional utility probe."""
        if presence is None:
            presence = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if presence.dtype != torch.bool or tuple(presence.shape) != tuple(tokens.shape[:2]):
            raise ValueError("Presence must be boolean [B,7] in canonical token order")
        presence = presence.to(device=tokens.device)
        z_t = self.obs_encoder.fuse_tokens(tokens, presence)
        u_hat = self.utility_predictor(z_t.detach())
        action_utility = u_hat.detach() if action_utility is None else action_utility.detach()
        if action_utility.shape != u_hat.shape:
            raise ValueError("Action utility must have shape [B,9]")
        context = torch.cat((self.z_projection(z_t), self.utility_projection(action_utility)), dim=-1)
        modulated = torch.stack(
            [self.film_hypernets[index](tokens[:, index], context) for index in range(5)]
            + [tokens[:, 5], tokens[:, 6]], dim=1,
        )
        missing = (self.obs_encoder.missing_tokens + self.obs_encoder.sensor_identity).unsqueeze(0)
        masked = torch.where(presence.unsqueeze(-1), modulated, missing)
        conditioning = self.final_fusion(torch.cat((masked.flatten(1), presence.to(tokens.dtype)), dim=-1))
        return {"z_t": z_t, "tokens": tokens, "u_hat": u_hat, "conditioning": conditioning}

    def encode_stage2(self, observations: dict[str, Tensor], presence: Tensor | None = None,
                      action_utility: Tensor | None = None) -> dict[str, Tensor]:
        """Encode the T=2 observations and form one direct DP conditioning vector."""
        tokens = self.obs_encoder.encode_tokens(self.normalizer.normalize(observations))
        return self.condition_from_tokens(tokens, presence, action_utility)

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
                "z_t": encoded["z_t"], "tokens": encoded["tokens"],
                "conditioning": encoded["conditioning"]}

    @torch.no_grad()
    def paired_denoising_losses(self, batch: dict, noise: Tensor | None = None,
                                timesteps: Tensor | None = None,
                                generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Compare full and nine removal conditions with identical diffusion noise."""
        if self.training or self.obs_encoder.training or self.model.training:
            raise RuntimeError("Paired mask diagnostics require eval mode")
        sample = self.prepare_denoising_sample(batch, noise, timesteps, generator)
        tokens = self.obs_encoder.encode_tokens(self.normalizer.normalize(batch["obs"]))
        losses = []
        for presence in self.condition_presence:
            expanded = presence.expand(tokens.shape[0], -1)
            conditioning = self.condition_from_tokens(tokens, expanded)["conditioning"]
            losses.append(self.denoising_loss_per_sample(sample, conditioning))
        absolute = torch.stack(losses, dim=1)
        return {"losses": absolute, "utility": absolute[:, 1:] - absolute[:, :1]}

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor], presence: Tensor | None = None,
                       generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Sample the standard DP action chunk using Stage 2 conditioning."""
        conditioning = self.encode_stage2(obs_dict, presence)["conditioning"]
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
