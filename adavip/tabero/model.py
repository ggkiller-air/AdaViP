"""Late-language, full-input diffusion Teacher for Tabero Field+FS."""

from __future__ import annotations

import copy
import math

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from torchvision.models import resnet18

from .schema import IMAGE_KEYS
from .unet import ConditionalUnet1D


class Normalizer(nn.Module):
    """Checkpointed limits for 7D state, 13D target and 396D marker channels."""

    def __init__(self) -> None:
        super().__init__()
        for key, width in (("state", 7), ("action", 13), ("marker", 396)):
            for suffix, value in (("scale", torch.ones(width)), ("offset", torch.zeros(width)),
                                  ("min", torch.zeros(width)), ("max", torch.zeros(width))):
                self.register_buffer(f"{key}_{suffix}", value)

    @torch.no_grad()
    def fit(self, statistics: dict) -> None:
        """Normalize constant dimensions to zero, without dividing by tiny ranges."""
        for key, values in statistics.items():
            minimum, maximum = values["min"], values["max"]
            span = maximum - minimum
            constant = span < 1e-4
            scale = torch.where(constant, torch.ones_like(span), 2 / span.clamp_min(1e-4))
            offset = torch.where(constant, -minimum, -1 - scale * minimum)
            for name, value in (("scale", scale), ("offset", offset), ("min", minimum), ("max", maximum)):
                getattr(self, f"{key}_{name}").copy_(value)

    def normalize(self, key: str, value: Tensor) -> Tensor:
        return value.float() * getattr(self, f"{key}_scale") + getattr(self, f"{key}_offset")

    def unnormalize(self, key: str, value: Tensor) -> Tensor:
        return (value - getattr(self, f"{key}_offset")) / getattr(self, f"{key}_scale")


class CausalTCNBlock(nn.Module):
    """Reference-style causal three-tap residual block without BatchNorm/dropout."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(3)])
        self.residual = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, values: Tensor) -> Tensor:
        padded = nn.functional.pad(values, (0, 0, 2, 0))
        outputs = sum(layer(padded[:, 2 - index:2 - index + values.shape[1]])
                      for index, layer in enumerate(self.kernels))
        return nn.functional.silu(outputs + self.residual(values))


def replace_batch_norm(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(math.gcd(32, child.num_features), child.num_features))
        else:
            replace_batch_norm(child)


class FieldEncoder(nn.Module):
    """Two cameras, bilateral markers and pose; language is late.

    Like the pinned native TCN, two causal three-tap blocks and last-frame
    pooling use only the last five marker frames. The nine-frame input keeps
    the native data layout; its reference and first three history frames have
    no path to the output. Changing this requires a new Teacher/label protocol.
    """

    def __init__(self, history: int = 2, token_dim: int = 256, backbone: nn.Module | None = None,
                 image_feature_dim: int = 512) -> None:
        super().__init__()
        self.history, self.token_dim = history, token_dim
        if backbone is None:
            backbone = resnet18(weights=None)
            backbone.fc = nn.Identity()
            replace_batch_norm(backbone)
        self.images = nn.ModuleDict({key: copy.deepcopy(backbone) for key in IMAGE_KEYS})
        self.temporal = nn.ModuleDict({key: nn.Sequential(
            nn.Linear(history * image_feature_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
            for key in IMAGE_KEYS})
        self.tcn = nn.Sequential(CausalTCNBlock(396, token_dim), CausalTCNBlock(token_dim, token_dim))
        self.tactile_projection = nn.Linear(token_dim, token_dim)
        self.state = nn.Sequential(nn.Linear(history * 7, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
        self.identity = nn.Parameter(torch.randn(4, token_dim) * 0.02)
        self.fusion = nn.Sequential(nn.Linear(4 * token_dim + 4, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))
        self.language_head = nn.Sequential(nn.Linear(token_dim + 512, token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim))

    def encode_tokens(self, obs: dict, normalizer: Normalizer) -> Tensor:
        """Marker input already contains reference + eight causal frames at the anchor."""
        batch = len(obs["state"])
        if obs["state"].shape != (batch, self.history, 7) or obs["marker"].shape != (batch, 9, 198, 2):
            raise ValueError("Invalid Field state/history dimensions")
        tokens = []
        for key in IMAGE_KEYS:
            image = obs[key]
            if image.shape[:3] != (batch, self.history, 3) or image.dtype != torch.uint8:
                raise ValueError("Images must be RGB uint8 [B,To,3,H,W]")
            features = self.images[key](image.flatten(0, 1).float().div(255).sub(0.5).mul(2))
            tokens.append(self.temporal[key](features.reshape(batch, -1)))
        markers = normalizer.normalize("marker", obs["marker"].reshape(batch, 9, 396))
        tokens.append(self.tactile_projection(self.tcn(markers)[:, -1]))
        tokens.append(self.state(normalizer.normalize("state", obs["state"]).flatten(1)))
        return torch.stack(tokens, dim=1) + self.identity[None]

    def condition(self, tokens: Tensor, language: Tensor, presence: Tensor) -> Tensor:
        if presence.shape != (len(tokens), 5) or presence.dtype != torch.bool:
            raise ValueError("Expected five binary modality presence flags")
        visible = tokens.masked_fill(~presence[:, :4, None], 0)
        z = self.fusion(torch.cat((visible.flatten(1), presence[:, :4].to(tokens.dtype)), dim=-1))
        language = language.masked_fill(~presence[:, 4:5], 0)
        return self.language_head(torch.cat((z, language), dim=-1))


class TaberoL1Policy(nn.Module):
    """Action-only DDPM: H16 x 13D; execute eight steps from the current anchor."""

    def __init__(self, config: dict, encoder: FieldEncoder | None = None) -> None:
        super().__init__()
        if config["full_input_probability"] != 1 or config["force_loss_weight"] <= 0:
            raise ValueError("Stage1 requires full inputs and a positive force objective")
        self.config = copy.deepcopy(config)
        self.horizon, self.history = config["horizon"], config["n_obs_steps"]
        self.normalizer = Normalizer()
        self.obs_encoder = encoder or FieldEncoder(self.history, config["token_dim"])
        self.model = ConditionalUnet1D(13, config["token_dim"], config["diffusion_step_embed_dim"], tuple(config["down_dims"]))
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config["num_diffusion_steps"],
                                             beta_schedule="squaredcos_cap_v2", variance_type="fixed_small",
                                             clip_sample=True, prediction_type="epsilon")
        presence = torch.ones(7, 5, dtype=torch.bool)
        for index in range(5):
            presence[index + 1, index] = False
        presence[6, :2] = False
        self.register_buffer("condition_presence", presence)
        self.register_buffer("loss_weights", torch.tensor([1.] * 7 + [config["force_loss_weight"]] * 6))

    def loss_components(self, error: Tensor, mask: Tensor) -> dict:
        """Weighted full 13D MSE, plus separate unweighted position/force MSEs."""
        if mask.shape != error.shape or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("Invalid action/time loss mask")
        if not torch.all(mask.sum(dim=(1, 2)) > 0):
            raise ValueError("Every sample needs supervised actions")
        masked = error.masked_fill(~mask.bool(), 0)
        return {"loss": (masked * self.loss_weights).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)),
                "position": masked[:, :, :7].sum(dim=(1, 2)) / mask[:, :, :7].sum(dim=(1, 2)).clamp_min(1),
                "force": masked[:, :, 7:].sum(dim=(1, 2)) / mask[:, :, 7:].sum(dim=(1, 2)).clamp_min(1)}

    def compute_losses(self, batch: dict, condition_ids: Tensor | None = None,
                       noise: Tensor | None = None, timesteps: Tensor | None = None) -> dict:
        action, mask = batch["action"], batch["action_loss_mask"]
        if action.shape[1:] != (self.horizon, 13):
            raise ValueError("Expected Hx13D position/force targets")
        count = len(action)
        if condition_ids is None:
            condition_ids = torch.zeros(count, dtype=torch.long, device=action.device)
        if self.training and torch.any(condition_ids != 0):
            raise ValueError("L1 Stage1 has no modality dropout")
        tokens = self.obs_encoder.encode_tokens(batch["obs"], self.normalizer)
        condition = self.obs_encoder.condition(tokens, batch["obs"]["language"], self.condition_presence[condition_ids])
        clean = self.normalizer.normalize("action", action).masked_fill(~mask.bool(), 0)
        noise = torch.randn_like(clean) if noise is None else noise
        timesteps = torch.randint(self.config["num_diffusion_steps"], (count,), device=action.device) if timesteps is None else timesteps
        noisy = self.noise_scheduler.add_noise(clean, noise, timesteps)
        prediction = self.model(noisy, timesteps, global_cond=condition)
        return self.loss_components((prediction.float() - noise.float()).square(), mask)

    def compute_loss(self, batch: dict) -> Tensor:
        result = self.compute_losses(batch)
        self.last_metrics = {name: value.detach().mean() for name, value in result.items() if name != "loss"}
        return result["loss"].mean()

    @torch.no_grad()
    def predict_action(self, obs: dict) -> dict:
        tokens = self.obs_encoder.encode_tokens(obs, self.normalizer)
        condition = self.obs_encoder.condition(tokens, obs["language"], self.condition_presence[:1].expand(len(tokens), -1))
        trajectory = torch.randn(len(tokens), self.horizon, 13, device=tokens.device)
        self.noise_scheduler.set_timesteps(self.config["num_inference_steps"])
        for time in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, time, global_cond=condition)
            trajectory = self.noise_scheduler.step(prediction, time, trajectory).prev_sample
        absolute = self.normalizer.unnormalize("action", trajectory)
        absolute[:, :, :6] += obs["state"][:, -1:, :6]
        start = self.history - 1
        return {"action": absolute[:, start:start + self.config["n_action_steps"]], "action_pred": absolute}
