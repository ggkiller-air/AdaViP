"""Tabero Field+FS S0 Student with the inherited late-language context."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .model import TaberoL1Policy


class Film(nn.Module):
    """Identity-initialized channel modulation for one native sensor token."""

    def __init__(self, dimension):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2 * dimension))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, token, context):
        scale, bias = self.network(context).chunk(2, dim=-1)
        return token * (1 + scale) + bias


class TaberoUViPPolicy(TaberoL1Policy):
    """Six signed utilities, three FiLM heads and the DP S0 attention fusion."""

    def __init__(self, config, utility_rms, encoder=None):
        super().__init__(config, encoder)
        dimension = config['token_dim']
        rms = torch.as_tensor(utility_rms, dtype=torch.float32)
        if rms.ndim != 2 or rms.shape[1] != 6 or not torch.isfinite(rms).all() or (rms <= 0).any():
            raise ValueError('Expected positive per-instruction six-dimensional Utility RMS')
        self.register_buffer('utility_rms', rms.clone())
        self.utility_predictor = nn.Sequential(nn.Linear(dimension, dimension), nn.SiLU(), nn.Linear(dimension, 6))
        self.z_projection = nn.Sequential(nn.Linear(dimension, 64), nn.SiLU())
        self.utility_projection = nn.Sequential(nn.Linear(6, 64), nn.SiLU())
        self.film_hypernets = nn.ModuleList([Film(dimension) for _ in range(3)])
        self.language_token_projection = nn.Sequential(nn.Linear(512, dimension), nn.LayerNorm(dimension), nn.SiLU())
        self.query_projection = nn.Sequential(nn.Linear(6, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        nn.init.zeros_(self.query_projection[-1].weight)
        nn.init.zeros_(self.query_projection[-1].bias)
        self.cross_attention = nn.MultiheadAttention(dimension, 4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(dimension)
        self.fusion_ff = nn.Sequential(nn.Linear(dimension, 2 * dimension), nn.SiLU(), nn.Linear(2 * dimension, dimension))
        self.phase = 'C'

    def initialize_from_teacher(self, teacher):
        """Copy the exact encoder and normalizer while retaining a fresh U-Net."""
        if self.config['token_dim'] != teacher.config['token_dim']:
            raise ValueError('Teacher/Student token dimensions differ')
        self.obs_encoder.load_state_dict(teacher.obs_encoder.state_dict(), strict=True)
        self.normalizer.load_state_dict(teacher.normalizer.state_dict(), strict=True)
        for name, value in teacher.obs_encoder.state_dict().items():
            torch.testing.assert_close(self.obs_encoder.state_dict()[name], value, rtol=0, atol=0)

    def set_phase(self, phase):
        """A learns Utility; B adds the actor; C unfreezes inherited encoders."""
        if phase not in ('A', 'B', 'C'):
            raise ValueError(phase)
        self.phase = phase
        self.requires_grad_(phase != 'A')
        self.obs_encoder.requires_grad_(phase == 'C')
        self.utility_predictor.requires_grad_(True)

    def encode_stage2(self, obs):
        sensors = self.obs_encoder.encode_tokens(obs, self.normalizer)
        presence = self.condition_presence[:1].expand(len(sensors), -1)
        context_state = self.obs_encoder.condition(sensors, obs['language'], presence)
        utility = self.utility_predictor(context_state.detach())
        detached = utility.detach()
        context = torch.cat((self.z_projection(context_state), self.utility_projection(detached)), -1)
        tokens = torch.cat((sensors, self.language_token_projection(obs['language'])[:, None]), 1)
        modulated = torch.stack([head(tokens[:, index], context) for index, head in enumerate(self.film_hypernets)]
                                + [tokens[:, 3], tokens[:, 4]], 1)
        query = context_state + self.query_projection(detached)
        attended, _ = self.cross_attention(query[:, None], modulated, modulated, need_weights=False)
        fused = self.fusion_norm(attended[:, 0])
        condition = self.fusion_norm(fused + self.fusion_ff(fused))
        return dict(z_t=context_state, utility=utility, tokens=tokens, conditioning=condition)

    def training_losses(self, batch, phase, noise=None, timesteps=None):
        encoded = self.encode_stage2(batch['obs'])
        target = batch['utility'].to(encoded['utility']) / self.utility_rms[batch['task_index']]
        utility_loss = nn.functional.smooth_l1_loss(encoded['utility'], target)
        action_loss = utility_loss.detach().new_zeros(())
        position = force = action_loss
        if phase != 'A':
            mask = batch['action_loss_mask']
            clean = self.normalizer.normalize('action', batch['action']).masked_fill(~mask.bool(), 0)
            noise = torch.randn_like(clean) if noise is None else noise
            timesteps = torch.randint(self.config['num_diffusion_steps'], (len(clean),), device=clean.device) if timesteps is None else timesteps
            noisy = self.noise_scheduler.add_noise(clean, noise, timesteps)
            prediction = self.model(noisy, timesteps, global_cond=encoded['conditioning'])
            components = self.loss_components((prediction.float() - noise.float()).square(), mask)
            action_loss, position, force = (components[key].mean() for key in ('loss', 'position', 'force'))
        return dict(action_loss=action_loss, utility_loss=utility_loss, position_loss=position,
                    force_loss=force, **encoded)

    def compute_losses(self, *args, **kwargs):
        raise RuntimeError('Student training must use training_losses with its explicit A/B/C phase')

    def compute_loss(self, batch):
        result = self.training_losses(batch, self.phase)
        return result['utility_loss'] if self.phase == 'A' else result['action_loss'] + self.config['utility_weight'] * result['utility_loss']

    @torch.no_grad()
    def predict_action(self, obs):
        condition = self.encode_stage2(obs)['conditioning']
        trajectory = torch.randn(len(condition), self.horizon, 13, device=condition.device)
        self.noise_scheduler.set_timesteps(self.config['num_inference_steps'])
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.model(trajectory, timestep, global_cond=condition)
            trajectory = self.noise_scheduler.step(prediction, timestep, trajectory).prev_sample
        absolute = self.normalizer.unnormalize('action', trajectory)
        absolute[:, :, :6] += obs['state'][:, -1:, :6]
        start = self.history - 1
        return dict(action=absolute[:, start:start + self.config['n_action_steps']], action_pred=absolute)


def load_student(path, weights='ema', device='cpu'):
    """Restore a portable Stage 2 checkpoint without training data or its Teacher."""
    if weights not in ('ema', 'raw'):
        raise ValueError(weights)
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    state = saved['ema' if weights == 'ema' else 'policy']
    model = TaberoUViPPolicy(saved['config'], state['utility_rms'])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
