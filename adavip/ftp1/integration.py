"""UViP adapter around unmodified native FTP-1 training and cached inference.

Import with the pinned FTP-1 src directory on PYTHONPATH. All original action
queries, attention layouts, state/time conditioning and heads are inherited.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch.ftp1_pytorch import FTP1Pytorch
from openpi.models_pytorch.ftp1_attention_masks import build_prefix_attention_layout, build_tactile_attention_layout

from .modulation import FTP1UtilityModulation, per_sample_flow_loss


class LoRALinear(nn.Module):
    """Frozen original linear plus zero-initialized rank-r update, no dropout."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, value):
        update = torch.nn.functional.linear(
            torch.nn.functional.linear(value.float(), self.lora_a), self.lora_b)
        return self.base(value) + (update * self.scale).to(value.dtype)


def add_action_lora(model: FTP1Pytorch, rank: int = 16, alpha: float = 16) -> list[str]:
    """Restrict adapters to seven projections in each of 18 action layers."""
    target = model.paligemma_with_expert.gemma_expert.model
    names = []
    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for name, module in list(target.named_modules()):
        if name.split(".")[-1] in allowed and isinstance(module, nn.Linear):
            parent_name, leaf = name.rsplit(".", 1)
            setattr(target.get_submodule(parent_name), leaf, LoRALinear(module, rank, alpha))
            names.append(f"paligemma_with_expert.gemma_expert.model.{name}")
    if len(names) != 18 * 7:
        raise RuntimeError(f"Expected exactly 126 action LoRA targets, found {len(names)}")
    return names


class FTP1UViP(FTP1Pytorch):
    """Observation FiLM applied before both native full forward and KV prefill."""

    def attach_uvip(self, rank: int = 16, alpha: float = 16):
        self.requires_grad_(False)
        self.uvip = FTP1UtilityModulation().to(next(self.parameters()).device)
        self.lora_targets = add_action_lora(self, rank, alpha)
        self.uvip_enabled = True
        self.presence = None
        self._prepared = None
        self._raw_features = None
        self.last_utility = None
        self._velocity = None
        self.action_out_proj.register_forward_hook(self._capture_velocity)
        self.set_phase("A")
        return self

    def _capture_velocity(self, module, inputs, output):
        self._velocity = output

    def set_phase(self, phase: str):
        if phase not in {"A", "B", "C"}:
            raise ValueError(phase)
        self.phase = phase
        for name, value in self.named_parameters():
            utility = name.startswith("uvip.utility_predictor.")
            other_uvip = name.startswith("uvip.") and not utility
            action_lora = name.startswith("paligemma_with_expert.gemma_expert.model.") and (
                name.endswith(".lora_a") or name.endswith(".lora_b"))
            value.requires_grad_(utility or (phase != "A" and (other_uvip or action_lora)))
        # No stochastic augmentation/dropout in the frozen native branches.
        self.eval()

    def trainable_audit(self):
        return {name: list(value.shape) for name, value in self.named_parameters() if value.requires_grad}

    def _preprocess_observation(self, observation, *, train=True):
        # Native forward hardcodes train=True; scoring/student use deterministic
        # inputs so each cached utility target describes exactly the same window.
        self._prepared = None
        args, prefix, tactile, state_pack = self.encode_native(observation) if self._raw_features is None else self._raw_features
        images = args[0]
        if len(images) != 2 or tactile[0] is None or self.state_input_mode != "adarms":
            raise ValueError("Final six-task protocol requires two cameras, tactile, state adaRMS")
        n = self._image_tokens_per_image
        pv, pm, pa = prefix
        tv, tm = tactile
        sr, se, sm, sa = state_pack
        batch = len(pv)
        flags = torch.ones(batch, 6, dtype=torch.bool, device=pv.device) if self.presence is None else self.presence
        if flags.shape != (batch, 6) or flags.dtype != torch.bool:
            raise ValueError("Expected boolean [B,6] V1,V2,T1,T2,P,L flags")
        positions = torch.arange(tv.shape[1], device=tv.device)[None]
        # Both physical pads belong to the right hand: FAAS areas 0 and 1.
        tokens = {"V1": pv[:, :n], "V2": pv[:, n:2*n], "T1": tv, "T2": tv,
                  "P": se, "L": pv[:, 2*n:]}
        masks = {"V1": pm[:, :n] & flags[:, 0:1], "V2": pm[:, n:2*n] & flags[:, 1:2],
                 "T1": tm & (positions == 0) & flags[:, 2:3],
                 "T2": tm & (positions == 1) & flags[:, 3:4],
                 "P": sm & flags[:, 4:5], "L": pm[:, 2*n:] & flags[:, 5:6]}
        if (tm & (positions > 1)).any():
            raise ValueError("Unexpected valid tactile areas outside the two released pads")
        if self.uvip_enabled:
            result = self.uvip({k: v.float() for k, v in tokens.items()}, masks)
            mod = result["tokens"]
            self.last_utility = result["utility"]
        else:
            mod = {k: v.masked_fill(~masks[k][..., None], 0) for k, v in tokens.items()}
            self.last_utility = None
        new_prefix = torch.cat([mod["V1"], mod["V2"], mod["L"]], 1).to(pv.dtype)
        new_tactile = (mod["T1"] + mod["T2"]).to(tv.dtype)
        # Clear raw Fourier output as well: native adaRMS otherwise bypasses P mask.
        sr = sr.masked_fill(~flags[:, 4, None, None], 0)
        self._prepared = ((new_prefix, torch.cat([masks["V1"], masks["V2"], masks["L"]], 1), pa),
                          (new_tactile, masks["T1"] | masks["T2"]),
                          (sr, mod["P"].to(se.dtype), masks["P"], sa))
        return args

    @torch.no_grad()
    def encode_native(self, observation):
        args = super()._preprocess_observation(observation, train=False)
        images, imasks, language, lmasks, state, smasks, tactiles, areas, sensors, _, _ = args
        return (args, super().embed_prefix(images, imasks, language, lmasks),
                super().embed_tactile(tactiles, areas, sensors), super()._encode_state_tokens(state, smasks))

    @torch.no_grad()
    def prefill(self, observation):
        """Native inference KV cache; reusable for paired Teacher noise/time draws."""
        self._preprocess_observation(observation, train=False)
        (prefix, pm, pa), (tactile, tm), state = self._prepared
        prefix = prefix.to(self._get_vlm_branch_dtype())
        tactile = tactile.to(self._get_tactile_branch_dtype())
        pl = build_prefix_attention_layout(pm, pa)
        tl = build_tactile_attention_layout(tm, position_offset=pl.position_ids[:, -1:] + 1)
        pg = self.paligemma_with_expert
        for branch in (pg.paligemma.language_model, pg.gemma_expert.model, pg.gemma_tactile_expert.model):
            branch.config._attn_implementation = self._resolve_attn_implementation(for_inference=True)
        _, pc = pg.forward(attention_mask=self._prepare_attention_masks_4d(pl.att_2d_masks, dtype=prefix.dtype),
                           position_ids=pl.position_ids, inputs_embeds=[prefix, None, None],
                           past_key_values=None, use_cache=True)
        _, tc = pg.forward(attention_mask=self._prepare_attention_masks_4d(tl.att_2d_masks, dtype=tactile.dtype),
                           position_ids=tl.position_ids, inputs_embeds=[None, tactile, None],
                           past_key_values=None, use_cache=True)
        cache = DynamicCache()
        for layer in range(len(pc)):
            cache.update(torch.cat([pc[layer][0], tc[layer][0]], 2),
                         torch.cat([pc[layer][1], tc[layer][1]], 2), layer)
        self._prepared = None
        return pm, cache, prefix, (tactile, tm), state

    def cached_velocity(self, cache_pack, noisy_actions, time):
        pm, cache, prefix, tactile, state = cache_pack
        return self.denoise_step(pm, cache, noisy_actions, time, prefix_embs=prefix,
                                tactile_pack=tactile, state_embs_pack=state,
                                attn_implementation=self._resolve_attn_implementation(for_inference=True))

    def embed_prefix(self, *args):
        return self._prepared[0] if self._prepared is not None else super().embed_prefix(*args)

    def embed_tactile(self, *args):
        return self._prepared[1] if self._prepared is not None else super().embed_tactile(*args)

    def _encode_state_tokens(self, *args):
        return self._prepared[2] if self._prepared is not None else super()._encode_state_tokens(*args)

    def forward(self, observation, actions, noise=None, time=None):
        noise = self.sample_noise(actions.shape, actions.device) if noise is None else noise
        time = self.sample_time(len(actions), actions.device) if time is None else time
        result = super().forward(observation, actions, noise=noise, time=time)
        mask = observation.action_masks
        if mask is None:
            mask = torch.ones_like(actions, dtype=torch.bool)
        result["per_sample_loss"] = per_sample_flow_loss(self._velocity, noise - actions, mask)
        result["velocity"] = self._velocity
        result["utility"] = self.last_utility
        self._velocity = None
        self._prepared = None
        return result


def load_released(checkpoint: str | Path, device: str = "cuda", rank: int = 16):
    """Load native weights first; then attach UViP/zero LoRA without lost tensors."""
    from openpi.policies.ftp1_load_utils import load_ftp1_model

    model, config, directory = load_ftp1_model(Path(checkpoint), device=device)
    # Preserve the fully loaded native object and native state_dict key layout.
    model.__class__ = FTP1UViP
    model.attach_uvip(rank=rank, alpha=rank)
    return model, config, directory
