"""Native JAX Field+FS Pi0 with observation-only UViP and action LoRA.

The original Pi0 sampling loop calls this class's embed_prefix before filling
its KV cache. No action-query, attention, loss weights or controller changes.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as native_model
from openpi.models.pi0 import Pi0, make_attn_mask


GROUPS = ("V1", "V2", "T", "P", "L")
WIDTHS = (2048, 2048, 2048, 1024, 2048)
UTILITY_NAMES = (*GROUPS, "vision")


class UtilityModulation(nnx.Module):
    """Six signed utilities, three independent identity FiLM heads."""

    def __init__(self, rngs):
        self.projections = nnx.Dict({name: nnx.Linear(width, 256, rngs=rngs)
                                    for name, width in zip(GROUPS, WIDTHS)})
        self.norms = nnx.Dict({name: nnx.LayerNorm(256, rngs=rngs) for name in GROUPS})
        self.context_in = nnx.Linear(1285, 256, rngs=rngs)
        self.context_out = nnx.Linear(256, 256, rngs=rngs)
        self.utility_predictor = nnx.Dict(
            first=nnx.Linear(256, 256, rngs=rngs), last=nnx.Linear(256, 6, rngs=rngs))
        self.context_condition = nnx.Linear(256, 64, rngs=rngs)
        self.utility_condition = nnx.Linear(6, 64, rngs=rngs)
        self.film = nnx.Dict({name: nnx.Linear(128, 4096, kernel_init=nnx.initializers.zeros,
                                            bias_init=nnx.initializers.zeros, rngs=rngs)
                              for name in GROUPS[:3]})

    def __call__(self, tokens, masks):
        pooled, present, visible = [], [], {}
        for name in GROUPS:
            value, valid = tokens[name].astype(jnp.float32), masks[name]
            visible[name] = jnp.where(valid[..., None], value, 0)
            has = jnp.any(valid, axis=1)
            mean = visible[name].sum(1) / jnp.maximum(valid.sum(1, keepdims=True), 1)
            pooled.append(nnx.swish(self.norms[name](self.projections[name](mean))) * has[:, None])
            present.append(has)
        flags = jnp.stack(present, 1).astype(jnp.float32)
        context = self.context_out(nnx.swish(self.context_in(jnp.concatenate([*pooled, flags], -1))))
        utility = self.utility_predictor.last(nnx.swish(
            self.utility_predictor.first(jax.lax.stop_gradient(context))))
        condition = jnp.concatenate([
            nnx.swish(self.context_condition(context)),
            nnx.swish(self.utility_condition(jax.lax.stop_gradient(utility)))], -1)
        modulated = dict(visible)
        for name in GROUPS[:3]:
            gamma, beta = jnp.split(self.film[name](condition), 2, axis=-1)
            modulated[name] = jnp.where(masks[name][..., None],
                                       (1 + gamma[:, None]) * visible[name] + beta[:, None], 0)
        return modulated, utility, context


def trainable_filter(phase="B"):
    """Only existing expert-1 LoRA and new UViP; VLM's existing LoRA frozen."""
    if phase not in {"A", "B", "C"}:
        raise ValueError(phase)

    def selected(path, value):
        name = "/".join(map(str, path))
        utility = name.startswith("uvip/utility_predictor/")
        if phase == "A":
            return utility
        uvip = name.startswith("uvip/")
        # Native Gemma names action-expert weights with _1; scanned layer axis
        # is inside these arrays and must not be mistaken for the VLM branch.
        action_lora = name.startswith("PaliGemma/llm/") and "lora" in name and any(
            part.endswith("_1") for part in map(str, path))
        return uvip or action_lora

    return nnx.All(nnx.Param, selected)


class TaberoUViP(Pi0):
    """Native Pi0 subclass; explicit masked loss and inherited cached sampling."""

    def attach_uvip(self, seed=42):
        self.uvip = UtilityModulation(nnx.Rngs(seed))
        self.uvip_enabled = True
        return self

    def encode_observation(self, obs, presence=None):
        batch = obs.state.shape[0]
        flags = jnp.ones((batch, 5), jnp.bool_) if presence is None else presence
        values, masks, extra = {}, {}, []
        for index, (name, image) in enumerate(obs.images.items()):
            value, _ = self.PaliGemma.img(image, train=False)
            value = jax.lax.stop_gradient(value)
            valid = jnp.broadcast_to(obs.image_masks[name][:, None], value.shape[:2])
            if index < 2:
                key = ("V1", "V2")[index]
                values[key], masks[key] = value, valid & flags[:, index:index+1]
            else:
                # Native third camera is an invalid placeholder; preserve its
                # block length/position semantics without including it in c.
                extra.append((value, valid))
        values["L"] = jax.lax.stop_gradient(self.PaliGemma.llm(obs.tokenized_prompt, method="embed"))
        masks["L"] = obs.tokenized_prompt_mask & flags[:, 4:5]
        values["T"] = jax.lax.stop_gradient(self._encode_tactile_prefix(obs.tactile_prefix)[:, None])
        masks["T"] = flags[:, 2:3]
        values["P"] = jax.lax.stop_gradient(self.state_proj(obs.state)[:, None])
        masks["P"] = flags[:, 3:4]
        if self.uvip_enabled:
            mod, utility, _ = self.uvip(values, masks)
        else:
            mod = {name: jnp.where(masks[name][..., None], value, 0) for name, value in values.items()}
            utility = jnp.zeros((batch, 6), jnp.float32)
        dtype = values["V1"].dtype
        blocks = [mod["V1"].astype(dtype), mod["V2"].astype(dtype)]
        valid = [masks["V1"], masks["V2"]]
        for value, mask in extra:
            blocks.append(value)
            valid.append(mask)
        blocks.extend([mod["L"].astype(dtype), mod["T"].astype(dtype)])
        valid.extend([masks["L"], masks["T"]])
        prefix = jnp.concatenate(blocks, 1)
        return (prefix, jnp.concatenate(valid, 1), jnp.zeros(prefix.shape[1], jnp.bool_)), utility

    def embed_prefix(self, obs):
        # Called by the unmodified native sample_actions before KV prefill.
        return self.encode_observation(obs)[0]

    def utility(self, obs):
        return self.encode_observation(obs)[1]

    def prefill(self, observation, presence=None):
        """Teacher prefix cache; same positions/mask as native sample_actions."""
        obs = native_model.preprocess_observation(None, observation, train=False, tactile_type=self.tactile_type)
        (tokens, mask, ar), _ = self.encode_observation(obs, presence)
        _, cache = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar),
                                    positions=jnp.cumsum(mask, 1) - 1)
        return mask, cache

    def cached_velocity(self, observation, noisy_actions, time, cache_pack, presence=None):
        pm, cache = cache_pack
        st, sm, sa, cond = super().embed_suffix(observation, noisy_actions, time)
        if presence is not None:
            keep = presence[:, 3]
            st = st.at[:, 0].set(jnp.where(keep[:, None], st[:, 0], 0))
            sm = sm.at[:, 0].set(sm[:, 0] & keep)
        prefix_mask = jnp.broadcast_to(pm[:, None, :], (len(st), st.shape[1], pm.shape[1]))
        mask = jnp.concatenate([prefix_mask, make_attn_mask(sm, sa)], -1)
        positions = pm.sum(-1)[:, None] + jnp.cumsum(sm, -1) - 1
        (_, suffix), _ = self.PaliGemma.llm([None, st], mask=mask, positions=positions,
                                          kv_cache=cache, adarms_cond=[None, cond])
        return self.action_out_proj(suffix[:, -self.action_horizon:])

    def flow_loss(self, rng, observation, actions, *, presence=None, noise=None, time=None, encoded_prefix=None):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        obs = native_model.preprocess_observation(preprocess_rng, observation,
                                                train=False, tactile_type=self.tactile_type)
        noise = jax.random.normal(noise_rng, actions.shape) if noise is None else noise
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * .999 + .001 if time is None else time
        xt = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        target = noise - actions
        if encoded_prefix is None:
            (pt, pm, pa), utility = self.encode_observation(obs, presence)
        else:
            pt, pm, pa = encoded_prefix
            utility = jnp.zeros((actions.shape[0],6),jnp.float32)
        st, sm, sa, cond = super().embed_suffix(obs, xt, time)
        if presence is not None:
            # Field+FS Pi0 has exactly one state token before H50 action tokens.
            keep = presence[:, 3]
            st = st.at[:, 0].set(jnp.where(keep[:, None], st[:, 0], 0))
            sm = sm.at[:, 0].set(sm[:, 0] & keep)
        mask = jnp.concatenate([pm, sm], 1)
        ar = jnp.concatenate([pa, sa], 0)
        (_, suffix), _ = self.PaliGemma.llm([pt, st], mask=make_attn_mask(mask, ar),
                                          positions=jnp.cumsum(mask, 1) - 1, adarms_cond=[None, cond])
        velocity = self.action_out_proj(suffix[:, -self.action_horizon:])
        err = jnp.square(velocity - target)
        # Exactly the released weighted_full objective, including padding.
        weights = jnp.ones(self.action_dim, err.dtype).at[7:13].set(self.tactile_loss_weight)
        weights = weights.at[13:].set(self.padding_loss_weight)
        loss = jnp.mean(err * weights, -1)
        return {"loss": loss, "per_sample_loss": loss.mean(-1), "utility": utility,
                "velocity": velocity, "position_loss": err[..., :7].mean((1, 2)),
                "force_loss": err[..., 7:13].mean((1, 2)), "padding_loss": err[..., 13:].mean((1, 2))}


def load_released(params_path, seed=42):
    """Native shape-strict load, including both released VLM and action LoRA."""
    from openpi.models import pi0_config
    from openpi.shared.tactile_type import TactileType

    config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
        effective_action_dim=13, tactile_type=TactileType.EXPERT_HIS_C_FUT,
        tactile_dim=6, tactile_dim_in=0, tactile_prefix_dim_in=9*198*2,
        tactile_prefix_history=8, tactile_prefix_encoder_type="tcn",
        tactile_prefix_use_reference_frame=True, tactile_prefix_diff_from_reference=False,
        tactile_streams=("tactile_prefix",), tactile_loss_weight=.1)
    params = native_model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.load(params, remove_extra_params=False)
    model.__class__ = TaberoUViP
    model.attach_uvip(seed)
    # FP32 master adapters prevent sub-bfloat16 learning-rate updates vanishing.
    selected = nnx.state(model, trainable_filter("B"))
    selected = jax.tree.map(lambda value: value.astype(jnp.float32), selected)
    nnx.update(model, selected)
    return model, config
