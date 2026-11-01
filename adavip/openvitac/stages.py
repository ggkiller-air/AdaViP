"""Independent Stage 0.5 labels, Stage 1 heads and Stage 2 sensor modulation."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from adavip.openvitac.schema import CONDITION_NAMES, IMAGE_KEYS, SENSOR_KEYS, TACTILE_KEYS, UTILITY_DIM


def fit_utility_rms(utility, train_rows, floor=1e-6):
    """Pool signed targets over training windows only; never recenter or clip signs."""
    values = np.asarray(utility, dtype=np.float64)[train_rows]
    if values.ndim != 2 or values.shape[1] != UTILITY_DIM or not len(values) or not np.isfinite(values).all() or floor <= 0:
        raise ValueError("Expected finite train-only [N,8] utility")
    return np.maximum(np.sqrt(np.mean(values ** 2, axis=0)), floor).astype(np.float32)


class Stage1Dataset(torch.utils.data.Dataset):
    """Join immutable signed labels by stable window id; future targets stay outside obs."""

    def __init__(self, source, label_path, contract, future_keys=TACTILE_KEYS, future_offset=4):
        self.source, self.future_keys, self.future_offset = source, tuple(future_keys), future_offset
        self.labels = UtilityLabelStore.load(label_path, contract)
        if contract["dataset_manifest"] != source.manifest or future_offset <= 0:
            raise ValueError("Label/source contract mismatch or invalid future offset")
        by_id = {int(v): i for i, v in enumerate(self.labels["window_ids"])}
        if len(by_id) != len(self.labels["window_ids"]):
            raise ValueError("Duplicate label window ids")
        self.rows = []
        for task, episode, anchor, start, stop in source.index:
            window_id = int(task * 10000000 + anchor)
            if window_id not in by_id:
                raise ValueError("Labels are incomplete for requested source split")
            self.rows.append(by_id[window_id])

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        sample = self.source[index]
        row = self.rows[index]
        sample["utility"] = torch.from_numpy(self.labels["utility"][row].astype(np.float32))
        sample["utility_variance"] = torch.from_numpy(self.labels["utility_variance"][row].astype(np.float32))
        if self.future_keys:
            future = self.source.get_future(index, (self.future_offset,), keys=self.future_keys)
            sample["future_obs"] = {key: value[0] for key, value in future["obs"].items()}
            sample["future_valid"] = future["valid_mask"][0]
        return sample

    def make_task_balanced_sampler(self, generator=None):
        return self.source.make_task_balanced_sampler(generator)


class Stage2Dataset(Stage1Dataset):
    """Use the same verified label join without decoding future images."""

    def __init__(self, source, label_path, contract):
        super().__init__(source, label_path, contract, future_keys=())


@torch.no_grad()
def paired_utility_labels(teacher, batch, draws=2, generator=None):
    """Stage 0.5: signed masked-minus-full denoising losses, paired within each draw.

    This bounded callable never enumerates a dataset or launches generation.
    K is explicit; no ManiFeel labels or noise archive can be reused.
    """
    if teacher.training or draws < 2:
        raise ValueError("Use a frozen eval Teacher and at least two draws")
    losses = torch.stack([teacher.paired_denoising_losses(batch, generator=generator)["losses"]
                          for _ in range(draws)], dim=1)
    differences = losses[:, :, 1:] - losses[:, :, :1]
    return {"utility": differences.mean(1), "utility_variance": differences.var(1, unbiased=True) / draws,
            "weighted_losses": losses.mean(1), "draws": draws,
            "condition_names": CONDITION_NAMES, "raw_signed_utility": differences}


def utility_contract(dataset_manifest, teacher_sha256, normalizer_sha256):
    """Bind future archives to six sensors, eight targets, data, codec and checkpoint."""
    value = {"schema": "uvip.openvitac.utility.v1", "sensor_keys": SENSOR_KEYS,
             "condition_names": CONDITION_NAMES, "utility_dim": UTILITY_DIM,
             "teacher_sha256": teacher_sha256, "normalizer_sha256": normalizer_sha256,
             "dataset_manifest": dataset_manifest}
    value = json.loads(json.dumps(value))
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


class UtilityLabelStore:
    """Small, immutable archive with explicit ordering and index/provenance checks."""

    @staticmethod
    def save(path, contract, window_ids, labels):
        utility = labels["utility"].detach().cpu().numpy()
        variance = labels["utility_variance"].detach().cpu().numpy()
        losses = labels["weighted_losses"].detach().cpu().numpy()
        if utility.shape != (len(window_ids), UTILITY_DIM) or variance.shape != utility.shape or losses.shape != (len(window_ids), len(CONDITION_NAMES)):
            raise ValueError("Label shapes/order do not follow OpenViTac")
        if len(np.unique(window_ids)) != len(window_ids):
            raise ValueError("Duplicate window ids")
        if path.exists():
            raise FileExistsError(path)
        from pathlib import Path
        temporary = Path(str(path) + ".partial")
        with temporary.open("wb") as stream:
            np.savez(stream, contract=np.array(json.dumps(contract, sort_keys=True)), window_ids=window_ids,
                     utility=utility, utility_variance=variance, weighted_losses=losses)
        temporary.replace(path)

    @staticmethod
    def load(path, contract):
        with np.load(path, allow_pickle=False) as archive:
            if json.loads(str(archive["contract"])) != contract:
                raise ValueError("Teacher/data/action/mask/normalizer provenance mismatch")
            values = {key: archive[key].copy() for key in ("window_ids", "utility", "utility_variance", "weighted_losses")}
        if values["utility"].shape != (len(values["window_ids"]), UTILITY_DIM):
            raise ValueError("Wrong Utility dimension")
        if not all(np.isfinite(values[key]).all() for key in ("utility", "utility_variance", "weighted_losses")) or (values["utility_variance"] < 0).any():
            raise ValueError("Nonfinite/invalid Utility labels")
        np.testing.assert_allclose(values["utility"], values["weighted_losses"][:, 1:] - values["weighted_losses"][:, :1], atol=1e-6, rtol=1e-5)
        return values


class Stage1Heads(nn.Module):
    """Stage 1: eight Utility outputs and independent future targets for 2 or 4 images."""

    def __init__(self, token_dim=256, future_keys=TACTILE_KEYS, resolution=64):
        super().__init__()
        if not set(future_keys) <= set(IMAGE_KEYS) or resolution < 1:
            raise ValueError("Unknown future sensors")
        self.future_keys, self.resolution = tuple(future_keys), resolution
        self.utility_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(), nn.Linear(token_dim, UTILITY_DIM))
        self.future_heads = nn.ModuleDict({key: nn.Sequential(nn.Linear(token_dim, 128), nn.SiLU(),
                                                             nn.Linear(128, 3 * resolution ** 2), nn.Sigmoid())
                                          for key in future_keys})

    def forward(self, z_t, target_utility, future_obs=None, future_valid=None):
        predicted = self.utility_head(z_t)
        if target_utility.shape != predicted.shape:
            raise ValueError("Expected eight signed utility targets")
        utility_loss = F.smooth_l1_loss(predicted, target_utility)
        losses = []
        if future_obs is not None:
            valid = future_valid.bool()
            for key in self.future_keys:
                prediction = self.future_heads[key](z_t).reshape(-1, 3, self.resolution, self.resolution)
                target = F.interpolate(future_obs[key], size=(self.resolution, self.resolution),
                                       mode="bilinear", align_corners=False)
                losses.append(F.mse_loss(prediction[valid], target[valid]) if valid.any() else prediction.sum() * 0)
        future_loss = torch.stack(losses).mean() if losses else z_t.sum() * 0
        return {"utility_prediction": predicted, "utility_loss": utility_loss, "future_loss": future_loss}


class Stage2Policy(nn.Module):
    """Four image HyperNets, unmodulated P/L and detached signed Utility context.

    The wrapped DP is constructed independently; initialize_from_teacher copies
    encoder and frozen normalizer only, preserving fresh denoiser weights.
    """

    def __init__(self, policy, utility_rms=None):
        super().__init__()
        self.policy = policy
        dim = policy.obs_encoder.token_dim
        rms = torch.ones(UTILITY_DIM) if utility_rms is None else torch.as_tensor(utility_rms, dtype=torch.float32)
        if rms.shape != (UTILITY_DIM,) or not torch.isfinite(rms).all() or (rms <= 0).any():
            raise ValueError("Expected eight finite positive Utility RMS scales")
        self.register_buffer("utility_rms", rms)
        self.utility_predictor = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, UTILITY_DIM))
        self.context_projection = nn.Sequential(nn.Linear(dim + UTILITY_DIM, 128), nn.SiLU())
        self.film_hypernets = nn.ModuleList([nn.Linear(128, 2 * dim) for _ in IMAGE_KEYS])
        for head in self.film_hypernets:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.final_fusion = copy.deepcopy(policy.obs_encoder.fusion)

    def initialize_from_teacher(self, teacher):
        self.policy.obs_encoder.load_state_dict(teacher.obs_encoder.state_dict(), strict=True)
        self.final_fusion.load_state_dict(teacher.obs_encoder.fusion.state_dict(), strict=True)
        self.policy.set_normalizer(teacher.normalizer)

    def encode(self, observations):
        encoder = self.policy.obs_encoder
        tokens = encoder.encode_tokens(self.policy.normalizer.normalize(observations))
        z_t = encoder.fuse_tokens(tokens)
        utility = self.utility_predictor(z_t.detach())
        context = self.context_projection(torch.cat((z_t, utility.detach()), dim=-1))
        images = []
        for i, head in enumerate(self.film_hypernets):
            gamma, beta = head(context).chunk(2, dim=-1)
            images.append((1 + gamma) * tokens[:, i] + beta)
        modulated = torch.stack(images + [tokens[:, 4], tokens[:, 5]], dim=1)
        flags = modulated.new_ones(modulated.shape[:2])
        conditioning = self.final_fusion(torch.cat((modulated.flatten(1), flags), dim=-1))
        return {"conditioning": conditioning, "utility_prediction": utility, "tokens": tokens, "z_t": z_t}

    def training_losses(self, batch, phase="C"):
        if phase not in ("A", "B", "C"):
            raise ValueError("Unknown Stage 2 phase")
        encoded = self.encode(batch["obs"])
        utility_loss = F.smooth_l1_loss(encoded["utility_prediction"], batch["utility"] / self.utility_rms)
        action_loss = (self.policy.denoising_loss_per_sample(self.policy.prepare_denoising_sample(batch), encoded["conditioning"]).mean()
                       if phase != "A" else utility_loss.detach().new_zeros(()))
        return {**encoded, "utility_loss": utility_loss, "action_loss": action_loss, "loss": utility_loss + action_loss}

    def configure_phase(self, phase):
        """A: Utility only; B: frozen encoder; C: train image encoder, fusion and DP."""
        if phase not in ("A", "B", "C"):
            raise ValueError("Unknown phase")
        self.policy.requires_grad_(phase != "A")
        self.policy.obs_encoder.requires_grad_(phase == "C")
        self.final_fusion.requires_grad_(phase != "A")
        self.film_hypernets.requires_grad_(phase != "A")
        self.context_projection.requires_grad_(phase != "A")
        self.policy.normalizer.requires_grad_(False)
        self.policy._dummy_variable.requires_grad_(False)

    @torch.no_grad()
    def predict_action(self, observations, generator=None):
        return self.policy.sample_from_condition(self.encode(observations)["conditioning"], generator)
