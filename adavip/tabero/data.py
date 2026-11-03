"""Episode-bounded Field windows and 13D mixed position/force targets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import CACHE_VERSION, IMAGE_KEYS, fingerprint


class TaberoDataset(Dataset):
    """Store absolute actions; derive all six pose deltas at the current anchor."""

    def __init__(self, directory: str, horizon: int = 16, n_obs_steps: int = 2,
                 preload_ram: bool = True) -> None:
        self.directory = Path(directory)
        if not (self.directory / "COMPLETE.json").exists():
            raise ValueError("Field cache must be complete before training")
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        content = {key: value for key, value in self.manifest.items() if key != "cache_id"}
        if self.manifest["version"] != CACHE_VERSION or fingerprint(content) != self.manifest["cache_id"]:
            raise ValueError("Invalid Field cache manifest")
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        if min(horizon, n_obs_steps) < 1 or n_obs_steps > horizon:
            raise ValueError("Invalid temporal window")
        self.episodes = self.manifest["episodes"]
        self.ends = np.array([episode["end"] for episode in self.episodes])
        shapes = {**{key: (len(self), 3, 224, 224) for key in IMAGE_KEYS},
                  "state": (len(self), 7), "action": (len(self), 13), "marker": (len(self), 9, 198, 2),
                  "language": (len(self.manifest["tasks"]), 512)}
        self.arrays = {}
        for key, shape in shapes.items():
            value = np.load(self.directory / f"{key}.npy", mmap_mode=None if preload_ram else "r")
            if value.shape != shape or value.dtype != (np.uint8 if key in IMAGE_KEYS else np.float32):
                raise ValueError(f"Invalid cached {key}")
            if key not in IMAGE_KEYS and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite cached {key}")
            value.flags.writeable = False
            self.arrays[key] = value
        if self.ends[-1] != len(self):
            raise ValueError("Episode boundaries do not cover the cache")

    def __len__(self) -> int:
        return self.manifest["total_steps"]

    def __getitem__(self, index: int) -> dict:
        episode = self.episodes[np.searchsorted(self.ends, index, side="right")]
        offsets = index - self.n_obs_steps + 1
        observation_indices = np.clip(offsets + np.arange(self.n_obs_steps), episode["start"], episode["end"] - 1)
        action_indices = offsets + np.arange(self.horizon)
        valid = (action_indices >= episode["start"]) & (action_indices < episode["end"])
        clipped = np.clip(action_indices, episode["start"], episode["end"] - 1)
        actions = self.arrays["action"][clipped].copy()
        # Match upstream DeltaActions(make_bool_mask(6,-1)): one CURRENT pose
        # for the entire chunk, including rotation-vector subtraction.
        actions[:, :6] -= self.arrays["state"][index, :6]
        observations = {key: torch.from_numpy(self.arrays[key][observation_indices]) for key in IMAGE_KEYS}
        observations["state"] = torch.from_numpy(self.arrays["state"][observation_indices])
        observations["marker"] = torch.from_numpy(self.arrays["marker"][index].copy())
        observations["language"] = torch.from_numpy(self.arrays["language"][episode["task_index"]].copy())
        return {"obs": observations, "action": torch.from_numpy(actions),
                "action_loss_mask": torch.from_numpy(np.repeat(valid[:, None], 13, axis=1)),
                "source_index": index, "task_index": episode["task_index"]}

    def epoch_indices(self, seed: int, epoch: int) -> list[int]:
        """Uniform frame permutation; preserve the released instruction distribution."""
        generator = torch.Generator().manual_seed(seed + epoch)
        return torch.randperm(len(self), generator=generator).tolist()

    def normalization_statistics(self) -> dict:
        """Fit pose deltas from supervised anchor/chunk pairs, excluding padding."""
        stats = {}
        for name in ("state", "marker"):
            values = self.arrays[name].reshape(-1, 7 if name == "state" else 396)
            stats[name] = {"min": torch.from_numpy(values.min(axis=0)),
                           "max": torch.from_numpy(values.max(axis=0))}
        minimum, maximum = np.full(13, np.inf), np.full(13, -np.inf)
        for episode in self.episodes:
            anchors = np.arange(episode["start"], episode["end"])
            for offset in range(1 - self.n_obs_steps, self.horizon - self.n_obs_steps + 1):
                target_indices = anchors + offset
                valid = (target_indices >= episode["start"]) & (target_indices < episode["end"])
                if not valid.any():
                    continue
                actions = self.arrays["action"][target_indices[valid]].copy()
                actions[:, :6] -= self.arrays["state"][anchors[valid], :6]
                minimum = np.minimum(minimum, actions.min(axis=0))
                maximum = np.maximum(maximum, actions.max(axis=0))
        stats["action"] = {"min": torch.tensor(minimum, dtype=torch.float32),
                           "max": torch.tensor(maximum, dtype=torch.float32)}
        return stats
