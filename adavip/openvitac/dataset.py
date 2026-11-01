"""UViP batch API over immutable videos and converted numeric observations."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from adavip.openvitac.schema import (ACTION_CODEC, IMAGE_KEYS, SCHEMA_VERSION, SENSOR_KEYS,
                                     complete_anchors, make_shape_meta, validate_shape_meta)
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def resize_rgb(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Deterministic RGB uint8 letterbox; preserve aspect ratio and tactile orientation."""
    source_h, source_w = image.shape[:2]
    scale = min(height / source_h, width / source_w)
    resized_h, resized_w = max(1, round(source_h * scale)), max(1, round(source_w * scale))
    if (source_h, source_w) != (resized_h, resized_w):
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    top, left = (height - resized_h) // 2, (width - resized_w) // 2
    return cv2.copyMakeBorder(image, top, height - resized_h - top,
                              left, width - resized_w - left, cv2.BORDER_CONSTANT, value=0)


class OpenViTacDataset(Dataset):
    """Return causal, complete action windows, split by complete source episode.

    Images use a bounded per-worker episode LRU. No decoded full-dataset RAM
    cache or train-time features are silently created. Numeric arrays are mmap.
    """

    def __init__(self, root, split="train", shape_meta=None, n_action_steps=8,
                 raw_root=None, cache_episodes=1, verify_files=True):
        self.root = Path(root)
        content = (self.root / "manifest.json").read_text()
        self.manifest = json.loads(content)
        marker = json.loads((self.root / "COMPLETE.json").read_text())
        if marker["status"] != "complete" or hashlib.sha256(content.encode()).hexdigest() != marker["manifest_sha256"]:
            raise ValueError("Incomplete or corrupt converted dataset")
        if self.manifest["schema"] != SCHEMA_VERSION or self.manifest["action_codec"] != ACTION_CODEC:
            raise ValueError("Dataset/action schema mismatch")
        if verify_files:
            for name, record in self.manifest["files"].items():
                path = self.root / name
                if path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("Corrupt numeric/language artifact: " + name)
        if split not in ("train", "validation", "all"):
            raise ValueError("Unknown split")
        self.split = split
        self.raw_root = Path(raw_root or self.manifest["raw_root"])
        self.shape_meta = shape_meta or make_shape_meta()
        validate_shape_meta(self.shape_meta)
        self.horizon = self.manifest["horizon"]
        self.n_obs_steps = self.manifest["n_obs_steps"]
        self.n_action_steps = n_action_steps
        if not 1 <= n_action_steps <= self.horizon - self.n_obs_steps + 1:
            raise ValueError("Invalid executed action chunk length")
        self.cache_episodes = max(0, int(cache_episodes))
        self._videos = OrderedDict()
        archive = np.load(self.root / "language.npz", allow_pickle=False)
        self.embeddings = archive["embeddings"].copy()
        self.language_texts = archive["texts"].tolist()
        if self.embeddings.shape != (len(self.language_texts), self.shape_meta["obs"]["task_embedding"]["shape"][0]):
            raise ValueError("Frozen language archive dimension mismatch")
        self.tasks, records = [], []
        self.task_specs = [SimpleNamespace(task_id=t["task_id"]) for t in self.manifest["tasks"]]
        self.task_lengths = []
        self.train_episodes, self.val_episodes = [], []
        for task_index, task in enumerate(self.manifest["tasks"]):
            directory = self.root / task["directory"]
            episodes = json.loads((directory / "episodes.json").read_text())
            arrays = {key: np.load(directory / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                      for key in ("state", "action", "action_valid", "language_id", "timestamp", "source_time")}
            windows = np.load(directory / "windows.npy", mmap_mode="r", allow_pickle=False)
            train = {e["episode_index"] for e in episodes if e["split"] == "train"}
            validation = {e["episode_index"] for e in episodes if e["split"] == "validation"}
            if train & validation:
                raise ValueError("Episode split leakage")
            self.train_episodes.append(sorted(train))
            self.val_episodes.append(sorted(validation))
            chosen = train if split == "train" else validation if split == "validation" else train | validation
            selected = windows[np.isin(windows[:, 0], list(chosen))]
            records.extend((task_index, *row) for row in selected.tolist())
            self.task_lengths.append(len(selected))
            self.tasks.append({"spec": task, "episodes": episodes, "arrays": arrays})
        self.index = np.asarray(records, dtype=np.int64).reshape(-1, 5)
        if not len(self.index):
            raise ValueError("No complete windows in split")
        self._normalizer = None

    def __len__(self) -> int:
        return len(self.index)

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_videos"] = OrderedDict()
        return state

    def _decode_episode(self, task_index: int, episode_index: int) -> dict:
        cv2.setNumThreads(1)
        identity = (task_index, episode_index)
        if identity in self._videos:
            self._videos.move_to_end(identity)
            return self._videos[identity]
        episode = self.tasks[task_index]["episodes"][episode_index]
        length = episode["stop"] - episode["start"]
        images = {}
        for key in IMAGE_KEYS:
            height, width = self.shape_meta["obs"][key]["shape"][1:]
            frames = np.empty((length, height, width, 3), dtype=np.uint8)
            with av.open(str(self.raw_root / episode["videos"][key])) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                count = 0
                for count, frame in enumerate(container.decode(stream), start=1):
                    if count > length or frame.pts is None or abs(float(frame.pts * frame.time_base) - (count - 1) / 60) > 1e-5:
                        raise ValueError("Decoded video frame/time alignment mismatch")
                    frames[count - 1] = resize_rgb(frame.to_ndarray(format="rgb24"), height, width)
                if count != length:
                    raise ValueError("Decoded frame count differs from Parquet episode")
            frames.flags.writeable = False
            images[key] = frames
        if self.cache_episodes:
            self._videos[identity] = images
            while len(self._videos) > self.cache_episodes:
                self._videos.popitem(last=False)
        return images

    def __getitem__(self, index: int) -> dict:
        task, episode_id, anchor, segment_start, segment_stop = self.index[index].tolist()
        episode = self.tasks[task]["episodes"][episode_id]
        arrays = self.tasks[task]["arrays"]
        obs_indices = np.arange(anchor - self.n_obs_steps + 1, anchor + 1)
        action_indices = np.arange(obs_indices[0], obs_indices[0] + self.horizon)
        if obs_indices[0] < segment_start or action_indices[-1] + 1 >= segment_stop or not arrays["action_valid"][action_indices].all():
            raise ValueError("Window crosses episode/source-time gap or invalid next-state target")
        images = self._decode_episode(task, episode_id)
        local = obs_indices - episode["start"]
        observations = {key: torch.from_numpy(np.moveaxis(images[key][local], -1, 1).astype(np.float32) / 255)
                        for key in IMAGE_KEYS}
        observations["state"] = torch.from_numpy(np.array(arrays["state"][obs_indices], copy=True))
        language_ids = arrays["language_id"][obs_indices]
        if not np.all(language_ids == episode["language_id"]):
            raise ValueError("Language changed inside history")
        observations["task_embedding"] = torch.from_numpy(self.embeddings[language_ids].copy())
        return {"obs": observations, "action": torch.from_numpy(np.array(arrays["action"][action_indices], copy=True)),
                "action_loss_mask": torch.ones(self.horizon, 8),
                "observation_valid_mask": torch.ones(self.n_obs_steps, dtype=torch.bool),
                "action_valid_mask": torch.ones(self.horizon, dtype=torch.bool),
                "task_index": torch.tensor(task), "episode_index": torch.tensor(episode_id),
                "timestep": torch.tensor(anchor - episode["start"]), "source_index": torch.tensor(anchor),
                "window_id": torch.tensor(task * 10000000 + anchor),
                "language_id": torch.tensor(episode["language_id"]),
                "loader_pid": torch.tensor(os.getpid()),
                "loader_affinity": torch.tensor(sorted(os.sched_getaffinity(0)))}

    def get_future(self, index: int, offsets: tuple = (4,), keys: tuple = IMAGE_KEYS) -> dict:
        """Return future RGB targets with validity against both episode and source gaps."""
        if not offsets or any(k <= 0 for k in offsets) or not set(keys) <= set(IMAGE_KEYS):
            raise ValueError("Invalid future target specification")
        task, episode, anchor, start, stop = self.index[index].tolist()
        targets = anchor + np.array(offsets)
        valid = (targets >= start) & (targets < stop)
        images = self._decode_episode(task, episode)
        first = self.tasks[task]["episodes"][episode]["start"]
        result = {}
        for key in keys:
            shape = self.shape_meta["obs"][key]["shape"]
            values = np.zeros((len(offsets), *shape), dtype=np.float32)
            values[valid] = np.moveaxis(images[key][targets[valid] - first], -1, 1).astype(np.float32) / 255
            result[key] = torch.from_numpy(values)
        return {"obs": result, "valid_mask": torch.from_numpy(valid)}

    def make_task_balanced_sampler(self, generator=None) -> WeightedRandomSampler:
        """Give each of the eleven benchmark tasks equal expected sampling mass."""
        counts = np.bincount(self.index[:, 0], minlength=len(self.tasks))
        weights = 1 / counts[self.index[:, 0]].astype(np.float64)
        return WeightedRandomSampler(torch.from_numpy(weights), len(self), replacement=True, generator=generator)

    def get_validation_dataset(self):
        """Reuse the frozen episode split, preprocessing and training statistics."""
        return type(self)(self.root, split="validation", shape_meta=self.shape_meta,
                          n_action_steps=self.n_action_steps, raw_root=self.raw_root,
                          cache_episodes=self.cache_episodes)

    def get_normalizer(self) -> LinearNormalizer:
        """Fit pooled train-only P/action statistics, using only valid derived targets."""
        if self._normalizer is None:
            states, actions = [], []
            for task_index, task in enumerate(self.tasks):
                arrays = task["arrays"]
                for episode in task["episodes"]:
                    if episode["episode_index"] not in self.train_episodes[task_index]:
                        continue
                    sl = slice(episode["start"], episode["stop"])
                    states.append(arrays["state"][sl])
                    actions.append(arrays["action"][sl][arrays["action_valid"][sl]])
            normalizer = LinearNormalizer()
            for key, values in (("state", states), ("action", actions)):
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(np.concatenate(values), mode="limits")
            for key in IMAGE_KEYS:
                normalizer[key] = get_image_range_normalizer()
            normalizer["task_embedding"] = SingleFieldLinearNormalizer.create_identity()
            normalizer.requires_grad_(False)
            self._normalizer = normalizer
        import copy
        return copy.deepcopy(self._normalizer)
