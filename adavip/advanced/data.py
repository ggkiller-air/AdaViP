"""Native dataset contracts; advanced targets never reuse DP next-joint labels."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from .common import atomic_json, fingerprint, file_hash


TASK_PROMPTS = {
    "insert_hole": "insert the stick to the hole.",
    "insert_tube": "insert the tube to the fixed slot.",
    "lift_can": "grasp the can and lifts it vertically without slippage.",
    "lift_bottle": "grasp the bottle and lift it vertically, keeping its final base within 5 cm of the wall.",
    "pull_out_key": "pull out the key.",
    "put_bottle_in_shelf": "grasp the bottle, then position it into the shelf cavity.",
}


def build_ftp_cache(task, directory):
    """Match the pinned parser's actual BGR224 output, current-joint timeline.

    The pinned parser converts JPEG to RGB and reverses it again at each call
    site. This intentionally follows executable code, rather than its RGB
    docstring. Raw HDF5 is decoded directly at224, never resized from DP256.
    """
    directory = Path(directory)
    if (directory / "COMPLETE.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    source = Path("/data/wangzihao/datasets/univtac/isaac45") / task / "hdf5"
    paths = sorted(source.glob("*.hdf5"), key=lambda p: int(p.stem))
    episodes, total = [], 0
    for path in paths:
        with h5py.File(path, "r") as store:
            count = len(store["embodiment/joint"]) - 1
        episodes.append({"path": str(path), "start": total, "end": total + count,
                         "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
        total += count
    if len(episodes) != 50:
        raise ValueError(f"Expected 50 released task episodes, found {len(episodes)}")
    arrays = {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype=np.uint8,
                                             shape=(total, 224, 224, 3))
              for name in ("head", "wrist", "tactile_left", "tactile_right")}
    joint = np.lib.format.open_memmap(directory / "joint.npy", mode="w+", dtype=np.float32, shape=(total, 8))
    for episode in episodes:
        start, end = episode["start"], episode["end"]
        with h5py.File(episode["path"], "r") as store:
            joint[start:end] = store["embodiment/joint"][:-1, :8]
            keys = {"head": "observation/head/rgb", "wrist": "observation/wrist/rgb"}
            for side in ("left", "right"):
                keys[f"tactile_{side}"] = next(k for k in (
                    f"tactile/{side}_gsmini/rgb_marker", f"tactile/{side}_tactile/rgb_marker") if k in store)
            for name, key in keys.items():
                frames = store[key][:end-start]
                for offset, encoded in enumerate(frames):
                    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Corrupt {episode['path']} {key} {offset}")
                    arrays[name][start+offset] = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
        print(f"FTP cache {task}: {end}/{total}", flush=True)
    for value in [*arrays.values(), joint]:
        value.flush()
    manifest = {"version": "ftp1-native-parser-bgr224-current-mix-v1", "task": task,
                "episodes": episodes, "total_steps": total, "prompt": TASK_PROMPTS[task],
                "color": "native executable parser BGR", "action_slots": [9,10,11,12,13,14,15,44],
                "timeline": "joint[anchor+h]-joint[anchor] for arm; absolute gripper; no next-step shift; repeat final target"}
    manifest["cache_id"] = fingerprint(manifest)
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(directory / "COMPLETE.json", {"cache_id": manifest["cache_id"]})


class FTPData:
    """Frozen release normalizers, exact slots9:16/44, no held-out episodes."""

    def __init__(self, task, directory, checkpoint, *, build=True):
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.json_utils import _deserialize_params_value

        self.task, self.directory, self.checkpoint = task, Path(directory), Path(checkpoint)
        if build:
            build_ftp_cache(task, self.directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.ends = np.asarray([e["end"] for e in self.manifest["episodes"]])
        self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r")
                       for name in ("head", "wrist", "tactile_left", "tactile_right", "joint")}
        self.domain = next(iter(json.loads((self.checkpoint / "tactile_input_config_file.json").read_text())))
        config = json.loads((self.checkpoint / "train_config.json").read_text())
        if (config["action_joint_rep"], config["proprioception_joint_rep"], config["disable_history"]) != ("mix", "abs", True):
            raise ValueError("Unexpected released action/history contract")
        norm_dir = self.checkpoint / "normalization"
        domain_file = norm_dir / self.domain / "independent_norm_stats_all_t0_zscore.json"
        shared_file = norm_dir / "share_norm_stats_all_t0_zscore.json"
        self.norms = []
        for path in (domain_file, shared_file):
            value = json.loads(path.read_text())
            value["params"] = _deserialize_params_value(value["params"])
            # Native key inference expects unbatched [T,N,H,W,C]. Give the
            # exact same release stats an explicit runtime key for [B,T,N,H,W,C].
            canonical = "GelSightMini_image_224_224_3"
            for field in ("params","norm_dim","data_dim"):
                tactile_stats = value[field].get("tactile",{})
                if canonical in tactile_stats:
                    tactile_stats["right_tactile_gripper"] = tactile_stats[canonical]
            self.norms.append(value)
        self.tokens, self.token_mask = PaligemmaTokenizer(150).tokenize(self.manifest["prompt"])
        published = json.loads((norm_dir / "dataset_stats.json").read_text())
        published_frames = published["train_split"]["total"]["frames"]
        if published_frames != len(self) or published["val_split"]["total"]["trajectories"] != 0:
            raise ValueError(f"Published dataset differs: frames {published_frames} versus {len(self)}")
        self.contract = {"cache": self.manifest["cache_id"], "teacher": str(self.checkpoint),
                         "norm_hashes": [file_hash(domain_file), file_hash(shared_file)],
                         "model_config": file_hash(self.checkpoint / "model_config.json"),
                         "train_config": file_hash(self.checkpoint / "train_config.json")}

    def __len__(self):
        return self.manifest["total_steps"]

    def batch(self, ids, device="cuda"):
        import torch
        from openpi.models.model import Observation
        from openpi.normalization import apply_norm_stats

        ids = np.asarray(ids, np.int64)
        end = self.ends[np.searchsorted(self.ends, ids, side="right")]
        targets = np.minimum(ids[:, None] + np.arange(32), end[:, None]-1)
        joints = self.arrays["joint"][ids]
        state = np.zeros((len(ids), 1, 120), np.float64)
        state[:, 0, 9:16], state[:, 0, 44] = joints[:, :7], joints[:, 7]
        action = np.zeros((len(ids), 32, 120), np.float64)
        future = self.arrays["joint"][targets].astype(np.float64)
        # Native concatenation with zero slots promotes to float64 BEFORE
        # relative-arm subtraction, then native normalization casts to float32.
        action[..., 9:16] = future[..., :7] - joints[:, None, :7].astype(np.float64)
        action[..., 44] = future[..., 7]
        tactile = np.stack([self.arrays["tactile_left"][ids], self.arrays["tactile_right"][ids]], axis=1)[:, None]
        sample = {"state": state, "actions": action, "tactile": {"right_tactile_gripper": tactile.astype(np.float32)},
                  "tactile_sensor": {"right_tactile_gripper": "GelSightMini"},
                  "tactile_type": {"right_tactile_gripper": "image"},
                  "tactile_function_area": {"right_tactile_gripper": np.array([0,1])}}
        for norm in self.norms:
            sample = apply_norm_stats(sample, norm, unnormalize=False)
        tensor = lambda x, dtype=None: torch.as_tensor(np.asarray(x), dtype=dtype, device=device)
        mask = np.zeros_like(action, bool)
        mask[..., 9:16], mask[..., 44] = True, True
        obs = Observation(
            images={key: tensor(self.arrays[name][ids].astype(np.float32) * (2/255)-1)
                    for key, name in (("camera_ego_rgb_0", "head"), ("right_wrist_camera_rgb_0", "wrist"))},
            image_masks={key: torch.ones(len(ids), dtype=torch.bool, device=device)
                         for key in ("camera_ego_rgb_0", "right_wrist_camera_rgb_0")},
            state=tensor(sample["state"],torch.float32), state_masks=torch.ones(len(ids), 1, dtype=torch.bool, device=device),
            tokenized_prompt=tensor(np.broadcast_to(self.tokens, (len(ids), 150)).copy(), torch.long),
            tokenized_prompt_mask=tensor(np.broadcast_to(self.token_mask, (len(ids), 150)).copy(), torch.bool),
            tactiles={"right_tactile_gripper": tensor(sample["tactile"]["right_tactile_gripper"])},
            tactile_function_areas={"right_tactile_gripper": tensor(np.tile([0,1], (len(ids),1)), torch.long)},
            tactile_sensors={"right_tactile_gripper": ["GelSightMini"]*len(ids)},
            domain_names=[self.domain]*len(ids), action_masks=tensor(mask, torch.float32))
        return obs, tensor(sample["actions"],torch.float32)


class TaberoData:
    """Reuse immutable raw Field cache; apply original release transforms/stats."""

    def __init__(self, directory, norm_path):
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.shared.normalize import load as load_norm
        from openpi.transforms import Normalize, PadStatesAndActions

        self.directory = Path(directory)
        if not (self.directory / "COMPLETE.json").exists():
            raise ValueError("Incomplete Field cache")
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.ends = np.asarray([e["end"] for e in self.manifest["episodes"]])
        self.task_ids = np.concatenate([np.full(e["end"]-e["start"], e["task_index"], np.int32)
                                        for e in self.manifest["episodes"]])
        self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r")
                       for name in ("image", "wrist_image", "state", "action", "marker")}
        # Native normalize.load accepts directory containing norm_stats.json.
        self.normalize = Normalize(load_norm(Path(norm_path).parent))
        self.pad = PadStatesAndActions(32)
        tokenizer = PaligemmaTokenizer(48)
        pairs = [tokenizer.tokenize(text) for text in self.manifest["tasks"]]
        self.tokens = np.stack([p[0] for p in pairs]).astype(np.int32)
        self.token_mask = np.stack([p[1] for p in pairs])
        self.contract = {"cache": self.manifest["cache_id"], "norm_sha256": file_hash(norm_path),
                         "horizon": 50, "dim": 32, "native_effective_dim": 13,
                         "force_weight": .1, "padding_weight": 1., "history": "reference+8"}

    def __len__(self):
        return self.manifest["total_steps"]

    def batch(self, ids):
        from openpi.models.model import Observation
        import jax.numpy as jnp

        ids = np.asarray(ids, np.int64)
        ends = self.ends[np.searchsorted(self.ends, ids, side="right")]
        indices = np.minimum(ids[:, None] + np.arange(50), ends[:, None]-1)
        state = self.arrays["state"][ids].copy()
        action = self.arrays["action"][indices].copy()
        action[..., :6] -= state[:, None, :6]
        data = self.pad(self.normalize({"state": state, "actions": action,
                                      "tactile_prefix": self.arrays["marker"][ids].reshape(len(ids), 9, 396).copy()}))
        images = {key: np.moveaxis(self.arrays[name][ids], 1, -1).astype(np.float32) * (2/255)-1
                  for key, name in (("base_0_rgb", "image"), ("left_wrist_0_rgb", "wrist_image"))}
        images["right_wrist_0_rgb"] = np.zeros_like(images["base_0_rgb"])
        masks = {key: np.full(len(ids), key != "right_wrist_0_rgb", bool) for key in images}
        tasks = self.task_ids[ids]
        obs = Observation(images={k: jnp.asarray(v) for k,v in images.items()},
                          image_masks={k: jnp.asarray(v) for k,v in masks.items()},
                          state=jnp.asarray(data["state"]), tactile_prefix=jnp.asarray(data["tactile_prefix"]),
                          tokenized_prompt=jnp.asarray(self.tokens[tasks]),
                          tokenized_prompt_mask=jnp.asarray(self.token_mask[tasks]))
        return obs, jnp.asarray(data["actions"])
