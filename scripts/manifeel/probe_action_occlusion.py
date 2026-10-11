#!/usr/bin/env python3
"""Create paired-noise action occlusion maps on fixed Power Plug histories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

from adavip.manifeel.action_probe import normalized_action_change, patch_blur_observations, seeded_prediction
from scripts.manifeel.probe_action_sensitivity import load_policy, read_histories


def main() -> None:
    """Blur patches, replay diffusion randomness per pair and plot action changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--steps", nargs="+", type=int, default=[33, 102, 170])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if args.batch_size <= 0 or args.grid_size <= 0 or not args.seeds:
        parser.error("positive batch/grid sizes and nonempty seeds are required")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("an explicitly approved CUDA device is required")
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    policy, metadata = load_policy(args.checkpoint, device)
    dataset = args.dataset or Path(metadata["dataset_from_checkpoint"])
    root = zarr.open(str(dataset), mode="r")
    ends = root["meta/episode_ends"][:]
    if not 0 <= args.episode < len(ends):
        parser.error("episode is outside the dataset")
    episode_start = int(ends[args.episode - 1]) if args.episode else 0
    length = int(ends[args.episode]) - episode_start
    if any(not 0 <= step < length for step in args.steps):
        parser.error("snapshot step is outside the selected episode")
    keys = (getattr(policy.obs_encoder, "vision_key", "wrist"),
            getattr(policy.obs_encoder, "tactile_key", "right_tactile_camera_taxim"))
    shape = (len(args.seeds), len(args.steps), 2, args.grid_size, args.grid_size)
    changes = {metric: np.zeros(shape, dtype=np.float32) for metric in ("all", "translation", "rotation")}
    snapshots = []
    action_scale = policy.normalizer["action"].params_dict["scale"].detach()
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    for step_index, step in enumerate(args.steps):
        window = episode_start + np.maximum(np.arange(step - policy.n_obs_steps + 1, step + 1), 0)
        observations = read_histories(root["data"], window[None], policy, device)
        snapshots.append(np.stack([observations[key][0, -1].cpu().numpy() for key in keys]))
        for modality_index, key in enumerate(keys):
            height, width = observations[key].shape[-2:]
            if args.grid_size > min(height, width):
                parser.error("grid size exceeds image dimensions")
            rows = np.linspace(0, height, args.grid_size + 1, dtype=int)
            columns = np.linspace(0, width, args.grid_size + 1, dtype=int)
            boxes = [(int(rows[row]), int(rows[row + 1]), int(columns[column]), int(columns[column + 1]))
                     for row in range(args.grid_size) for column in range(args.grid_size)]
            for start in range(0, len(boxes), args.batch_size):
                selected = boxes[start:start + args.batch_size]
                changed = patch_blur_observations(observations, key, selected)
                original = {name: value.expand(len(selected), *value.shape[1:])
                            for name, value in observations.items()}
                for seed_index, seed in enumerate(args.seeds):
                    paired_seed = seed + 100000 * step_index
                    before = seeded_prediction(policy, original, paired_seed)
                    after = seeded_prediction(policy, changed, paired_seed)
                    for metric, values in normalized_action_change(before, after, action_scale).items():
                        changes[metric][seed_index, step_index, modality_index].reshape(-1)[start:start + len(selected)] = values.cpu().numpy()
        print(f"Snapshot frame {step}: completed; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    arrays = {f"rms_{metric}": values for metric, values in changes.items()}
    arrays.update({"inputs": np.stack(snapshots), "steps": np.asarray(args.steps), "seeds": np.asarray(args.seeds),
                   "modalities": np.asarray(keys)})
    np.savez_compressed(args.output_dir / "action_occlusion.npz", **arrays)
    for metric, values in changes.items():
        mean = values.mean(axis=0)
        maximum = max(float(mean.max()), 1e-8)
        figure, axes = plt.subplots(2, len(args.steps), figsize=(4 * len(args.steps), 7),
                                    squeeze=False, layout="constrained")
        for step_index, step in enumerate(args.steps):
            for modality_index, key in enumerate(keys):
                axis = axes[modality_index, step_index]
                image = np.moveaxis(arrays["inputs"][step_index, modality_index], 0, -1)
                axis.imshow(np.clip(image, 0, 1))
                overlay = axis.imshow(mean[step_index, modality_index], extent=(0, image.shape[1], image.shape[0], 0),
                                       interpolation="nearest", cmap="magma", vmin=0, vmax=maximum, alpha=0.55)
                axis.set_title(f"{key}\nepisode {args.episode}, frame {step}")
                axis.axis("off")
        figure.colorbar(overlay, ax=axes.ravel().tolist(), label=f"Normalized {metric} action RMS change")
        figure.suptitle(f"Action occlusion: {args.grid_size}x{args.grid_size} patch blur; paired noise, no phase labels")
        figure.savefig(args.output_dir / f"action_occlusion_{metric}.png", dpi=170)
        plt.close(figure)
    metadata.update({"dataset": str(dataset.resolve()), "episode": args.episode, "steps": args.steps,
                     "seeds": args.seeds, "grid_size": args.grid_size, "batch_size": args.batch_size,
                     "blur_kernel": 31, "blur_sigma": 10.0, "history_patch_mode": "same spatial patch in all history frames",
                     "noise_pairing": "original and changed use identical RNG seed and batch shape; noise varies by batch position",
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "elapsed_seconds": time.perf_counter() - started,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                     "scope": "Spatial sensitivity of predicted actions, not attention, causal utility or success-rate attribution."})
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
