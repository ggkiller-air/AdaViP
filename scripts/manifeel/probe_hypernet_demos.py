#!/usr/bin/env python3
"""Probe HyperResNet modulation on recorded demonstrations using CPU only."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import dill
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

from adavip.manifeel.hypernet_probe import (
    HyperNetProbeRecorder,
    select_episode_frames,
    summarize_probe,
    temporal_modulation_ratio,
)


def load_encoder(checkpoint: Path) -> tuple[torch.nn.Module, dict, dict]:
    """Load only the saved encoder and observation normalization from a trusted checkpoint."""
    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    encoder_cfg = cfg.policy.obs_encoder
    expected_target = "adavip.manifeel.hyper_resnet_obs_encoder.HyperResNetObsEncoder"
    if encoder_cfg._target_ != expected_target:
        raise ValueError(f"expected {expected_target}, got {encoder_cfg._target_}")
    source = "ema_model" if cfg.training.use_ema else "model"
    state = payload["state_dicts"][source]
    encoder = hydra.utils.instantiate(encoder_cfg).cpu().eval()
    prefix = "obs_encoder."
    encoder.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                             if key.startswith(prefix)}, strict=True)
    normalizers = {}
    for key in encoder.key_shape_map:
        normalizer_prefix = f"normalizer.params_dict.{key}."
        normalizers[key] = (state[normalizer_prefix + "scale"].clone(),
                            state[normalizer_prefix + "offset"].clone())
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "weights": source,
        "dataset_from_checkpoint": str(cfg.task.dataset.zarr_path),
        "encoder_config": OmegaConf.to_container(encoder_cfg, resolve=True),
        "alpha": float(encoder.alpha.detach()),
        "fusion_gate_tanh": float(encoder.fusion_gate.detach().tanh()),
        "use_fusion_hypernet": bool(encoder.use_fusion_hypernet),
    }
    if encoder.use_fusion_hypernet:
        metadata["fusion_alpha"] = float(encoder.fusion_alpha.detach())
    del payload, state
    gc.collect()
    return encoder, normalizers, metadata


def read_observations(data: object, indices: np.ndarray, encoder: torch.nn.Module,
                      normalizers: dict) -> dict[str, torch.Tensor]:
    """Match upstream image resizing and checkpoint-saved observation normalization."""
    observations = {}
    for key, shape in encoder.key_shape_map.items():
        values = np.asarray(data[key].oindex[indices.tolist()], dtype=np.float32)
        if key in encoder.rgb_keys:
            if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                raise ValueError(f"{key} demonstrations must contain finite RGB values in [0, 1]")
            values = np.stack([cv2.resize(image, (shape[2], shape[1])) for image in values])
            values = np.moveaxis(values, -1, 1).copy()
        tensor = torch.from_numpy(values)
        scale, offset = normalizers[key]
        if key in encoder.rgb_keys:
            scale = scale.reshape(-1, 1, 1)
            offset = offset.reshape(-1, 1, 1)
        observations[key] = tensor * scale + offset
        if not torch.isfinite(observations[key]).all():
            raise ValueError(f"nonfinite normalized observation: {key}")
    return observations


def plot_episode(arrays: dict[str, np.ndarray], episode: int, output: Path) -> None:
    """Plot internal signals against recorded frame indices, without inferred phase labels."""
    selected = arrays["episode"] == episode
    steps = arrays["episode_step"][selected]
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    for modality_index, modality in enumerate(("vision", "tactile")):
        color = ("tab:blue", "tab:orange")[modality_index]
        axes[0].plot(steps, 100 * arrays["residual_ratio"][selected, modality_index],
                     color=color, label=modality)
        axes[0].plot(steps, 100 * arrays["dynamic_residual_ratio"][selected, modality_index],
                     color=color, linestyle="--", label=f"{modality}: deviation from episode-mean modulation")
    axes[0].set_ylabel("Modulation / feature (%)")
    axes[0].legend()
    if "attention" in arrays:
        head_count = arrays["attention"].shape[1]
        image = axes[1].pcolormesh(steps, np.arange(head_count),
                                   arrays["attention"][selected, :, 1].T,
                                   shading="nearest", vmin=0, vmax=1, cmap="coolwarm")
        axes[1].set_yticks(np.arange(head_count))
        figure.colorbar(image, ax=axes[1], label="Attention to TacRGB")
    else:
        axes[1].text(0.5, 0.5, "Fusion disabled", transform=axes[1].transAxes, ha="center")
    axes[1].set_ylabel("Attention head")
    if "force_field_norm_mean" in arrays:
        axes[2].plot(steps, arrays["force_field_norm_mean"][selected], color="black")
    axes[2].set_ylabel("Recorded force-field\nmean vector norm")
    axes[3].plot(steps, arrays["normalized_state_norm"][selected], color="gray")
    axes[3].set_ylabel("Normalized state norm")
    axes[3].set_xlabel("Demonstration frame (not simulator rollout)")
    figure.suptitle(f"Episode {episode}: descriptive internal signals, not modality trust")
    figure.savefig(output / f"episode{episode}_signals.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for modality_index, modality in enumerate(("vision", "tactile")):
        for parameter_index, parameter in enumerate(("gamma", "beta")):
            values = arrays[parameter][selected, modality_index].T
            image = axes[modality_index, parameter_index].imshow(
                values, origin="lower", aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
            axes[modality_index, parameter_index].set_title(f"{modality}: {parameter}")
            axes[modality_index, parameter_index].set_ylabel("Feature channel")
            axes[modality_index, parameter_index].set_xlabel("Sampled frame index")
            figure.colorbar(image, ax=axes[modality_index, parameter_index])
    figure.suptitle(f"Episode {episode}: bounded HyperNet outputs (before alpha scaling)")
    figure.savefig(output / f"episode{episode}_modulation.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for modality_index, modality in enumerate(("vision", "tactile")):
        for parameter_index, parameter in enumerate(("gamma", "beta")):
            values = arrays[parameter][selected, modality_index].T
            centered = values - values.mean(axis=1, keepdims=True)
            limit = max(float(np.quantile(np.abs(centered), 0.99)), 1e-3)
            image = axes[modality_index, parameter_index].imshow(
                centered, origin="lower", aspect="auto", vmin=-limit, vmax=limit, cmap="coolwarm")
            axes[modality_index, parameter_index].set_title(f"{modality}: centered {parameter}")
            axes[modality_index, parameter_index].set_ylabel("Feature channel")
            axes[modality_index, parameter_index].set_xlabel("Sampled frame index")
            figure.colorbar(image, ax=axes[modality_index, parameter_index])
    figure.suptitle(f"Episode {episode}: temporal deviations from each channel's episode mean")
    figure.savefig(output / f"episode{episode}_modulation_centered.png", dpi=160)
    plt.close(figure)

    if "fusion_gamma" in arrays:
        figure, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
        for parameter_index, parameter in enumerate(("fusion_gamma", "fusion_beta")):
            image = axes[parameter_index].imshow(
                arrays[parameter][selected].T, origin="lower", aspect="auto",
                vmin=-1, vmax=1, cmap="coolwarm")
            axes[parameter_index].set_title(parameter)
            axes[parameter_index].set_ylabel("Feature channel")
            axes[parameter_index].set_xlabel("Sampled frame index")
            figure.colorbar(image, ax=axes[parameter_index])
        axes[2].plot(steps, 100 * arrays["fusion_hypernet_residual_ratio"][selected])
        axes[2].set_ylabel("Extra modulation / feature (%)")
        axes[2].set_xlabel("Demonstration frame")
        figure.suptitle(f"Episode {episode}: extra fusion HyperNet, descriptive only")
        figure.savefig(output / f"episode{episode}_fusion_hypernet.png", dpi=160)
        plt.close(figure)


def save_montage(data: object, encoder: torch.nn.Module, arrays: dict[str, np.ndarray],
                 episode: int, output: Path) -> None:
    """Save synchronized policy-input views at a fixed set of demonstration times."""
    selected = np.flatnonzero(arrays["episode"] == episode)
    chosen = selected[np.unique(np.linspace(0, len(selected) - 1, min(5, len(selected)), dtype=int))]
    figure, axes = plt.subplots(2, len(chosen), figsize=(3 * len(chosen), 6),
                                squeeze=False, layout="constrained")
    for column, sample in enumerate(chosen):
        for row, key in enumerate((encoder.vision_key, encoder.tactile_key)):
            image = data[key][int(arrays["dataset_index"][sample])]
            shape = encoder.key_shape_map[key]
            axes[row, column].imshow(np.clip(cv2.resize(image, (shape[2], shape[1])), 0, 1))
            axes[row, column].set_title(f"{key}\nframe {arrays['episode_step'][sample]}")
            axes[row, column].axis("off")
    figure.suptitle(f"Episode {episode}: recorded inputs (not saliency maps)")
    figure.savefig(output / f"episode{episode}_inputs.png", dpi=140)
    plt.close(figure)


def main() -> None:
    """Run a bounded CPU-only demonstration probe and save reproducible figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames-per-episode", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch size and thread count must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Probe output already exists: {args.output_dir}")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    encoder, normalizers, metadata = load_encoder(args.checkpoint)
    dataset = args.dataset or Path(metadata["dataset_from_checkpoint"])
    root = zarr.open(str(dataset), mode="r")
    indices, episodes, steps = select_episode_frames(
        root["meta/episode_ends"][:], args.episodes, args.frames_per_episode)
    args.output_dir.mkdir(parents=True)
    recorder = HyperNetProbeRecorder(encoder)
    state_norms = []
    try:
        with torch.inference_mode():
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start:start + args.batch_size]
                observations = read_observations(root["data"], batch_indices, encoder, normalizers)
                state_norms.extend(observations[encoder.state_key].norm(dim=-1).tolist())
                encoder(observations)
                print(f"CPU frames: {min(start + args.batch_size, len(indices))}/{len(indices)}", flush=True)
        arrays = recorder.arrays()
    finally:
        recorder.close()
    arrays.update({"dataset_index": indices, "episode": episodes, "episode_step": steps,
                   "normalized_state_norm": np.asarray(state_norms)})
    arrays["dynamic_residual_ratio"] = temporal_modulation_ratio(arrays, metadata["alpha"])
    force_key = "tactile_force_field_right"
    if force_key in root["data"]:
        force = root["data"][force_key].oindex[indices.tolist()]
        arrays["force_field_norm_mean"] = np.linalg.norm(force, axis=-1).reshape(len(indices), -1).mean(axis=1)
    np.savez_compressed(args.output_dir / "hypernet_demo_trace.npz", **arrays)
    metadata.update({"dataset": str(dataset.resolve()), "device": "cpu", "threads": args.threads,
                     "batch_size": args.batch_size, "episodes": args.episodes,
                     "frames_per_episode": args.frames_per_episode,
                     "scope": "Recorded demonstrations; no action prediction or closed-loop evaluation.",
                     "caveat": "Modulation magnitude and attention are not causal modality trust. No automatic phase labels.",
                     "summary": summarize_probe(arrays),
                     "episode_summaries": {str(episode): summarize_probe(
                         {key: value[episodes == episode] for key, value in arrays.items()})
                         for episode in args.episodes}})
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for episode in args.episodes:
        plot_episode(arrays, episode, args.output_dir)
        save_montage(root["data"], encoder, arrays, episode, args.output_dir)
    print(json.dumps(metadata["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
