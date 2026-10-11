#!/usr/bin/env python3
"""Measure paired-noise Power Plug action sensitivity on recorded histories."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import dill
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

from adavip.manifeel.action_probe import (
    capture_hypernet_output,
    normalized_action_change,
    observation_windows,
    override_hypernet,
    perturb_observations,
    seeded_prediction,
)
from scripts.manifeel.probe_hypernet_demos import read_observations


VARIANTS = ("repeat", "vision_blur", "tactile_blur", "vision_delay", "tactile_delay",
            "state_delay", "mean_hypernet", "lagged_hypernet", "zero_hypernet")


def load_policy(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Restore saved policy, scheduler configuration and normalizers without a simulator."""
    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    source = "ema_model" if cfg.training.use_ema else "model"
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][source], strict=True)
    metadata = {"checkpoint": str(checkpoint.resolve()), "weights": source,
                "policy_config": OmegaConf.to_container(cfg.policy, resolve=True),
                "dataset_from_checkpoint": str(cfg.task.dataset.zarr_path)}
    del payload
    gc.collect()
    return policy.to(device).eval(), metadata


def read_histories(data: object, windows: np.ndarray, policy: torch.nn.Module,
                    device: torch.device) -> dict[str, torch.Tensor]:
    """Read raw histories; policy.predict_action applies the saved normalizer once."""
    identity = {key: (torch.ones(1), torch.zeros(1)) for key in policy.obs_encoder.key_shape_map}
    flattened = read_observations(data, windows.reshape(-1), policy.obs_encoder, identity)
    return {key: value.reshape(*windows.shape, *value.shape[1:]).to(device)
            for key, value in flattened.items()}


def plot_sensitivity(arrays: dict[str, np.ndarray], variants: list[str], output: Path) -> None:
    """Plot paired action changes without converting elapsed time into task phases."""
    episodes = np.unique(arrays["episode"])
    figure, axes = plt.subplots(len(episodes), 1, figsize=(13, 3.5 * len(episodes)),
                                squeeze=False, layout="constrained")
    for row_index, episode in enumerate(episodes):
        selected = arrays["episode"] == episode
        for variant_index, variant in enumerate(variants):
            values = arrays["rms_all"][:, selected, variant_index].mean(axis=0)
            axes[row_index, 0].plot(arrays["episode_step"][selected], values,
                                    marker="o", markersize=3, label=variant)
        axes[row_index, 0].set_ylabel(f"Episode {episode}\nNormalized action RMS change")
        axes[row_index, 0].set_xlabel("Recorded demonstration frame; no inferred phase labels")
        axes[row_index, 0].legend(ncol=3, fontsize=8)
    figure.suptitle("Open-loop action sensitivity; paired diffusion noise, not success-rate effects")
    figure.savefig(output / "action_sensitivity.png", dpi=170)
    plt.close(figure)


def main() -> None:
    """Run a bounded GPU action probe with independent calibration demonstrations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--lag", type=int, default=10)
    parser.add_argument("--calibration-episodes", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if args.batch_size <= 0 or args.lag < 0 or not args.seeds:
        parser.error("batch size must be positive, lag nonnegative, and seeds nonempty")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("this action probe requires an explicitly approved CUDA device")
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    policy, metadata = load_policy(args.checkpoint, device)
    has_hypernet = hasattr(policy.obs_encoder, "hypernet")
    if not has_hypernet and any("hypernet" in variant for variant in args.variants):
        parser.error("HyperNet interventions require a HyperResNet checkpoint")
    dataset = args.dataset or Path(metadata["dataset_from_checkpoint"])
    root = zarr.open(str(dataset), mode="r")
    episode_ends = root["meta/episode_ends"][:]
    windows, episodes, steps = observation_windows(
        episode_ends, args.episodes, args.samples_per_episode, policy.n_obs_steps)
    delayed_windows, _, _ = observation_windows(
        episode_ends, args.episodes, args.samples_per_episode, policy.n_obs_steps, args.lag)
    fixed_modulation = None
    if "mean_hypernet" in args.variants:
        if set(args.episodes) & set(args.calibration_episodes):
            parser.error("calibration and probe episodes must be disjoint")
        calibration, _, _ = observation_windows(
            episode_ends, args.calibration_episodes, args.calibration_samples, policy.n_obs_steps)
        generated = []
        for start in range(0, len(calibration), args.batch_size):
            observations = read_histories(root["data"], calibration[start:start + args.batch_size], policy, device)
            generated.append(capture_hypernet_output(policy, observations).tanh())
        mean_modulation = torch.cat(generated).mean(dim=0, keepdim=True)
        fixed_modulation = mean_modulation.clamp(-1 + 1e-6, 1 - 1e-6).atanh()
    args.output_dir.mkdir(parents=True)
    action_scale = policy.normalizer["action"].params_dict["scale"].detach()
    arrays = {"episode": episodes, "episode_step": steps, "observation_indices": windows,
              "delayed_observation_indices": delayed_windows,
              "seeds": np.asarray(args.seeds), "variants": np.asarray(args.variants),
              "action_scale": action_scale.cpu().numpy()}
    changes = {name: np.zeros((len(args.seeds), len(windows), len(args.variants)), dtype=np.float32)
               for name in ("all", "translation", "rotation")}
    actions = np.zeros((len(args.seeds), len(windows), len(args.variants) + 1,
                        policy.n_action_steps, policy.action_dim), dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, len(windows), args.batch_size):
        stop = min(start + args.batch_size, len(windows))
        observations = read_histories(root["data"], windows[start:stop], policy, device)
        delayed = read_histories(root["data"], delayed_windows[start:stop], policy, device)
        lagged_modulation = capture_hypernet_output(policy, delayed) if "lagged_hypernet" in args.variants else None
        keys = {"vision": getattr(policy.obs_encoder, "vision_key", "wrist"),
                "tactile": getattr(policy.obs_encoder, "tactile_key", "right_tactile_camera_taxim"),
                "state": getattr(policy.obs_encoder, "state_key", "state")}
        for seed_index, seed in enumerate(args.seeds):
            paired_seed = seed + 100000 * start
            original = seeded_prediction(policy, observations, paired_seed)
            actions[seed_index, start:stop, 0] = original.cpu().numpy()
            for variant_index, variant in enumerate(args.variants):
                changed_observations = observations
                replacement = None
                if variant.endswith(("_blur", "_delay")):
                    modality, mode = variant.split("_")
                    changed_observations = perturb_observations(observations, delayed, keys[modality], mode)
                elif variant == "mean_hypernet":
                    replacement = fixed_modulation
                elif variant == "lagged_hypernet":
                    replacement = lagged_modulation
                elif variant == "zero_hypernet":
                    replacement = torch.zeros(1, policy.obs_encoder.hypernet[-1].out_features, device=device)
                with override_hypernet(policy.obs_encoder, replacement):
                    changed = seeded_prediction(policy, changed_observations, paired_seed)
                if variant == "repeat" and not torch.equal(original, changed):
                    raise ValueError("fixed-noise original/repeat action predictions differ")
                actions[seed_index, start:stop, variant_index + 1] = changed.cpu().numpy()
                for name, value in normalized_action_change(original, changed, action_scale).items():
                    changes[name][seed_index, start:stop, variant_index] = value.cpu().numpy()
        print(f"Histories {stop}/{len(windows)}; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    arrays.update({f"rms_{name}": value for name, value in changes.items()})
    arrays["actions"] = actions
    if fixed_modulation is not None:
        arrays["calibration_mean_gamma_beta"] = fixed_modulation.tanh().cpu().numpy()
    np.savez_compressed(args.output_dir / "action_sensitivity.npz", **arrays)
    summary = {variant: {name: float(values[..., variant_index].mean())
                         for name, values in changes.items()}
               for variant_index, variant in enumerate(args.variants)}
    metadata.update({"dataset": str(dataset.resolve()), "device": str(device),
                     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "gpu_name": torch.cuda.get_device_name(device),
                     "episodes": args.episodes, "samples_per_episode": args.samples_per_episode,
                     "seeds": args.seeds, "variants": args.variants, "lag_frames": args.lag,
                     "calibration_episodes": args.calibration_episodes if fixed_modulation is not None else [],
                     "calibration_samples_per_episode": args.calibration_samples,
                     "batch_size": args.batch_size, "threads": 2, "num_workers": 0,
                     "n_obs_steps": policy.n_obs_steps, "num_inference_steps": policy.num_inference_steps,
                     "elapsed_prediction_seconds": time.perf_counter() - started,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
                     "scope": "Open-loop paired-noise inference interventions; no inferred phases or new success rate.",
                     "summary": summary})
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_sensitivity(arrays, args.variants, args.output_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
