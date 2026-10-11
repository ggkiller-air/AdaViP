#!/usr/bin/env python3
"""Probe action effects of modulation swaps on physically staged rollout conditions."""

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

from adavip.manifeel.action_probe import normalized_action_change, reconstruct_condition, seeded_condition_prediction
from adavip.manifeel.contact_probe import contact_phase_masks
from scripts.manifeel.probe_action_sensitivity import load_policy
from scripts.manifeel.summarize_power_plug_probes import load_arrays


VARIANTS = ("repeat", "mean_hypernet", "mean_vision", "mean_tactile", "zero_hypernet",
            "lagged_hypernet", "cross_phase_hypernet")


def select_contact_pairs(phases: dict, steps: np.ndarray, maximum: int,
                          margin: int) -> tuple[np.ndarray, np.ndarray]:
    """Select one pre/post-contact condition per eligible environment, regardless of outcome."""
    chunks, environments = [], []
    eligible = np.flatnonzero(phases["pre_contact"].any(axis=0)
                              & phases["post_contact_pre_success"].any(axis=0))[:maximum]
    for environment in eligible:
        contact = phases["first_contact_step"][environment]
        before = np.flatnonzero(phases["pre_contact"][:, environment])
        after = np.flatnonzero(phases["post_contact_pre_success"][:, environment])
        distant_before = before[steps[before] <= contact - margin]
        distant_after = after[steps[after] >= contact + margin]
        chunks.extend([int(distant_before[-1] if len(distant_before) else before[-1]),
                        int(distant_after[0] if len(distant_after) else after[0])])
        environments.extend([environment, environment])
    return np.asarray(chunks, dtype=int), np.asarray(environments, dtype=int)


def main() -> None:
    """Use cached features to isolate modulation while retaining the original action sampler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--calibration-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-environments", type=int, default=20)
    parser.add_argument("--margin-steps", type=int, default=16)
    parser.add_argument("--contact-threshold", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if args.batch_size <= 0 or args.max_environments <= 0 or args.margin_steps < 0:
        parser.error("batch/environment counts must be positive and margin nonnegative")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("an explicitly approved CUDA device is required")
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    policy, metadata = load_policy(args.checkpoint, device)
    trace = load_arrays(args.run / "hypernet_rollout_trace.npz")
    rollout = load_arrays(args.run / "rollout_trace.npz")
    phases = contact_phase_masks(rollout["reward_by_step"], rollout["plug_socket_force"],
        rollout["initial_plug_socket_force"], trace["chunk_start_sim_step"], args.contact_threshold)
    chunks, environments = select_contact_pairs(phases, trace["chunk_start_sim_step"],
                                                args.max_environments, args.margin_steps)
    if not len(chunks):
        raise ValueError("no environments have both pre-contact and pre-success post-contact policy inputs")
    calibration = load_arrays(args.calibration_trace)["calibration_mean_gamma_beta"]
    width = trace["base_features"].shape[-1]
    gamma_vision, beta_vision, gamma_tactile, beta_tactile = np.split(calibration, 4, axis=-1)
    mean_gamma = torch.tensor(np.stack((gamma_vision[0], gamma_tactile[0])), device=device)
    mean_beta = torch.tensor(np.stack((beta_vision[0], beta_tactile[0])), device=device)
    action_scale = policy.normalizer["action"].params_dict["scale"].detach()
    changes = np.zeros((len(args.seeds), len(chunks), len(VARIANTS)), dtype=np.float32)
    reconstructed_error = 0.0
    args.output_dir.mkdir(parents=True)
    started = time.perf_counter()
    for start in range(0, len(chunks), args.batch_size):
        stop = min(start + args.batch_size, len(chunks))
        selected_chunks, selected_envs = chunks[start:stop], environments[start:stop]
        histories = policy.n_obs_steps
        batch_size = stop - start
        features = torch.tensor(trace["base_features"][selected_chunks, selected_envs], device=device).reshape(-1, 2, width)
        gamma = torch.tensor(trace["gamma"][selected_chunks, selected_envs], device=device).reshape(-1, 2, width)
        beta = torch.tensor(trace["beta"][selected_chunks, selected_envs], device=device).reshape(-1, 2, width)
        state = torch.tensor(trace["normalized_state"][selected_chunks, selected_envs], device=device).reshape(-1, 7)
        encoded = torch.tensor(trace["encoder_output"][selected_chunks, selected_envs], device=device)
        swapped = np.arange(start, stop) ^ 1
        lagged_chunks = np.maximum(selected_chunks - 1, 0)
        replacements = {
            "lagged_hypernet": (trace["gamma"][lagged_chunks, selected_envs], trace["beta"][lagged_chunks, selected_envs]),
            "cross_phase_hypernet": (trace["gamma"][chunks[swapped], environments[swapped]],
                                      trace["beta"][chunks[swapped], environments[swapped]])}
        with torch.inference_mode():
            exact = reconstruct_condition(policy.obs_encoder, features, gamma, beta, state).reshape(batch_size, histories, -1)
            error = float((exact - encoded).abs().max())
            reconstructed_error = max(reconstructed_error, error)
            if error > 1e-5:
                raise ValueError(f"cached reconstruction disagrees with recorded encoder output: {error}")
            for seed_index, seed in enumerate(args.seeds):
                paired_seed = seed + 100000 * start
                original = seeded_condition_prediction(policy, exact, paired_seed)
                for variant_index, variant in enumerate(VARIANTS):
                    changed_gamma, changed_beta = gamma.clone(), beta.clone()
                    if variant == "mean_hypernet":
                        changed_gamma[:] = mean_gamma
                        changed_beta[:] = mean_beta
                    elif variant in ("mean_vision", "mean_tactile"):
                        modality = 0 if variant == "mean_vision" else 1
                        changed_gamma[:, modality] = mean_gamma[modality]
                        changed_beta[:, modality] = mean_beta[modality]
                    elif variant == "zero_hypernet":
                        changed_gamma.zero_()
                        changed_beta.zero_()
                    elif variant in replacements:
                        changed_gamma = torch.tensor(replacements[variant][0], device=device).reshape(-1, 2, width)
                        changed_beta = torch.tensor(replacements[variant][1], device=device).reshape(-1, 2, width)
                    condition = reconstruct_condition(policy.obs_encoder, features, changed_gamma, changed_beta, state).reshape(batch_size, histories, -1)
                    changed = seeded_condition_prediction(policy, condition, paired_seed)
                    if variant == "repeat" and not torch.equal(original, changed):
                        raise ValueError("reconstructed-condition repeat action disagrees with original condition")
                    changes[seed_index, start:stop, variant_index] = normalized_action_change(original, changed, action_scale)["all"].cpu().numpy()
        print(f"Contact-staged histories {stop}/{len(chunks)}; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    np.savez_compressed(args.output_dir / "contact_condition_actions.npz", rms_all=changes,
        environments=environments, chunks=chunks, simulator_steps=trace["chunk_start_sim_step"][chunks],
        phases=np.tile(np.asarray(["pre_contact", "post_contact_pre_success"]), len(chunks) // 2),
        variants=np.asarray(VARIANTS), seeds=np.asarray(args.seeds))
    summary = {}
    for variant_index, variant in enumerate(VARIANTS):
        paired = changes[:, :, variant_index].mean(axis=0).reshape(-1, 2)
        summary[variant] = {"pre_contact_mean": float(paired[:, 0].mean()),
                            "post_contact_mean": float(paired[:, 1].mean()),
                            "paired_post_minus_pre_mean": float((paired[:, 1] - paired[:, 0]).mean())}
    figure, axis = plt.subplots(figsize=(14, 5), layout="constrained")
    positions = np.arange(len(VARIANTS))
    for phase_index, phase in enumerate(("pre_contact", "post_contact_pre_success")):
        values = [summary[variant]["pre_contact_mean" if phase_index == 0 else "post_contact_mean"] for variant in VARIANTS]
        bars = axis.bar(positions + (phase_index - 0.5) * 0.35, values, width=0.35, label=phase)
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    axis.set_xticks(positions, VARIANTS, rotation=20, ha="right")
    axis.set_ylabel("Paired-noise normalized action RMS change")
    axis.legend()
    axis.set_title("Modulation-only action interventions on contact-staged rollout conditions; not success-rate effects")
    figure.savefig(args.output_dir / "contact_condition_actions.png", dpi=170)
    plt.close(figure)
    metadata.update({"run": str(args.run.resolve()), "calibration_trace": str(args.calibration_trace.resolve()),
        "selected_environments": environments[::2].tolist(), "margin_steps": args.margin_steps,
        "contact_threshold": args.contact_threshold, "seeds": args.seeds, "batch_size": args.batch_size,
        "reconstruction_max_abs_error": reconstructed_error,
        "paired_control": "Dynamic reconstructed condition, not the differently batched saved condition; repeat must be bitwise identical.",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Current raw features, state, alpha and fusion parameters are held fixed; only generated gamma/beta change, with downstream fusion recomputed. No new task success measurement.",
        "summary": summary})
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
