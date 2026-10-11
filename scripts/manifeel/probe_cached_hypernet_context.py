#!/usr/bin/env python3
"""Isolate HyperNet context dependence using saved features and CPU-only encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from adavip.manifeel.action_probe import reconstruct_condition
from adavip.manifeel.contact_probe import contact_phase_masks, environment_level_summary
from adavip.manifeel.mechanism_probe import phase_matched_donors
from scripts.manifeel.probe_hypernet_demos import load_encoder


def main() -> None:
    """Change only the context seen by HyperNet, not the actual raw modality features."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(2)
    encoder, _, metadata = load_encoder(args.checkpoint)
    if encoder.use_fusion_hypernet:
        raise ValueError("this probe isolates encoder HyperNet, not an additional fusion HyperNet")
    with np.load(args.run / "hypernet_rollout_trace.npz", allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    with np.load(args.run / "rollout_trace.npz", allow_pickle=False) as archive:
        rollout = {key: archive[key] for key in archive.files}
    phases = contact_phase_masks(rollout["reward_by_step"], rollout["plug_socket_force"],
        rollout["initial_plug_socket_force"], trace["chunk_start_sim_step"], 0.001)
    masks = [phases["pre_contact"], phases["post_contact_pre_success"]]
    donors, paired = phase_matched_donors(masks)
    chunks, environments = np.where(paired)
    if not len(chunks):
        raise ValueError("no same-time same-phase context pairs")
    features = torch.from_numpy(trace["base_features"][chunks, environments, -1])
    state = torch.from_numpy(trace["normalized_state"][chunks, environments, -1])
    donor_features = torch.from_numpy(trace["base_features"][chunks, donors[chunks, environments], -1])
    donor_state = torch.from_numpy(trace["normalized_state"][chunks, donors[chunks, environments], -1])
    width = features.shape[-1]
    context = torch.cat((functional.layer_norm(features[:, 0], (width,)),
                         functional.layer_norm(features[:, 1], (width,)), state), dim=-1)
    donor_context = torch.cat((functional.layer_norm(donor_features[:, 0], (width,)),
                               functional.layer_norm(donor_features[:, 1], (width,)), donor_state), dim=-1)
    variants = {"repeat": context.clone()}
    for label, start, stop in (("vision_context_swap", 0, width), ("tactile_context_swap", width, 2 * width),
                               ("state_context_swap", 2 * width, context.shape[-1])):
        changed = context.clone()
        changed[:, start:stop] = donor_context[:, start:stop]
        variants[label] = changed
    variants["mean_test_context_offline"] = context.mean(dim=0, keepdim=True).expand_as(context)
    variants["zero_state_context"] = context.clone()
    variants["zero_state_context"][:, 2 * width:] = 0
    args.output_dir.mkdir(parents=True)
    metrics, summary = {}, {}
    with torch.inference_mode():
        original_bounded = encoder.hypernet(context).tanh()
        saved = np.concatenate((trace["gamma"][chunks, environments, -1, 0],
                                trace["beta"][chunks, environments, -1, 0],
                                trace["gamma"][chunks, environments, -1, 1],
                                trace["beta"][chunks, environments, -1, 1]), axis=-1)
        replay_error = float((original_bounded - torch.from_numpy(saved)).abs().max())
        if replay_error > 2e-4:
            raise ValueError(f"CPU HyperNet replay disagrees with saved modulation: {replay_error}")
        original_modulated = None
        original_encoded = None
        for label, changed_context in variants.items():
            bounded = encoder.hypernet(changed_context).tanh()
            gamma_vision, beta_vision, gamma_tactile, beta_tactile = bounded.chunk(4, dim=-1)
            gamma = torch.stack((gamma_vision, gamma_tactile), dim=1)
            beta = torch.stack((beta_vision, beta_tactile), dim=1)
            modulated = features + encoder.alpha * (gamma * features + beta)
            encoded = reconstruct_condition(encoder, features, gamma, beta, state)
            if original_modulated is None:
                original_modulated, original_encoded = modulated, encoded
            residual_change = (modulated - original_modulated).norm(dim=-1) / features.norm(dim=-1).clamp_min(1e-8)
            condition_change = (encoded - original_encoded).norm(dim=-1) / original_encoded.norm(dim=-1).clamp_min(1e-8)
            values = torch.cat((residual_change, condition_change[:, None]), dim=-1).numpy()
            dense = np.zeros((*paired.shape, 3), dtype=values.dtype)
            dense[chunks, environments] = values
            metrics[label] = values
            summary[label] = {phase: environment_level_summary(dense, phases[phase] & paired)
                              for phase in ("pre_contact", "post_contact_pre_success")}
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    positions = np.arange(len(variants))
    for metric_index, title in enumerate(("Vision feature delta / raw norm", "Tactile feature delta / raw norm",
                                          "Fused condition delta / original norm")):
        for phase_index, phase in enumerate(("pre_contact", "post_contact_pre_success")):
            values = [summary[label][phase]["mean"][metric_index] for label in variants]
            axes[metric_index].bar(positions + (phase_index - 0.5) * 0.36, values, width=0.36, label=phase)
        axes[metric_index].set_xticks(positions, list(variants), rotation=40, ha="right", fontsize=8)
        axes[metric_index].set_title(title)
    axes[-1].legend(fontsize=8)
    figure.suptitle("Only HyperNet context changes; actual features/state remain fixed. Not action or success effects.")
    figure.savefig(args.output_dir / "context_dependence.png", dpi=160)
    plt.close(figure)
    np.savez_compressed(args.output_dir / "context_dependence.npz", **metrics, chunks=chunks,
                        environments=environments, donors=donors[chunks, environments])
    metadata.update({"run": str(args.run), "paired_histories": len(chunks), "cpu_replay_max_abs_error": replay_error,
        "donor_protocol": "Other environment, same observation time and contact phase, before benchmark success",
        "metric_order": ["vision_relative_feature_delta", "tactile_relative_feature_delta", "relative_fused_condition_delta"],
        "scope": "Offline context dependence, not task utility; mean_test_context uses the analyzed rollout only for description.",
        "summary": summary})
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"paired_histories": len(chunks), "cpu_replay_max_abs_error": replay_error,
        "phase_means": {label: {phase: entry["mean"] for phase, entry in item.items()} for label, item in summary.items()}}, indent=2))


if __name__ == "__main__":
    main()
