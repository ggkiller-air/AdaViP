#!/usr/bin/env python3
"""Replay attention logits to distinguish softmax saturation from phase adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adavip.manifeel.mechanism_probe import fixed_fusion_output, fusion_attention_logits
from scripts.manifeel.probe_hypernet_demos import load_encoder


def main() -> None:
    """Measure original softmax scores and descriptive temperature counterfactuals on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(2)
    encoder, _, metadata = load_encoder(args.checkpoint)
    with np.load(args.trace, allow_pickle=False) as archive:
        features, saved = archive["modulated_features"], archive["attention"]
        if args.rollout is not None:
            steps = archive["chunk_start_sim_step"]
    if args.rollout is not None:
        with np.load(args.rollout, allow_pickle=False) as archive:
            rewards = archive["reward_by_step"]
        success = np.concatenate((np.zeros((1, rewards.shape[1]), dtype=bool), rewards > 0.5))
        valid = ~np.maximum.accumulate(success, axis=0)[steps]
        features, saved = features[:, :, -1][valid], saved[:, :, -1][valid]
    else:
        if features.ndim != 3:
            raise ValueError("rollout traces require --rollout for post-success exclusion")
    with torch.inference_mode():
        logits = fusion_attention_logits(encoder.cross_attention, torch.from_numpy(features[:, 0]),
                                        torch.from_numpy(features[:, 1]))
        probabilities = logits.softmax(dim=-1)
        error = float((probabilities - torch.from_numpy(saved)).abs().max())
        if error > 2e-4:
            raise ValueError(f"logit replay disagrees with recorded attention: {error}")
        margin = (logits[..., 1] - logits[..., 0]).numpy()
        temperatures = np.array([1, 2, 4, 8, 16, 32, 64])
        sweep = torch.stack([(logits / temperature).softmax(dim=-1) for temperature in temperatures])
        entropy = -(sweep.clamp_min(1e-30) * sweep.clamp_min(1e-30).log2()).sum(dim=-1)
        original_entropy = entropy[0].mean(dim=0).numpy()
        vision, tactile = torch.from_numpy(features[:, 0]), torch.from_numpy(features[:, 1])
        tokens = torch.stack((vision, tactile), dim=1)
        dynamic_attended, _ = encoder.cross_attention(vision[:, None], tokens, tokens, need_weights=False)
        dynamic_attended = dynamic_attended[:, 0]
        original_vision = vision + encoder.fusion_gate.tanh() * dynamic_attended
        static_approximations = {}
        for label, weights in (("sample_mean_weights_offline", probabilities[..., 1].mean(dim=0)),
                               ("hard_majority_head_weights", (probabilities[..., 1].mean(dim=0) > 0.5).float())):
            attended = fixed_fusion_output(encoder.cross_attention, vision, tactile, weights)
            relative = (encoder.fusion_gate.tanh() * (attended - dynamic_attended)).norm(dim=-1) / original_vision.norm(dim=-1).clamp_min(1e-8)
            static_approximations[label] = {"tactile_weights": weights.tolist(),
                "mean_relative_fused_vision_error": float(relative.mean()),
                "max_relative_fused_vision_error": float(relative.max())}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for head in range(margin.shape[1]):
        axes[0].hist(margin[:, head], bins=35, histtype="step", label=f"head {head}")
        axes[1].plot(temperatures, entropy[:, :, head].mean(dim=1).numpy(), marker="o", label=f"head {head}")
    axes[0].set(xlabel="Actual tactile minus vision logit", ylabel="Recorded inputs")
    axes[0].axvline(0, color="gray", linestyle="--")
    axes[1].set(xlabel="Counterfactual softmax temperature (original=1)", ylabel="Mean attention entropy (bits)",
                xscale="log", ylim=(-0.02, 1.02))
    axes[1].legend()
    figure.suptitle("Attention-score saturation; temperature sweep is not an action or task-utility test")
    args.output_dir.mkdir(parents=True)
    figure.savefig(args.output_dir / "attention_saturation.png", dpi=170)
    plt.close(figure)
    metadata.update({"trace": str(args.trace), "inputs": len(features), "cpu_replay_max_abs_error": error,
        "mean_tactile_attention_by_head": probabilities[..., 1].mean(dim=0).tolist(),
        "saturated_fraction_max_probability_ge_099_by_head": (probabilities.max(dim=-1).values >= 0.99).float().mean(dim=0).tolist(),
        "original_mean_entropy_bits_by_head": original_entropy.tolist(),
        "signed_tactile_minus_vision_logit_quantiles": np.quantile(margin, [0.05, 0.5, 0.95], axis=0).tolist(),
        "temperatures": temperatures.tolist(), "temperature_entropy_by_head": entropy.mean(dim=1).tolist(),
        "fixed_attention_approximations": static_approximations,
        "scope": "Descriptive saved attention-score replay; input weighting is not utility; no policy inference or training."})
    np.savez_compressed(args.output_dir / "attention_logits.npz", logits=logits.numpy(),
                        probabilities=probabilities.numpy(), temperatures=temperatures)
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in ("cpu_replay_max_abs_error", "mean_tactile_attention_by_head",
        "original_mean_entropy_bits_by_head", "signed_tactile_minus_vision_logit_quantiles")}, indent=2))


if __name__ == "__main__":
    main()
