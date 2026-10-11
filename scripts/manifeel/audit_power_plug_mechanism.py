#!/usr/bin/env python3
"""Audit existing checkpoints and evaluation records without training or GPU inference."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import dill
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch

from adavip.manifeel.mechanism_probe import modulation_variance, parameter_distance


def checkpoint_audit(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    """Read saved protocol and keep only encoder and normalizer tensors on CPU."""
    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    source = "ema_model" if cfg.training.use_ema else "model"
    saved = payload["state_dicts"][source]
    tensors = {key: value.clone() for key, value in saved.items()
               if key.startswith(("obs_encoder.", "normalizer."))}
    paths = ("training.seed", "training.use_ema", "dataloader.batch_size", "val_dataloader.batch_size",
             "task.dataset.zarr_path", "task.dataset.max_train_episodes", "task.dataset.val_ratio",
             "policy.obs_encoder._target_", "policy.obs_encoder.use_fusion", "policy.obs_encoder.use_fusion_hypernet",
             "policy.obs_encoder.vision_key", "policy.shape_meta", "policy.noise_scheduler",
             "policy.n_obs_steps", "policy.n_action_steps", "policy.horizon", "policy.num_inference_steps")
    protocol = {}
    for key in paths:
        value = OmegaConf.select(cfg, key, default=None)
        protocol[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    learned = {}
    for key in ("residual_alpha", "fusion_residual_alpha", "fusion_gate"):
        tensor = tensors.get(f"obs_encoder.{key}")
        if tensor is not None:
            raw = float(tensor)
            maximum = float(OmegaConf.select(cfg, "policy.obs_encoder.alpha_max", default=0.1))
            learned[key] = {"raw": raw, "effective": float(tensor.tanh()) if key == "fusion_gate"
                            else maximum * float(tensor.sigmoid())}
    report = {"checkpoint": str(path.resolve()), "weights": source, "protocol": protocol, "learned_scalars": learned}
    del payload, saved
    gc.collect()
    return report, tensors


def main() -> None:
    """Audit bounded checkpoint selections and reuse existing fifty-environment results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, default=Path("results/hypernet_batch8_20261010"))
    parser.add_argument("--evaluation-root", type=Path, default=Path(
        "/data/wangzihao/outputs/manifeel/eval/batch8_power_plug_5methods_n50_steps500_seed100000_20260926/formal"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    series = {"baseline": "baseline", "static_fusion": "dp_power_plug_resnet_fusion_batch8_ep400_seed42",
              "hypernet_no_fusion": "dp_power_plug_hyper_resnet_no_fusion_batch8_ep400_seed42",
              "hypernet_fusion": "dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42"}
    scores, provenance = {}, {}
    for label, directory in series.items():
        scores[label], provenance[label] = [], []
        for epoch in (100, 200, 300):
            path = args.evaluation_root / directory / f"epoch{epoch}" / "eval_log.json"
            data = json.loads(path.read_text())
            score = float(data["test/mean_score"])
            rewards = [value for key, value in data.items() if key.startswith("test/sim_max_reward_")]
            if len(rewards) != 50 or not np.isclose(np.mean(rewards), score):
                raise ValueError(f"not a verified fifty-environment benchmark record: {path}")
            scores[label].append(score)
            provenance[label].append(str(path))
    figure, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    for label, values in scores.items():
        axis.plot([100, 200, 300], 100 * np.asarray(values), marker="o", label=label)
    axis.set(xlabel="Saved training epoch", ylabel="test/mean_score (%)", ylim=(0, 100), xticks=[100, 200, 300],
             title="Existing results: one reused evaluation seed, not independent validation")
    axis.legend()
    figure.savefig(args.output_dir / "checkpoint_selection_comparison.png", dpi=170)
    plt.close(figure)
    selection = {"same_epoch_main_minus_baseline_pp": (100 * (np.asarray(scores["hypernet_fusion"]) - scores["baseline"])).tolist(),
                 "best_of_three_main_minus_baseline_pp": 100 * (max(scores["hypernet_fusion"]) - max(scores["baseline"])),
                 "caveat": "Best-of-three on this shared evaluation seed is selection-biased, not an unbiased test estimate."}
    report = {"scores": scores, "source_logs": provenance, "selection": selection, "checkpoints": {}, "distances": {}}
    selections = {"baseline100": Path("checkpoints/batch8_tactile/baseline/latest_epoch100.ckpt"),
                  "baseline200": Path("checkpoints/batch8_tactile/baseline/latest_epoch200.ckpt"),
                  "main100": Path("checkpoints/0926") / series["hypernet_fusion"] / "checkpoints/latest_epoch100.ckpt",
                  "static_fusion100": Path("checkpoints/0926") / series["static_fusion"] / "checkpoints/latest_epoch100.ckpt",
                  "hypernet_no_fusion300": Path("checkpoints/0926") / series["hypernet_no_fusion"] / "checkpoints/latest_epoch300.ckpt"}
    baseline = None
    for label, path in selections.items():
        print(f"CPU checkpoint audit: {label}", flush=True)
        report["checkpoints"][label], tensors = checkpoint_audit(path)
        if baseline is None:
            baseline = tensors
        else:
            report["distances"][label] = {group: parameter_distance(baseline, tensors, prefix) for group, prefix in (
                ("resnet_parameters", "obs_encoder.key_model_map."), ("normalizer", "normalizer."))}
        del tensors
        gc.collect()
    run = args.prior_root / "gpu4_rollouts/hyperresnet_epoch100_n50_steps500_seed100000"
    with np.load(run / "hypernet_rollout_trace.npz", allow_pickle=False) as archive:
        gamma, beta = archive["gamma"][:, :, -1], archive["beta"][:, :, -1]
        steps = archive["chunk_start_sim_step"]
    with np.load(run / "rollout_trace.npz", allow_pickle=False) as archive:
        successful = np.concatenate([np.zeros((1, 50), dtype=bool), archive["reward_by_step"] > 0.5])
    valid = ~np.maximum.accumulate(successful, axis=0)[steps]
    report["main_rollout_modulation"] = {f"{kind}_{modality}": modulation_variance(values[:, :, index], valid)
        for kind, values in (("gamma", gamma), ("beta", beta)) for index, modality in enumerate(("vision", "tactile"))}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"selection": selection, "main_rollout_modulation": report["main_rollout_modulation"]}, indent=2))


if __name__ == "__main__":
    main()
