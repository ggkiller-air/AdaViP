"""Offline K64 paired-mask diagnostic for a trained Stage 2 MLP checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import dill
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adavip.manifeel.stage2_data import load_complete_ram_cache, make_stage2_dataset
from scripts.manifeel.train_stage2_mlp_t2 import build_policy, load_teacher


def per_dimension_correlation(prediction: np.ndarray, target: np.ndarray) -> list[float | None]:
    """Return Pearson coefficients, or null for a constant column."""
    values = []
    for index in range(prediction.shape[1]):
        left = prediction[:, index]
        right = target[:, index]
        if left.std() < 1e-12 or right.std() < 1e-12:
            values.append(None)
        else:
            values.append(float(np.corrcoef(left, right)[0, 1]))
    return values


def parse_args() -> argparse.Namespace:
    """Parse a checkpoint plus its immutable Stage 2 data contract."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-config", required=True)
    parser.add_argument("--dataset-root", default="/data/wangzihao/datasets/manifeel")
    parser.add_argument("--task-embeddings", default="/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-task", type=int, default=16)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Compare new-policy mask sensitivity with fixed Teacher Utility GT."""
    args = parse_args()
    if args.k < 1 or args.samples_per_task < 2 or args.batch_size < 1:
        raise ValueError("K, sample count and batch size must be positive")
    if torch.cuda.device_count() != 1 or "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This diagnostic requires one H200")
    saved = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill)
    contract = saved["contract"]
    if contract["variant"] != "stage2_mlp_fusion":
        raise ValueError("Checkpoint does not belong to the MLP control")
    if str(Path(args.teacher).resolve()) != contract["teacher"] or str(Path(args.cache).resolve()) != contract["cache"]:
        raise ValueError("Checkpoint Teacher/cache contract differs")
    dataset = make_stage2_dataset(args.labels, args.dataset_root, args.task_embeddings)
    if dataset.store.manifest["sha256"] != contract["labels_sha256"]:
        raise ValueError("Utility labels differ from the training run")
    load_complete_ram_cache(dataset.source, args.cache, args.cache_config)
    teacher = load_teacher(args.teacher, args.labels)
    policy = build_policy(teacher, np.asarray(contract["rms"], dtype=np.float64))
    policy.load_state_dict(saved["ema"], strict=True)
    policy.cuda().eval()
    del teacher, saved

    generator = np.random.default_rng(42)
    selected = []
    for task in range(9):
        indices = np.flatnonzero(dataset.task_indices == task)
        selected.extend(generator.choice(indices, min(args.samples_per_task, len(indices)), replace=False).tolist())
    loader = DataLoader(Subset(dataset, selected), batch_size=args.batch_size,
                        num_workers=6, pin_memory=True, persistent_workers=True,
                        multiprocessing_context="fork", prefetch_factor=2)
    predicted_utility = []
    teacher_utility = []
    policy_mask_utility = []
    torch_generator = torch.Generator(device="cuda").manual_seed(42)
    for batch in loader:
        batch = {key: ({sensor: value.cuda(non_blocking=True) for sensor, value in item.items()}
                       if key == "obs" else item.cuda(non_blocking=True))
                 for key, item in batch.items()}
        tokens = policy.obs_encoder.encode_tokens(policy.normalizer.normalize(batch["obs"]))
        conditions = []
        for presence in policy.condition_presence:
            encoded = policy.condition_from_tokens(tokens, presence.expand(len(tokens), -1))
            conditions.append(encoded["conditioning"])
            if len(conditions) == 1:
                predicted_utility.append(encoded["u_hat"].cpu().numpy())
        condition_matrix = torch.stack(conditions, dim=1).flatten(0, 1)
        total_losses = torch.zeros(len(tokens), 10, dtype=torch.float64, device="cuda")
        for _ in range(args.k):
            sample = policy.prepare_denoising_sample(batch, generator=torch_generator)
            repeated_sample = type(sample)(
                noisy_action=sample.noisy_action.repeat_interleave(10, dim=0),
                noise=sample.noise.repeat_interleave(10, dim=0),
                timesteps=sample.timesteps.repeat_interleave(10, dim=0),
                target=sample.target.repeat_interleave(10, dim=0),
                loss_mask=sample.loss_mask.repeat_interleave(10, dim=0),
            )
            losses = policy.denoising_loss_per_sample(repeated_sample, condition_matrix)
            total_losses += losses.reshape(len(tokens), 10).double()
        averaged = total_losses / args.k
        policy_mask_utility.append((averaged[:, 1:] - averaged[:, :1]).cpu().numpy())
        teacher_utility.append(batch["utility"].cpu().numpy())
    prediction = np.concatenate(predicted_utility).astype(np.float64)
    teacher_gt = np.concatenate(teacher_utility).astype(np.float64)
    model_sensitivity = np.concatenate(policy_mask_utility)
    normalized_gt = teacher_gt / np.asarray(contract["rms"], dtype=np.float64)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_windows": len(selected), "k": args.k, "sample_indices": selected,
        "utility_prediction_mae_9d": np.abs(prediction - normalized_gt).mean(axis=0).tolist(),
        "utility_prediction_correlation_9d": per_dimension_correlation(prediction, normalized_gt),
        "mask_sensitivity_correlation_9d": per_dimension_correlation(model_sensitivity, teacher_gt),
        "mask_sensitivity_sign_agreement_9d":
            (np.sign(model_sensitivity) == np.sign(teacher_gt)).mean(axis=0).tolist(),
        "mask_sensitivity_mae_9d": np.abs(model_sensitivity - teacher_gt).mean(axis=0).tolist(),
        "caveat": "Paired denoising mask sensitivity is offline OOD evidence, not simulator success rate.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "utility_diagnostic_complete", "output": str(output),
                      "num_windows": len(selected), "k": args.k}), flush=True)


if __name__ == "__main__":
    main()
