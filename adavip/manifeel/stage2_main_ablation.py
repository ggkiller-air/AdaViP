"""Main B--F ablations on the unchanged L1 S0 optimization protocol."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from adavip.manifeel.stage2_ablation_workspace import Stage2AblationTrainer, tensor_fingerprint


FULL_INITIALIZATION_SHA256 = "19c82812a5ac14d118e14337dc2bb8cda7b5071d0a1609c0cf8c2f1d6bf98803"
PERMUTATION_SEED = 20261109


def audit_pruned_initialization(full_state: dict, policy) -> dict:
    """Compare every surviving tensor, allowing only B's documented input-column slice."""
    counts = {}
    for name, value in policy.state_dict().items():
        if name.startswith("w_out.") and policy.ablation == "C_teacher_residual":
            assert not bool(value.count_nonzero())
            continue
        expected = full_state[name]
        if policy.ablation == "B" and name.startswith("film_hypernets.") and name.endswith("network.0.weight"):
            expected = expected[:, :64]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        role = name.split(".")[0]
        counts[role] = counts.get(role, 0) + value.numel()
    return {"ablation": policy.ablation, "surviving_tensors_exact": True,
            "surviving_elements_by_role": counts,
            "ablation_initialization_sha256": tensor_fingerprint(policy.state_dict())}


def save_shuffle_manifest(dataset, output: Path, seed: int) -> dict:
    """Persist the exact bijection and validate task identity, marginals and continuation."""
    rows = dataset.shuffle_utility(seed)
    original = np.asarray(dataset.store.utility)
    assert np.array_equal(np.sort(rows), np.arange(len(dataset)))
    assert np.array_equal(dataset.task_indices, dataset.task_indices[rows])
    report = {"seed": seed, "scope": "all existing Stage2 training windows, within task",
              "validation_shuffled": False, "validation_windows": 0,
              "sha256": hashlib.sha256(rows.tobytes()).hexdigest(), "tasks": {}}
    for task in np.unique(dataset.task_indices):
        selected = np.flatnonzero(dataset.task_indices == task)
        assert np.array_equal(np.sort(selected), np.sort(rows[selected]))
        # A bijection of whole rows preserves every multivariate statistic exactly.
        assert np.array_equal(original[np.sort(rows[selected])], original[selected])
        report["tasks"][int(task)] = {"windows": len(selected),
                                      "fixed_points": int(np.sum(rows[selected] == selected))}
    path = output / "utility_permutation.npy"
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), rows):
            raise ValueError("Saved Utility permutation differs on resume")
    else:
        np.save(path, rows, allow_pickle=False)
    (output / "shuffle.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def set_protocol_step(trainer, step: int) -> None:
    """Position LambdaLR without any optimizer/EMA update or random draw."""
    trainer.global_step = step
    scheduler = trainer.scheduler
    scheduler.last_epoch = step
    scheduler._step_count = step + 1
    rates = [base * fn(step) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    for group, rate in zip(trainer.optimizer.param_groups, rates):
        group["lr"] = rate
    scheduler._last_lr = rates


class MainAblationTrainer(Stage2AblationTrainer):
    """Reuse the production loop, with explicit skipped-A clock and clipping telemetry."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped_a_steps = self.steps_per_epoch if self.policy.utility_predictor is None else 0
        if self.skipped_a_steps:
            self.epoch = 2
            set_protocol_step(self, self.skipped_a_steps)

    def _new_stats(self):
        stats = super()._new_stats()
        stats.update(clip_updates=0, clip_triggered=0, clip_coefficient_sum=0.0,
                     utility_gradient_norm_sum=0.0, action_gradient_norm_sum=0.0)
        return stats

    def observe_before_clip(self):
        module = self.policy.utility_predictor
        grads = [] if module is None else [p.grad for p in module.parameters() if p.grad is not None]
        self.utility_norm = (torch.stack([g.detach().square().sum() for g in grads]).sum().sqrt()
                             if grads else torch.zeros((), device=self.device))

    def observe_clip(self, norm):
        total, utility = float(norm), float(self.utility_norm)
        coefficient = min(1.0, 1.0 / (total + 1e-6))
        self.stats["clip_updates"] += 1
        self.stats["clip_triggered"] += int(coefficient < 1)
        self.stats["clip_coefficient_sum"] += coefficient
        self.stats["utility_gradient_norm_sum"] += utility
        self.stats["action_gradient_norm_sum"] += max(0.0, total * total - utility * utility) ** 0.5

    def _metrics(self, stage):
        result = super()._metrics(stage)
        count = max(1, self.stats["clip_updates"])
        result.update(ablation=self.policy.ablation, protocol_step=self.global_step,
                      optimizer_updates=self.global_step - self.skipped_a_steps,
                      skipped_a_steps=self.skipped_a_steps,
                      clip_trigger_fraction=self.stats["clip_triggered"] / count,
                      mean_clip_coefficient=self.stats["clip_coefficient_sum"] / count,
                      mean_utility_gradient_norm=self.stats["utility_gradient_norm_sum"] / count,
                      mean_action_gradient_norm=self.stats["action_gradient_norm_sum"] / count)
        if self.policy.utility_predictor is None:
            result["utility_loss"] = None
            result["per_task_utility_loss"] = {key: None for key in result["per_task_utility_loss"]}
        return result

    def save(self):
        super().save()
        latest = self.output / "checkpoints/latest.ckpt"
        actual = self.global_step - self.skipped_a_steps
        if actual == 1 and self.device.type == "cuda":
            import shutil
            archive = Path("/data/wangzihao/archives/manifeel_uvip") / self.output.parent.name / self.output.name
            archive.mkdir(parents=True, exist_ok=True)
            target = archive / "first_update.ckpt"
            if not target.exists():
                temporary = archive / "first_update.ckpt.partial"
                shutil.copyfile(latest, temporary)
                os.replace(temporary, target)
        if self.epoch > self.epochs:
            target = latest.parent / f"epoch{self.epochs}.ckpt"
            if not target.exists():
                os.link(latest, target)

    def run(self, resume=None):
        if self.skipped_a_steps and resume is None:
            self.output.mkdir(parents=True, exist_ok=True)
            if (self.output / "checkpoints/latest.ckpt").exists():
                raise FileExistsError("Use resume for an existing ablation")
            self.save()
            latest = self.output / "checkpoints/latest.ckpt"
            os.link(latest, latest.parent / "epoch1.ckpt")
            (self.output / "stage_a_skipped.json").write_text(json.dumps({
                "epoch1_checkpoint": "initial weights; zero optimizer updates; next epoch=2",
                "protocol_step": self.global_step, "optimizer_updates": 0}) + "\n")
            # The shared loop requires explicit resume when latest already exists.
            resume = latest
        super().run(resume)
