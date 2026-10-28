"""Shared S1--S4 training, deferred RAM loading and exact optimizer continuation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Iterator

import cv2
import dill
import torch
from torch.utils.data import DataLoader, Sampler

from adavip.manifeel.stage2_policy import Stage2Policy
from scripts.manifeel.train_stage2_t2 import (
    atomic_checkpoint, make_optimizer_scheduler, restore_policy_state, restore_rng,
    rng_state, set_stage_trainability, stage_for_epoch, update_ema,
)


SNAPSHOT_EPOCHS = frozenset((1, 6, 10, 20, 50, 100, 200, 300, 400, 500))


def tensor_fingerprint(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes and values without modifying random streams."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def immutable_fingerprint(policy: Stage2Policy) -> str | None:
    """Fingerprint the exact weights that must remain at their E500 values."""
    if policy.is_dual_encoder:
        return tensor_fingerprint(policy.obs_encoder.state_dict())
    if policy.encoder_strategy == "s1":
        return tensor_fingerprint(policy.obs_encoder.image_encoders.state_dict())
    return None


def parameter_report(policy: Stage2Policy, gradients: bool = False) -> dict:
    """Report role/sensor parameter counts and current backward participation."""
    modules = {"context_encoder": policy.obs_encoder,
               "utility_predictor": policy.utility_predictor, "unet": policy.model}
    encoder = policy.action_encoder if policy.is_dual_encoder else policy.obs_encoder
    modules.update({f"action_resnet/{key}": module for key, module in encoder.image_encoders.items()})
    modules.update(action_temporal=encoder.temporal_projections,
                   action_proprio=encoder.state_projection, action_language=encoder.language_projection)
    result = {}
    for name, module in modules.items():
        parameters = list(module.parameters())
        record = {"total": sum(p.numel() for p in parameters),
                  "trainable": sum(p.numel() for p in parameters if p.requires_grad)}
        if gradients:
            active = [p.grad for p in parameters if p.grad is not None]
            record["with_gradient"] = sum(g.numel() for g in active)
            record["gradient_l2"] = (float(torch.stack([g.detach().square().sum() for g in active]).sum().sqrt())
                                     if active else None)
        result[name] = record
    identity = encoder.sensor_identity
    result["action_sensor_identity"] = {"total": identity.numel(),
                                        "trainable": identity.numel() if identity.requires_grad else 0,
                                        "has_gradient": identity.grad is not None}
    result["total"] = sum(p.numel() for p in policy.parameters())
    result["trainable"] = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return result


def validate_gradient_routes(policy: Stage2Policy, batch: dict) -> dict:
    """Check B/C backward on the first raw batch, preserving weights and all RNG."""
    saved_rng = rng_state()
    report = {}
    try:
        for stage in ("B", "C"):
            set_stage_trainability(policy, stage)
            policy.train()
            policy.zero_grad(set_to_none=True)
            losses = policy.training_losses(batch, stage)
            if not torch.isfinite(losses["action_loss"] + losses["utility_loss"]):
                raise FloatingPointError("Nonfinite first-batch route check")
            losses["utility_loss"].backward(retain_graph=True)
            if not any(p.grad is not None for p in policy.utility_predictor.parameters()):
                raise RuntimeError("Utility head has no Utility gradient")
            if any(p.grad is not None for name, p in policy.named_parameters()
                   if not name.startswith("utility_predictor.")):
                raise RuntimeError("Utility gradient escaped its head")
            policy.zero_grad(set_to_none=True)
            losses["action_loss"].backward()
            if any(p.grad is not None for p in policy.utility_predictor.parameters()):
                raise RuntimeError("Action gradient reached Utility head")
            if policy.is_dual_encoder and any(p.grad is not None for p in policy.obs_encoder.parameters()):
                raise RuntimeError("Frozen context branch received gradients")
            encoder = policy.action_encoder if policy.is_dual_encoder else policy.obs_encoder
            expected = policy.is_dual_encoder or (stage == "C" and policy.encoder_strategy != "s1")
            for key, module in encoder.image_encoders.items():
                active = any(p.grad is not None and bool(p.grad.count_nonzero()) for p in module.parameters())
                if active != expected:
                    raise RuntimeError(f"Incorrect {stage} ResNet gradient route: {key}")
            if policy.is_dual_encoder or stage == "C":
                for module in (encoder.temporal_projections, encoder.state_projection,
                               encoder.language_projection):
                    if not any(p.grad is not None and bool(p.grad.count_nonzero()) for p in module.parameters()):
                        raise RuntimeError("Action tokenizer has no gradient")
                if encoder.sensor_identity.grad is None or not bool(encoder.sensor_identity.grad.count_nonzero()):
                    raise RuntimeError("Action sensor identity has no gradient")
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                          1.0, error_if_nonfinite=True)
            report[stage] = parameter_report(policy, gradients=True)
    finally:
        policy.zero_grad(set_to_none=True)
        restore_rng(saved_rng)
    return report


class Stage2EpochSampler(Sampler[int]):
    """Replay S0's one-based epoch seed 42+epoch with a resumable batch cursor."""

    def __init__(self, dataset: object, batch_size: int) -> None:
        self.dataset, self.batch_size = dataset, batch_size
        self.epoch, self.cursor = 1, 0

    def __iter__(self) -> Iterator[int]:
        sampler = self.dataset.make_task_balanced_sampler(torch.Generator().manual_seed(42 + self.epoch))
        return iter(list(sampler)[self.cursor * self.batch_size:])

    def __len__(self) -> int:
        return max(0, len(self.dataset) - self.cursor * self.batch_size)


def initialize_ablation_worker(worker_id: int) -> None:
    """Bind each CPU worker to an allocated core and disable nested thread pools."""
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    cores = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cores[worker_id % len(cores)]})


class Stage2AblationTrainer:
    """Use the original losses/optimizer with one shared, resumable S1--S4 loop."""

    def __init__(self, policy: Stage2Policy, dataset: object, output: Path,
                 contract: dict, cfg: dict, cache_loader: Callable[[], int],
                 device: str = "cuda:0", batch_size: int = 32, epochs: int = 600,
                 workers: int = 8, stop_after_steps: int = 0) -> None:
        self.policy, self.dataset, self.output = policy, dataset, Path(output)
        self.device = torch.device(device)
        self.batch_size, self.epochs, self.workers = batch_size, epochs, workers
        self.stop_after_steps, self.stop_requested = stop_after_steps, False
        self.cache_loader, self.cache_ready = cache_loader, False
        self.contract, self.cfg = contract, cfg
        self.contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.initialization = {
            "common_sha256": tensor_fingerprint({k: v for k, v in policy.state_dict().items()
                                                  if not k.startswith("action_encoder.")}),
            "immutable_sha256": immutable_fingerprint(policy),
            "action_encoder_sha256": (tensor_fingerprint(policy.action_encoder.state_dict())
                                       if policy.is_dual_encoder else None),
        }
        policy.to(self.device)
        self.ema = copy.deepcopy(policy).eval().requires_grad_(False)
        self.steps_per_epoch = math.ceil(len(dataset) / batch_size)
        self.optimizer, self.scheduler = make_optimizer_scheduler(policy, self.steps_per_epoch, epochs)
        self.epoch, self.batch_cursor, self.global_step = 1, 0, 0
        self.stats: dict = {}
        self.cumulative_counts = [0] * len(dataset.task_specs)
        self.sampler = Stage2EpochSampler(dataset, batch_size)
        self.loader: DataLoader | None = None
        self.transport_pause = 0.0
        self.routes_checked = False

    def log(self, event: dict, filename: str = "metrics.jsonl") -> None:
        """Append a finite JSON record to this run's isolated persistent log."""
        event = {"encoder_strategy": self.policy.encoder_strategy, **event}
        with (self.output / filename).open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event, allow_nan=False), flush=True)

    def should_stop(self) -> bool:
        """Observe Python signals and the Slurm batch shell's stop file."""
        name = os.environ.get("ADAVIP_STOP_FILE")
        return self.stop_requested or bool(name and Path(name).exists())

    def verify_immutable(self) -> None:
        """Reject any raw/EMA change to the frozen E500 path before checkpointing."""
        expected = self.initialization["immutable_sha256"]
        if expected is not None:
            for model in (self.policy, self.ema):
                if immutable_fingerprint(model) != expected:
                    raise RuntimeError("Frozen E500 weights changed")

    def save(self) -> None:
        """Atomically save architecture, raw/EMA, optimizer, metrics, cursor and RNG."""
        self.verify_immutable()
        payload = {"model": self.policy.state_dict(), "ema": self.ema.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                   "epoch": self.epoch, "batch_cursor": self.batch_cursor, "global_step": self.global_step,
                   "rng": rng_state(), "contract": self.contract, "contract_hash": self.contract_hash,
                   "cfg": self.cfg, "initialization": self.initialization, "epoch_metrics": self.stats,
                   "cumulative_task_counts": self.cumulative_counts,
                   "stage": stage_for_epoch(min(self.epoch, self.epochs))}
        atomic_checkpoint(self.output / "checkpoints/latest.ckpt", payload)

    def restore(self, path: Path) -> None:
        """Strictly restore the complete training state and immutable branch contract."""
        saved = torch.load(path, map_location="cpu", pickle_module=dill)
        if saved["contract_hash"] != self.contract_hash:
            raise ValueError("Resume contract differs from encoder ablation")
        if saved["initialization"] != self.initialization:
            raise ValueError("Resume initialization differs")
        restore_policy_state(self.policy, saved["model"], self.device)
        restore_policy_state(self.ema, saved["ema"], self.device)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.scheduler.load_state_dict(saved["scheduler"])
        self.epoch, self.batch_cursor, self.global_step = saved["epoch"], saved["batch_cursor"], saved["global_step"]
        self.stats, self.cumulative_counts = saved["epoch_metrics"], saved["cumulative_task_counts"]
        self.verify_immutable()
        restore_rng(saved["rng"])

    def _batches(self) -> Iterator[dict]:
        self.sampler.epoch, self.sampler.cursor = self.epoch, self.batch_cursor
        if not self.cache_ready:
            if len(self.sampler):
                cold = DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler,
                                  num_workers=0, pin_memory=self.device.type == "cuda",
                                  generator=torch.Generator().manual_seed(700042))
                yield next(iter(cold))
                del cold
            paused = time.perf_counter()
            self.save()
            self.log({"event": "formal_update_saved_before_ram", "global_step": self.global_step})
            if self.should_stop():
                return
            started = time.perf_counter()
            size = self.cache_loader()
            self.cache_ready = True
            self.log({"event": "all_data_ram_ready", "bytes": size,
                      "seconds": time.perf_counter() - started}, "runtime.jsonl")
            self.transport_pause = time.perf_counter() - paused
            self.sampler.cursor = self.batch_cursor
        if self.loader is None:
            options = {"batch_size": self.batch_size, "sampler": self.sampler,
                       "num_workers": self.workers, "pin_memory": self.device.type == "cuda",
                       "generator": torch.Generator().manual_seed(700042),
                       "worker_init_fn": initialize_ablation_worker}
            if self.workers:
                options.update(persistent_workers=True, prefetch_factor=2, multiprocessing_context="fork")
            self.loader = DataLoader(self.dataset, **options)
        yield from self.loader

    def _new_stats(self) -> dict:
        tasks = len(self.dataset.task_specs)
        return {"counts": torch.zeros(tasks, dtype=torch.int64),
                "action_sum": torch.zeros(tasks, dtype=torch.float64),
                "utility_sum": torch.zeros(tasks, dtype=torch.float64),
                "train_seconds": 0.0, "data_seconds": 0.0}

    def _metrics(self, stage: str) -> dict:
        counts = self.stats["counts"]
        total = int(counts.sum())
        task_ids = [spec.task_id for spec in self.dataset.task_specs]
        action = self.stats["action_sum"] / counts.clamp_min(1)
        utility = self.stats["utility_sum"] / counts.clamp_min(1)
        return {"epoch": self.epoch, "stage": stage, "global_step": self.global_step,
                "batch_cursor": self.batch_cursor,
                "action_loss": float(self.stats["action_sum"].sum() / max(1, total)) if stage != "A" else None,
                "utility_loss": float(self.stats["utility_sum"].sum() / max(1, total)),
                "per_task_action_loss": {key: float(action[i]) if counts[i] and stage != "A" else None
                                         for i, key in enumerate(task_ids)},
                "per_task_utility_loss": {key: float(utility[i]) if counts[i] else None
                                          for i, key in enumerate(task_ids)},
                "task_counts": counts.tolist(), "cumulative_task_counts": self.cumulative_counts,
                "samples_per_second": total / max(1e-9, self.stats["train_seconds"]),
                "data_wait_seconds": self.stats["data_seconds"],
                "lr": {group["name"]: group["lr"] for group in self.optimizer.param_groups}}

    def run(self, resume: Path | None = None) -> None:
        """Train the first raw batch, preload all RAM, then continue the same draws."""
        self.output.mkdir(parents=True, exist_ok=True)
        latest = self.output / "checkpoints/latest.ckpt"
        if not resume and latest.exists():
            raise FileExistsError("Use resume or a new ablation output directory")
        if resume:
            self.restore(resume)
        (self.output / "contract.json").write_text(json.dumps(self.contract, indent=2) + "\n")
        (self.output / "initialization.json").write_text(json.dumps(self.initialization, indent=2) + "\n")
        (self.output / "architecture.json").write_text(json.dumps(self.cfg, indent=2) + "\n")
        starting_step = self.global_step
        while self.epoch <= self.epochs:
            stage = stage_for_epoch(self.epoch)
            set_stage_trainability(self.policy, stage)
            self.policy.train()
            self.log({"event": "phase_parameters", "epoch": self.epoch, "stage": stage,
                      "parameters": parameter_report(self.policy)}, "gradients.jsonl")
            if not self.stats:
                self.stats = self._new_stats()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            requested = time.perf_counter()
            for batch in self._batches():
                requested += self.transport_pause
                self.transport_pause = 0.0
                waited = time.perf_counter() - requested
                batch = {key: ({sensor: value.to(self.device, non_blocking=True) for sensor, value in item.items()}
                               if key == "obs" else item.to(self.device, non_blocking=True))
                         for key, item in batch.items()}
                if not self.routes_checked:
                    started = time.perf_counter()
                    routes = validate_gradient_routes(self.policy, batch)
                    self.log({"event": "first_batch_routes_verified", "routes": routes}, "gradients.jsonl")
                    self.verify_immutable()
                    self.routes_checked = True
                    set_stage_trainability(self.policy, stage)
                    self.policy.train()
                    requested += time.perf_counter() - started
                    if self.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                result = self.policy.training_losses(batch, stage)
                loss = result["utility_loss"] if stage == "A" else result["action_loss"] + 0.05 * result["utility_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite Stage 2 loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad],
                                                     1.0, error_if_nonfinite=True)
                if self.batch_cursor == 0 or (self.global_step + 1) % 1000 == 0:
                    self.log({"event": "backward", "epoch": self.epoch, "stage": stage,
                              "global_step": self.global_step + 1,
                              "parameters": parameter_report(self.policy, gradients=True)}, "gradients.jsonl")
                self.optimizer.step()
                self.scheduler.step()
                update_ema(self.ema, self.policy, 0.9999)
                self.global_step += 1
                self.batch_cursor += 1
                tasks = batch["task_index"].detach().cpu()
                counts = torch.bincount(tasks, minlength=len(self.cumulative_counts))
                self.stats["counts"] += counts
                self.stats["action_sum"].index_add_(0, tasks, result["action_loss_per_sample"].detach().double().cpu())
                self.stats["utility_sum"].index_add_(0, tasks, result["utility_loss_per_sample"].detach().double().cpu())
                self.cumulative_counts = [a + b for a, b in zip(self.cumulative_counts, counts.tolist())]
                self.stats["data_seconds"] += waited
                self.stats["train_seconds"] += time.perf_counter() - requested
                if self.global_step == 1 or self.global_step % 20 == 0:
                    record = {"event": "train", **self._metrics(stage), "gradient_norm": float(norm)}
                    if self.device.type == "cuda":
                        record.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
                    self.log(record)
                if self.stop_after_steps and self.global_step - starting_step >= self.stop_after_steps:
                    self.stop_requested = True
                if self.should_stop():
                    self.save()
                    (self.output / "SAFE_STOP.json").write_text(json.dumps({"global_step": self.global_step}) + "\n")
                    return
                requested = time.perf_counter()
            if self.should_stop():
                self.save()
                (self.output / "SAFE_STOP.json").write_text(json.dumps({"global_step": self.global_step}) + "\n")
                return
            record = {"event": "epoch_complete", **self._metrics(stage)}
            if self.device.type == "cuda":
                record.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                              gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            self.log(record)
            completed = self.epoch
            self.epoch += 1
            self.batch_cursor, self.stats = 0, {}
            self.save()
            if completed in SNAPSHOT_EPOCHS:
                os.link(latest, self.output / "checkpoints" / f"epoch{completed}.ckpt")
        (self.output / "COMPLETE.json").write_text(json.dumps({"epoch": self.epochs,
            "global_step": self.global_step, "contract_hash": self.contract_hash}) + "\n")
