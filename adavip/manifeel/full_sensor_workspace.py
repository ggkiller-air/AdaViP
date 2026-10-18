"""Independent, resumable Stage-0 training with paired EMA validation."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import dill
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler

from adavip.manifeel.full_sensor_schema import CONDITION_NAMES
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def training_protocol(cfg: DictConfig, runtime_independent: bool = True) -> dict:
    """Keep scientific settings strict while allowing lossless data transport changes."""
    protocol = OmegaConf.to_container(cfg, resolve=True)
    for key in ("output_dir", "resume"):
        protocol.pop(key, None)
    if runtime_independent:
        runtime = protocol.get("runtime", {})
        for key in ("preflight_windows", "min_samples_per_second", "max_data_wait_fraction",
                    "stop_after_steps", "stop_after_epochs", "allow_epoch_extension"):
            runtime.pop(key, None)
        if not runtime:
            protocol.pop("runtime", None)
        for key in ("checkpoint_every", "checkpoint_every_steps", "checkpoint_first_step", "checkpoint_epochs"):
            protocol["training"].pop(key, None)
        for key in ("decoded_cache_dir", "decoded_cache_mode"):
            protocol["dataset"].pop(key, None)
        for section in ("dataloader", "val_dataloader"):
            for key in ("num_workers", "timeout", "persistent_workers", "prefetch_factor",
                        "pin_memory", "multiprocessing_context"):
                protocol[section].pop(key, None)
    return protocol


def manifest_fingerprint(manifest: dict) -> str:
    """Hash canonical JSON metadata, not the multi-terabyte image payload."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()


def training_protocol_legacy(cfg: DictConfig) -> dict:
    """Recognize the previous cache-independent checkpoint fingerprint exactly."""
    protocol = training_protocol(cfg, runtime_independent=False)
    runtime = protocol.get("runtime", {})
    for key in ("preflight_windows", "min_samples_per_second", "max_data_wait_fraction"):
        runtime.pop(key, None)
    if not runtime:
        protocol.pop("runtime", None)
    for key in ("decoded_cache_dir", "decoded_cache_mode"):
        protocol["dataset"].pop(key, None)
    for section in ("dataloader", "val_dataloader"):
        for key in ("num_workers", "timeout", "persistent_workers", "prefetch_factor",
                    "pin_memory", "multiprocessing_context"):
            protocol[section].pop(key, None)
    return protocol


def normalizer_fingerprint(normalizer: torch.nn.Module) -> str:
    """Hash every frozen normalization tensor in a device-independent order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(normalizer.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class OffsetSampler(Sampler[int]):
    """Replay a deterministic epoch's sample order without reading consumed data."""

    def __init__(self, sampler: Sampler[int], offset: int) -> None:
        self.sampler = sampler
        self.offset = offset

    def __iter__(self) -> Iterator[int]:
        return itertools.islice(iter(self.sampler), self.offset, None)

    def __len__(self) -> int:
        return max(0, len(self.sampler) - self.offset)


class FullSensorWorkspace(BaseWorkspace):
    """Keep raw/EMA weights, optimizer, provenance and next-batch RNG state.

    Validation always uses the same selected EMA (or raw) policy for all ten
    conditions. Resume rejects protocol changes and replays the sampler cursor.
    """

    include_keys = (
        "epoch", "global_step", "best_validation_loss", "data_provenance", "rng_state", "ema_step",
        "batch_in_epoch", "epoch_metrics_state",
        "lr_schedule_extension",
    )

    def __init__(self, cfg: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(cfg, output_dir)
        seed = int(cfg.training.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        if self.ema_model is not None:
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.epoch = 0
        self.global_step = 0
        self.ema_step = 0
        self.best_validation_loss = float("inf")
        self.data_provenance: dict | None = None
        self.rng_state: dict | None = None
        self.batch_in_epoch = 0
        self.epoch_metrics_state: dict = {}
        self._stop_requested: str | None = None
        self._throughput_windows: list[dict] = []
        self.throughput_preflight_failed = False
        self.lr_schedule_extension: dict | None = None

    def _make_dataset(self) -> Any:
        return hydra.utils.instantiate(self.cfg.dataset)

    def _device(self) -> torch.device:
        return torch.device(self.cfg.training.device)

    def _initialize_runtime(self, device: torch.device) -> None:
        pass

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        return values

    def _should_stop(self) -> bool:
        stop_file = os.environ.get("ADAVIP_STOP_FILE")
        if stop_file and Path(stop_file).exists() and not self._stop_requested:
            self.request_stop(Path(stop_file).read_text().strip() or "stop_file")
        return bool(self._stop_requested)

    def _training_losses(self, batch: dict, condition_ids: torch.Tensor) -> torch.Tensor:
        return self.model.compute_loss_per_sample(batch, condition_ids)

    def _backward(self, losses: torch.Tensor, batch: dict) -> None:
        losses.mean().backward()

    def _accumulate(self, state: dict, batch: dict, losses: torch.Tensor,
                    condition_ids: torch.Tensor) -> tuple[int, float]:
        detached = losses.detach().double().cpu()
        valid = batch.get("_distributed_valid", torch.ones_like(detached, dtype=torch.bool)).cpu().bool()
        detached = detached[valid]
        conditions, tasks = condition_ids.cpu()[valid], batch["task_index"].cpu()[valid]
        contributions = []
        for indices, size in ((conditions, len(CONDITION_NAMES)), (tasks, len(state["task_counts"]))):
            sums, counts = torch.zeros(size, dtype=torch.float64), torch.zeros(size, dtype=torch.float64)
            sums.index_add_(0, indices, detached)
            counts.index_add_(0, indices, torch.ones_like(detached))
            contributions.extend((sums, counts))
        reduced = self._reduce(torch.cat(contributions))
        offset = 0
        for key in ("condition_sums", "condition_counts", "task_sums", "task_counts"):
            size = len(state[key])
            state[key] += reduced[offset:offset + size]
            offset += size
        count = int(reduced[len(CONDITION_NAMES):2 * len(CONDITION_NAMES)].sum())
        return count, float(reduced[:len(CONDITION_NAMES)].sum() / count)

    def _validation_losses(self, policy: Any, batch: dict, generator: torch.Generator,
                           dataset: Any, batch_index: int) -> torch.Tensor:
        return policy.paired_denoising_losses(batch, generator=generator)["losses"]

    def _lr_multiplier(self, total_steps: int) -> float:
        if self.lr_schedule_extension is not None:
            anchor = self.lr_schedule_extension
            progress = (self.global_step - anchor["step"]) / max(1, total_steps - 1 - anchor["step"])
            return anchor["multiplier"] * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        warmup = int(self.cfg.training.lr_warmup_steps)
        if self.global_step < warmup:
            return (self.global_step + 1) / max(1, warmup)
        progress = (self.global_step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def request_stop(self, reason: str = "requested") -> None:
        """Request a checkpoint and clean exit at the next safe training boundary."""
        self._stop_requested = reason

    def _log_progress(self, record: dict) -> None:
        record.setdefault("timestamp", time.time())
        with (Path(self.output_dir) / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    def _save_interruption(self) -> None:
        self._capture_rng()
        self._save("latest")
        self._save("interrupt")
        self._log_progress({
            "event": "stopped", "reason": self._stop_requested, "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch, "global_step": self.global_step,
        })

    def _check_throughput_preflight(self, record: dict) -> None:
        options = self.cfg.get("runtime", {})
        window_count = int(options.get("preflight_windows", 0))
        if window_count < 2 or len(self._throughput_windows) >= window_count:
            return
        self._throughput_windows.append(record)
        if len(self._throughput_windows) != window_count:
            return
        measured = self._throughput_windows[1:]
        seconds = sum(1 / item["samples_per_second"] for item in measured)
        throughput = len(measured) / seconds
        waiting = sum(item["data_wait_fraction"] / item["samples_per_second"] for item in measured) / seconds
        passed = (throughput >= float(options.get("min_samples_per_second", 0))
                  and waiting <= float(options.get("max_data_wait_fraction", 1)))
        self._log_progress({"event": "throughput_preflight", "passed": passed,
                            "global_step": self.global_step, "samples_per_second": throughput,
                            "data_wait_fraction": waiting, "measured_windows": len(measured)})
        if not passed:
            self.throughput_preflight_failed = True
            self.request_stop("throughput_preflight_failed")

    @classmethod
    def create_from_checkpoint(
        cls, path: str, exclude_keys: tuple | None = None,
        include_keys: tuple | None = None, map_location: str = "cpu", **kwargs: Any,
    ) -> "FullSensorWorkspace":
        """Load a trusted workspace checkpoint without assuming its original GPU."""
        payload = torch.load(path, map_location=map_location, pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys, **kwargs)
        return instance

    def _capture_rng(self) -> None:
        self.rng_state = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def _restore_rng(self) -> None:
        if self.rng_state is None:
            raise ValueError("Resume checkpoint has no RNG state")
        if "ranks" in self.rng_state:
            raise ValueError("A DDP RNG checkpoint requires torchrun with its original world size")
        random.setstate(self.rng_state["python"])
        np.random.set_state(self.rng_state["numpy"])
        torch.set_rng_state(self.rng_state["torch"].cpu())
        if self.rng_state["cuda"]:
            torch.cuda.set_rng_state_all([value.cpu() for value in self.rng_state["cuda"]])

    def _save(self, tag: str) -> str:
        checkpoint = Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        temporary = checkpoint.with_suffix(".partial")
        self.save_checkpoint(path=temporary, use_thread=False)
        temporary.replace(checkpoint)
        return str(checkpoint)

    def _loader(self, dataset: Any, training: bool, epoch: int = 0, start_batch: int = 0) -> DataLoader:
        options = OmegaConf.to_container(self.cfg.dataloader if training else self.cfg.val_dataloader, resolve=True)
        if options.get("shuffle", False):
            raise ValueError("Use the task-balanced sampler, not shuffle")
        if options.get("drop_last", False):
            raise ValueError("Dropping samples would change validation/task weighting")
        if options["num_workers"] == 0:
            options.pop("prefetch_factor", None)
            options["persistent_workers"] = False
            options["timeout"] = 0
        generator = torch.Generator().manual_seed(int(self.cfg.training.seed) + epoch)
        sampler = dataset.make_task_balanced_sampler(generator) if training else None
        if training and start_batch:
            sampler = OffsetSampler(sampler, start_batch * options["batch_size"])
        return DataLoader(dataset, sampler=sampler, generator=generator, **options)

    @torch.no_grad()
    def validate(self, dataset: Any, device: torch.device) -> dict[str, float]:
        """Aggregate per-sample paired losses by task, then report micro and macro."""
        policy = self.ema_model if self.ema_model is not None else self.model
        policy.eval()
        sums = torch.zeros(len(dataset.task_specs), len(CONDITION_NAMES), dtype=torch.float64)
        counts = torch.zeros(len(dataset.task_specs), dtype=torch.float64)
        generator = torch.Generator(device=device).manual_seed(int(self.cfg.training.validation_seed))
        for batch_index, batch in enumerate(self._loader(dataset, False)):
            if self.cfg.training.max_val_steps is not None and batch_index >= self.cfg.training.max_val_steps:
                break
            batch = dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))
            losses = self._validation_losses(policy, batch, generator, dataset, batch_index).double().cpu()
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite paired validation loss")
            valid = batch.get("_distributed_valid", torch.ones(len(losses), dtype=torch.bool)).cpu().bool()
            losses = losses[valid]
            task_indices = batch["task_index"].cpu()[valid]
            sums.index_add_(0, task_indices, losses)
            counts.index_add_(0, task_indices, torch.ones(len(task_indices), dtype=torch.float64))
        sums = self._reduce(sums)
        counts = self._reduce(counts)
        if self._should_stop():
            raise InterruptedError("Validation interrupted before epoch commit")
        if not counts.sum():
            raise ValueError("Validation requires at least one sample")
        observed = counts > 0
        means = sums / counts.clamp_min(1).unsqueeze(1)
        result = {"validation/task_coverage": float(observed.sum()), "validation/sample_count": float(counts.sum())}
        for condition_index, condition in enumerate(CONDITION_NAMES):
            result[f"validation/micro/{condition}"] = float(sums[:, condition_index].sum() / counts.sum())
            result[f"validation/macro/{condition}"] = float(means[observed, condition_index].mean())
            for task_index, spec in enumerate(dataset.task_specs):
                if observed[task_index]:
                    result[f"validation/{spec.task_id}/{condition}"] = float(means[task_index, condition_index])
        return result

    def run(self) -> None:
        """Train bounded or full epochs, validate and write reproducible checkpoints."""
        cfg = self.cfg
        if cfg.training.num_epochs < 1 or cfg.training.gradient_clip_norm <= 0 or cfg.training.checkpoint_every < 1:
            raise ValueError("Epoch budget, checkpoint interval and gradient clip norm must be positive")
        if cfg.training.lr_warmup_steps < 0:
            raise ValueError("LR warmup cannot be negative")
        for field in ("log_every_steps", "checkpoint_every_steps", "checkpoint_first_step"):
            if cfg.training.get(field, 1) < 1:
                raise ValueError(f"{field} must be positive")
        for field in ("max_train_steps", "max_val_steps"):
            if cfg.training[field] is not None and cfg.training[field] < 1:
                raise ValueError(f"{field} must be positive or null")
        device = self._device()
        output = Path(self.output_dir)
        if not cfg.resume and (output / "checkpoints/latest.ckpt").exists():
            raise FileExistsError("Refusing to overwrite an existing run; select resume or a new output_dir")
        output.mkdir(parents=True, exist_ok=True)
        dataset = self._make_dataset()
        validation = dataset.get_validation_dataset()
        normalizer = dataset.get_normalizer()
        protocol = training_protocol(cfg)
        expected = {
            "manifest": manifest_fingerprint(dataset.manifest),
            "normalizer": normalizer_fingerprint(normalizer),
            "protocol": manifest_fingerprint(protocol),
        }
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        if cfg.resume:
            payload = self.load_checkpoint(
                path=cfg.resume, map_location="cpu", include_keys=self.include_keys,
            )
            saved = self.data_provenance
            saved_protocol = training_protocol(payload["cfg"])
            valid_saved_hashes = {
                manifest_fingerprint(saved_protocol),
                manifest_fingerprint(training_protocol(payload["cfg"], runtime_independent=False)),
            }
            valid_saved_hashes.add(manifest_fingerprint(training_protocol_legacy(payload["cfg"])))
            comparable = copy.deepcopy(saved_protocol)
            extending = (bool(cfg.get("runtime", {}).get("allow_epoch_extension", False))
                         and cfg.training.num_epochs > payload["cfg"].training.num_epochs)
            if extending:
                comparable["training"]["num_epochs"] = cfg.training.num_epochs
            if (saved is None or saved.get("manifest") != expected["manifest"]
                    or saved.get("normalizer") != expected["normalizer"]
                    or saved.get("protocol") not in valid_saved_hashes
                    or comparable != protocol):
                raise ValueError("Resume rejected: dataset/split/normalizer/training protocol changed")
            if extending:
                self.lr_schedule_extension = {
                    "step": self.global_step,
                    "multiplier": self.optimizer.param_groups[0]["lr"] / cfg.optimizer.lr,
                    "previous_epochs": int(payload["cfg"].training.num_epochs),
                    "epochs": int(cfg.training.num_epochs),
                }
            self.data_provenance = expected
            for policy in (self.model, self.ema_model):
                if policy is not None and normalizer_fingerprint(policy.normalizer) != expected["normalizer"]:
                    raise ValueError("Checkpoint normalizer does not match train-only statistics")
        elif self.epoch or self.global_step:
            raise ValueError("A populated workspace requires an explicit resume checkpoint")
        else:
            self.data_provenance = expected
        self._write_metadata(dataset, expected, output)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        self._initialize_runtime(device)
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if self.ema_model is not None else None
        if ema is not None:
            ema.optimization_step = self.ema_step
        if cfg.resume:
            self._restore_rng()
        print(json.dumps({
            "event": "workspace_ready", "output_dir": str(output), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "batch_size": cfg.dataloader.batch_size, "num_workers": cfg.dataloader.num_workers,
            "train_samples": len(dataset), "validation_samples": len(validation),
            "next_epoch": self.epoch + 1, "global_step": self.global_step,
            "batch_in_epoch": self.batch_in_epoch,
            "validation_weights": "ema" if ema is not None else "raw",
            "condition_names": CONDITION_NAMES,
            "condition_probabilities": self.model.condition_probabilities.cpu().tolist(),
            "provenance": self.data_provenance,
        }), flush=True)
        epoch_steps = math.ceil(len(dataset) / cfg.dataloader.batch_size)
        if cfg.training.max_train_steps is not None:
            epoch_steps = min(epoch_steps, cfg.training.max_train_steps)
        total_steps = epoch_steps * cfg.training.num_epochs
        start_step, start_epoch = self.global_step, self.epoch
        if not 0 <= self.batch_in_epoch <= epoch_steps:
            raise ValueError("Checkpoint sampler cursor is outside the epoch")
        for epoch_index in range(self.epoch, cfg.training.num_epochs):
            epoch_started = time.perf_counter()
            initial_cursor = self.batch_in_epoch
            if self._should_stop():
                self._save_interruption()
                return
            self._log_progress({"event": "epoch_start", "epoch": epoch_index + 1,
                                "batch_in_epoch": self.batch_in_epoch, "global_step": self.global_step})
            self.model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if not self.epoch_metrics_state:
                self.epoch_metrics_state = {
                    "condition_sums": torch.zeros(len(CONDITION_NAMES), dtype=torch.float64),
                    "condition_counts": torch.zeros(len(CONDITION_NAMES), dtype=torch.float64),
                    "task_sums": torch.zeros(len(dataset.task_specs), dtype=torch.float64),
                    "task_counts": torch.zeros(len(dataset.task_specs), dtype=torch.float64),
                    "max_grad_norm": 0.0, "train_seconds": 0.0,
                }
            state = self.epoch_metrics_state
            condition_sums, condition_counts = state["condition_sums"], state["condition_counts"]
            task_sums, task_counts = state["task_sums"], state["task_counts"]
            window_samples = 0
            window_seconds = 0.0
            window_data_seconds = 0.0
            batch_requested = time.perf_counter()
            loader = self._loader(dataset, True, epoch_index, self.batch_in_epoch)
            for batch_index, batch in enumerate(loader, start=self.batch_in_epoch):
                if batch_index >= epoch_steps:
                    break
                batch_ready = time.perf_counter()
                data_seconds = batch_ready - batch_requested
                batch = dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))
                multiplier = self._lr_multiplier(total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = cfg.optimizer.lr * multiplier
                self.optimizer.zero_grad(set_to_none=True)
                condition_ids = self.model.sample_condition_ids(batch["action"].shape[0])
                losses = self._training_losses(batch, condition_ids)
                if not torch.isfinite(losses).all():
                    raise FloatingPointError("Nonfinite training loss")
                self._backward(losses, batch)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.training.gradient_clip_norm, error_if_nonfinite=True,
                )
                state["max_grad_norm"] = max(state["max_grad_norm"], float(grad_norm))
                self.optimizer.step()
                if ema is not None:
                    ema.step(self.model)
                    self.ema_step = ema.optimization_step
                self.global_step += 1
                sample_count, mean_loss = self._accumulate(state, batch, losses, condition_ids)
                self.batch_in_epoch = batch_index + 1
                step_seconds = time.perf_counter() - batch_requested
                state["train_seconds"] += step_seconds
                window_seconds += step_seconds
                window_data_seconds += data_seconds
                window_samples += sample_count
                if self.global_step % cfg.training.get("log_every_steps", 50) == 0:
                    record = {
                        "event": "train_step", "epoch": epoch_index + 1,
                        "batch_in_epoch": self.batch_in_epoch, "epoch_steps": epoch_steps,
                        "global_step": self.global_step, "loss": mean_loss,
                        "lr": self.optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                        "samples_per_second": window_samples / window_seconds,
                        "data_wait_fraction": window_data_seconds / window_seconds,
                        "condition_counts": dict(zip(CONDITION_NAMES, condition_counts.long().tolist())),
                    }
                    if device.type == "cuda":
                        record["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                    self._log_progress(record)
                    self._check_throughput_preflight(record)
                    window_samples, window_seconds, window_data_seconds = 0, 0.0, 0.0
                stop_steps = int(cfg.get("runtime", {}).get("stop_after_steps", 0))
                if stop_steps and self.global_step - start_step >= stop_steps:
                    self.request_stop("bounded_step_preflight")
                if self._should_stop():
                    self._save_interruption()
                    return
                if (self.global_step == cfg.training.get("checkpoint_first_step", 100)
                        or self.global_step % cfg.training.get("checkpoint_every_steps", 1000) == 0):
                    self._capture_rng()
                    self._save("latest")
                    self._log_progress({
                        "event": "checkpoint", "global_step": self.global_step,
                        "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                    })
                batch_requested = time.perf_counter()
            train_seconds = state["train_seconds"]
            self._log_progress({"event": "validation_start", "epoch": epoch_index + 1, "global_step": self.global_step})
            try:
                metrics = self.validate(validation, device)
            except InterruptedError:
                self._save_interruption()
                return
            metrics.update({
                "epoch": epoch_index + 1, "global_step": self.global_step,
                "train/loss": float(condition_sums.sum() / condition_counts.sum()),
                "train/max_grad_norm_before_clip": state["max_grad_norm"],
                "train/lr": self.optimizer.param_groups[0]["lr"],
                "train/seconds": train_seconds,
                "train/samples_per_second": float(condition_counts.sum()) / train_seconds,
            })
            if device.type == "cuda":
                metrics["runtime/peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            for index, name in enumerate(CONDITION_NAMES):
                metrics[f"train/count/{name}"] = int(condition_counts[index])
                if condition_counts[index]:
                    metrics[f"train/loss/{name}"] = float(condition_sums[index] / condition_counts[index])
            for index, spec in enumerate(dataset.task_specs):
                if task_counts[index]:
                    metrics[f"train/task/{spec.task_id}"] = float(task_sums[index] / task_counts[index])
            self.epoch = epoch_index + 1
            self.batch_in_epoch = 0
            self.epoch_metrics_state = {}
            score = metrics["validation/macro/full"]
            improved = score < self.best_validation_loss
            self.best_validation_loss = min(score, self.best_validation_loss)
            self._capture_rng()
            self._save("latest")
            if improved:
                self._save("best")
            if (self.epoch % cfg.training.checkpoint_every == 0
                    or self.epoch in cfg.training.get("checkpoint_epochs", [])):
                self._save(f"epoch{self.epoch}")
            metrics["runtime/epoch_wall_seconds"] = time.perf_counter() - epoch_started
            metrics["runtime/resumed_batch_cursor"] = initial_cursor
            self._write_metrics(metrics, output)
            stop_epochs = int(cfg.get("runtime", {}).get("stop_after_epochs", 0))
            if stop_epochs and self.epoch - start_epoch >= stop_epochs:
                self._log_progress({"event": "bounded_epoch_preflight", "epoch": self.epoch,
                                    "global_step": self.global_step})
                return

    def _write_metrics(self, metrics: dict, output: Path) -> None:
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)

    def _write_metadata(self, dataset: Any, expected: dict, output: Path) -> None:
        (output / "dataset_manifest.json").write_text(json.dumps(dataset.manifest, indent=2) + "\n")
        (output / "provenance.json").write_text(json.dumps(expected, indent=2) + "\n")
        OmegaConf.save(self.cfg, output / "config.yaml", resolve=True)
