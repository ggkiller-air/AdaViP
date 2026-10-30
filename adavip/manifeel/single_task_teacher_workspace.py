"""Resumable single-task teacher training using official DP optimization settings."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from adavip.manifeel.full_sensor_workspace import OffsetSampler, normalizer_fingerprint
from adavip.manifeel.single_task_teacher import initialize_teacher_worker
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def teacher_protocol(cfg: Any) -> dict:
    """Keep resume strict about scientific settings, allowing runtime/path changes."""
    values = OmegaConf.to_container(cfg, resolve=True)
    for key in ("output_dir", "resume", "runtime"):
        values.pop(key, None)
    return values


class SingleTaskTeacherWorkspace(BaseWorkspace):
    """Save raw/EMA/Adam/scheduler/RNG and resume the next shuffled batch exactly."""

    include_keys = ("epoch", "global_step", "batch_in_epoch", "epoch_state", "rng_state", "ema_step",
                    "data_provenance", "best_full_loss")

    def __init__(self, cfg: Any, output_dir: str | None = None) -> None:
        super().__init__(cfg, output_dir or cfg.output_dir)
        random.seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        torch.manual_seed(cfg.training.seed)
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.epoch = self.global_step = self.batch_in_epoch = self.ema_step = 0
        self.epoch_state: dict = {}
        self.rng_state: dict | None = None
        self.data_provenance: dict | None = None
        self.best_full_loss = float("inf")
        self.stop_reason: str | None = None
        self.scheduler = None
        self.condition_generator = None

    def request_stop(self, reason: str) -> None:
        """Checkpoint at the next optimizer boundary on scheduler/user interruption."""
        self.stop_reason = reason

    def load_payload(self, payload: dict, exclude_keys=None, include_keys=None, **kwargs: Any) -> None:
        """Permit inference-only restoration without constructing a training scheduler."""
        excluded = tuple(exclude_keys or ())
        if self.scheduler is None:
            excluded += ("scheduler",)
        super().load_payload(payload, exclude_keys=excluded, include_keys=include_keys, **kwargs)

    def log(self, event: str, **values: Any) -> None:
        """Persist machine-readable progress and keep scheduler stdout useful."""
        record = {"event": event, "timestamp": time.time(), "epoch": self.epoch,
                  "global_step": self.global_step, **values}
        with (Path(self.output_dir) / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def _capture_rng(self) -> None:
        self.rng_state = {"torch": torch.get_rng_state(), "python": random.getstate(),
                          "numpy": np.random.get_state(), "condition": self.condition_generator.get_state(),
                          "cuda": torch.cuda.get_rng_state_all() if self.model.device.type == "cuda" else []}

    def _restore_rng(self) -> None:
        torch.set_rng_state(self.rng_state["torch"].cpu())
        random.setstate(self.rng_state["python"])
        np.random.set_state(self.rng_state["numpy"])
        self.condition_generator.set_state(self.rng_state["condition"].cpu())
        if self.rng_state["cuda"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in self.rng_state["cuda"]])

    def save(self, tag: str = "latest") -> Path:
        """Atomically publish a complete workspace checkpoint."""
        self._capture_rng()
        path = Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        self.save_checkpoint(path=temporary, use_thread=False)
        os.replace(temporary, path)
        return path

    def _loader(self, dataset: Any, training: bool, start_batch: int = 0) -> DataLoader:
        options = OmegaConf.to_container(self.cfg.dataloader if training else self.cfg.val_dataloader)
        order_generator = torch.Generator().manual_seed(self.cfg.training.seed + self.epoch)
        indices = torch.randperm(len(dataset), generator=order_generator).tolist() if training else list(range(len(dataset)))
        # The loader generator does not consume action-noise randomness.
        loader_generator = torch.Generator().manual_seed(self.cfg.training.seed + 10000 + self.epoch)
        batch_size = options.pop("batch_size")
        workers = options["num_workers"]
        if workers == 0:
            options.pop("prefetch_factor", None)
            options["persistent_workers"] = False
            options["timeout"] = 0
        return DataLoader(dataset, batch_size=batch_size, sampler=OffsetSampler(indices, start_batch * batch_size),
                          generator=loader_generator, worker_init_fn=initialize_teacher_worker, **options)

    @torch.no_grad()
    def validate(self, dataset: Any, device: torch.device) -> dict:
        """Measure paired full/removal losses with fixed validation noise each time."""
        generator = torch.Generator(device=device).manual_seed(self.cfg.training.seed + 1042)
        totals = torch.zeros(len(self.model.condition_names), dtype=torch.float64)
        count = 0
        for batch in self._loader(dataset, False):
            batch = dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))
            values = self.ema_model.paired_denoising_losses(batch, generator)["losses"]
            totals += values.double().sum(0).cpu()
            count += len(values)
            if self._stop_requested():
                raise InterruptedError
        return {name: float(value / count) for name, value in zip(self.model.condition_names, totals)}

    def _stop_requested(self) -> bool:
        path = os.environ.get("ADAVIP_STOP_FILE")
        if path and Path(path).exists():
            self.request_stop("scheduler_stop_file")
        return bool(self.stop_reason)

    def run(self, dataset: Any | None = None, stop_after_steps: int = 0) -> None:
        """Train with batch8, shuffled official windows, AdamW/cosine and EMA."""
        cfg = self.cfg
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        device = torch.device(cfg.training.device)
        dataset = hydra.utils.instantiate(cfg.task.dataset) if dataset is None else dataset
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)
        manifest_hash = hashlib.sha256(json.dumps(dataset.manifest, sort_keys=True).encode()).hexdigest()
        expected = {"dataset_sha256": manifest_hash, "normalizer_sha256": normalizer_fingerprint(normalizer)}
        epoch_steps = math.ceil(len(dataset) / cfg.dataloader.batch_size)
        self.scheduler = get_scheduler("cosine", optimizer=self.optimizer, num_warmup_steps=500,
                                       num_training_steps=epoch_steps * cfg.training.num_epochs)
        self.condition_generator = torch.Generator(device=device).manual_seed(cfg.training.seed + 40000)
        if cfg.resume:
            payload = torch.load(cfg.resume, map_location="cpu", pickle_module=dill)
            if teacher_protocol(payload["cfg"]) != teacher_protocol(cfg):
                raise ValueError("Resume scientific protocol differs")
            self.load_payload(payload, exclude_keys=("condition_generator",),
                              include_keys=self.include_keys)
            if self.data_provenance != expected:
                raise ValueError("Resume dataset or normalizer differs")
            if normalizer_fingerprint(self.model.normalizer) != expected["normalizer_sha256"]:
                raise ValueError("Saved normalizer differs from the training split")
            del payload
        else:
            self.data_provenance = expected
        self.model.to(device).train()
        self.ema_model.to(device).eval()
        optimizer_to(self.optimizer, device)
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
        ema.optimization_step = self.ema_step
        validation = dataset.get_validation_dataset()
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
        (output / "dataset_manifest.json").write_text(json.dumps(dataset.manifest, indent=2) + "\n")
        if cfg.resume:
            self._restore_rng()
        initial_hash = hashlib.sha256()
        for name, value in self.model.state_dict().items():
            if name == "condition_probabilities":
                continue
            initial_hash.update(name.encode())
            initial_hash.update(value.detach().cpu().numpy().tobytes())
        self.log("startup", train_windows=len(dataset), validation_windows=len(validation),
                 train_episodes=len(dataset.manifest["train_episodes"]), val_episodes=len(dataset.manifest["val_episodes"]),
                 epoch_steps=epoch_steps, batch_size=cfg.dataloader.batch_size, target_epochs=cfg.training.num_epochs,
                 trainer_cpus=sorted(os.sched_getaffinity(0)), num_workers=cfg.dataloader.num_workers,
                 allocated_cpus=os.environ.get("SLURM_CPUS_PER_TASK"),
                 parameter_count=sum(param.numel() for param in self.model.parameters()),
                 initial_state_sha256=initial_hash.hexdigest(), provenance=expected,
                 dropout_probabilities=self.model.condition_probabilities.cpu().tolist(),
                 resumed_batch=self.batch_in_epoch, precision="fp32", gradient_clipping=False)
        starting_step = self.global_step
        while self.epoch < cfg.training.num_epochs:
            if not self.epoch_state:
                self.epoch_state = {"loss_sum": 0., "samples": 0, "seconds": 0.,
                                    "condition_counts": [0] * len(self.model.condition_names)}
            window_samples = 0
            window_seconds = window_wait = 0.
            requested = time.perf_counter()
            for batch_index, batch in enumerate(self._loader(dataset, True, self.batch_in_epoch), self.batch_in_epoch):
                waiting = time.perf_counter() - requested
                batch = dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))
                self.optimizer.zero_grad(set_to_none=True)
                ids = self.model.sample_condition_ids(len(batch["action"]), self.condition_generator)
                losses = self.model.losses(batch, ids)
                loss = losses.mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite teacher loss")
                loss.backward()
                if not all(torch.isfinite(param.grad).all() for param in self.model.parameters() if param.grad is not None):
                    raise FloatingPointError("Nonfinite teacher gradient")
                if self.global_step == 0:
                    encoder = self.model.obs_encoder
                    self.log("first_gradient_check", state_tokenizer_grad_norm=float(
                        encoder.state_projection[0].weight.grad.norm()), image_grad_norms={key: float(
                            next(encoder.image_encoders[key].parameters()).grad.norm()) for key in encoder.image_keys})
                self.optimizer.step()
                self.scheduler.step()
                ema.step(self.model)
                self.ema_step = ema.optimization_step
                self.global_step += 1
                self.batch_in_epoch = batch_index + 1
                elapsed = time.perf_counter() - requested
                count = len(batch["action"])
                state = self.epoch_state
                state["loss_sum"] += float(loss.detach()) * count
                state["samples"] += count
                state["seconds"] += elapsed
                state["condition_counts"] = (np.asarray(state["condition_counts"]) +
                    np.bincount(ids.cpu().numpy(), minlength=len(self.model.condition_names))).tolist()
                window_samples += count
                window_seconds += elapsed
                window_wait += waiting
                if self.global_step % cfg.runtime.log_every_steps == 0 or self.global_step == 1:
                    self.log("train_step", training_epoch=self.epoch + 1, batch_in_epoch=self.batch_in_epoch,
                             loss=float(loss.detach()), lr=self.optimizer.param_groups[0]["lr"],
                             samples_per_second=window_samples / window_seconds,
                             data_wait_fraction=window_wait / window_seconds,
                             condition_counts=dict(zip(self.model.condition_names, state["condition_counts"])),
                             peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0.)
                    window_samples = 0
                    window_seconds = window_wait = 0.
                if self._stop_requested() or (stop_after_steps and self.global_step - starting_step >= stop_after_steps):
                    self.save()
                    self.log("stopped", reason=self.stop_reason or "bounded_preflight")
                    if self.stop_reason:
                        (output / "SAFE_STOP.json").write_text(json.dumps({"reason": self.stop_reason}) + "\n")
                    return
                if self.global_step == 1 or self.global_step % cfg.runtime.checkpoint_every_steps == 0:
                    path = self.save()
                    self.log("checkpoint", path=str(path), bytes=path.stat().st_size)
                requested = time.perf_counter()
            self.epoch += 1
            self.batch_in_epoch = 0
            state = self.epoch_state
            full_loss = None
            if self.epoch % cfg.runtime.validate_every == 0 or self.epoch == cfg.training.num_epochs:
                try:
                    validation_losses = self.validate(validation, device)
                except InterruptedError:
                    self.epoch -= 1
                    self.batch_in_epoch = epoch_steps
                    self.save()
                    (output / "SAFE_STOP.json").write_text('{"reason":"validation_stop"}\n')
                    return
                full_loss = validation_losses["full"]
                self.log("validation", losses=validation_losses)
            self.log("epoch_complete", train_loss=state["loss_sum"] / state["samples"],
                     samples=state["samples"], samples_per_second=state["samples"] / state["seconds"],
                     condition_counts=dict(zip(self.model.condition_names, state["condition_counts"])))
            self.epoch_state = {}
            if full_loss is not None and full_loss < self.best_full_loss:
                self.best_full_loss = full_loss
                self.save("best")
            if self.epoch % cfg.runtime.checkpoint_every_epochs == 0 or self.epoch == cfg.training.num_epochs:
                self.save(f"epoch{self.epoch}")
                self.save()
        (output / "COMPLETE.json").write_text(json.dumps({"epoch": self.epoch, "global_step": self.global_step}) + "\n")
        self.log("complete")
