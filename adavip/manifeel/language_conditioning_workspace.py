"""From-scratch L1/L2 action-only training with exact resumable random streams."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Iterator

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Sampler

from adavip.manifeel.full_sensor_workspace import (
    FullSensorWorkspace, manifest_fingerprint, normalizer_fingerprint, training_protocol,
)
from adavip.manifeel.language_conditioning_dataset import initialize_worker
from adavip.manifeel.language_conditioning_diagnostics import measure_probes, select_probe_indices
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


class BalancedEpochSampler(Sampler[int]):
    """Replay the same task-balanced epoch independently of worker and model RNG."""

    def __init__(self, dataset: object, seed: int, batch_size: int) -> None:
        self.weights = torch.cat([torch.full((n,), 1 / n, dtype=torch.double) for n in dataset.task_lengths])
        self.seed, self.batch_size, self.epoch, self.offset = seed, batch_size, 0, 0

    def set_epoch(self, epoch: int, batch_offset: int) -> None:
        """Select the unconsumed suffix without loading already consumed samples."""
        self.epoch, self.offset = epoch, batch_offset * self.batch_size

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.multinomial(self.weights, len(self.weights), True, generator=generator)
        return iter(order[self.offset:].tolist())

    def __len__(self) -> int:
        return max(0, len(self.weights) - self.offset)


def source_fingerprint() -> str:
    """Bind resume to policy, data, diagnostics and training source contents."""
    root = Path(__file__).parent
    names = ("language_conditioned_obs_encoder.py", "language_conditioning_dataset.py",
             "language_conditioning_diagnostics.py", "language_conditioning_workspace.py",
             "reference_diffusion_policy.py", "full_sensor_obs_encoder.py", "full_sensor_dataset.py",
             "full_sensor_workspace.py", "decoded_sensor_cache.py", "full_sensor_schema.py",
             "task_embeddings.py", "task_protocol.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def common_initialization_fingerprint(policy: torch.nn.Module) -> str:
    """Hash every common parameter/buffer, excluding the two different heads."""
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        if name.startswith("obs_encoder.language_head."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class LanguageConditioningWorkspace(FullSensorWorkspace):
    """Reuse DP checkpoint format while training only full-input Action Loss."""

    include_keys = FullSensorWorkspace.include_keys + ("probe_indices", "cumulative_task_counts",)

    def __init__(self, cfg: object, output_dir: str | None = None) -> None:
        super().__init__(cfg, output_dir=output_dir)
        self.probe_indices: list[list[int]] = []
        self.cumulative_task_counts: list[int] = []
        self._noise_generator: torch.Generator | None = None
        self._persistent_loader: DataLoader | None = None
        self._sampler: BalancedEpochSampler | None = None
        self._transport_pause_seconds = 0.0
        self.common_initialization = common_initialization_fingerprint(self.model)

    def _capture_rng(self) -> None:
        super()._capture_rng()
        self.rng_state["training_noise"] = self._noise_generator.get_state()

    def _restore_rng(self) -> None:
        super()._restore_rng()
        self._noise_generator.set_state(self.rng_state["training_noise"])

    def _epoch_batches(self, dataset: object, epoch: int) -> Iterator[dict]:
        cfg = self.cfg
        if self._sampler is None:
            self._sampler = BalancedEpochSampler(dataset, int(cfg.training.seed), int(cfg.dataloader.batch_size))
        self._sampler.set_epoch(epoch, self.batch_in_epoch)
        if dataset.cache_directory and not dataset.verified_ram_cache:
            # This is the next actual training update, not a discarded preflight.
            if len(self._sampler):
                cold_loader = DataLoader(dataset, batch_size=cfg.dataloader.batch_size, sampler=self._sampler,
                                         num_workers=0, pin_memory=bool(cfg.dataloader.pin_memory),
                                         generator=torch.Generator().manual_seed(700042))
                yield next(iter(cold_loader))
                del cold_loader
            paused = time.perf_counter()
            self._save_latest()
            self._log_progress({"event": "first_formal_update_saved_before_cache", "global_step": self.global_step})
            if self._should_stop():
                return
            started = time.perf_counter()
            dataset.load_ram_cache()
            self._log_progress({"event": "all_data_ram_ready", "seconds": time.perf_counter() - started})
            self._transport_pause_seconds = time.perf_counter() - paused
            self._sampler.set_epoch(epoch, self.batch_in_epoch)
        if self._persistent_loader is None:
            options = OmegaConf.to_container(cfg.dataloader, resolve=True)
            options.pop("shuffle", None)
            if options["num_workers"] == 0:
                options.pop("prefetch_factor", None)
                options["persistent_workers"], options["timeout"] = False, 0
            self._persistent_loader = DataLoader(
                dataset, sampler=self._sampler, worker_init_fn=initialize_worker,
                generator=torch.Generator().manual_seed(int(cfg.training.seed) + 700000), **options,
            )
        yield from self._persistent_loader

    def _save_latest(self) -> None:
        self._capture_rng()
        self._save("latest")

    def _finish_stop(self, output: Path) -> None:
        self._save_latest()
        (output / "SAFE_STOP.json").write_text(json.dumps({"step": self.global_step, "reason": self._stop_requested}))
        self._log_progress({"event": "safe_stop", "global_step": self.global_step,
                            "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch})

    def run(self) -> None:
        """Start formal optimization immediately, then preload and use all decoded RAM."""
        cfg, device = self.cfg, self._device()
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not cfg.resume and (output / "checkpoints/latest.ckpt").exists():
            raise FileExistsError("Use resume or a new output directory")
        dataset = self._make_dataset()
        if cfg.runtime.expected_samples and len(dataset) != cfg.runtime.expected_samples:
            raise ValueError("Unexpected all-demonstration anchor count")
        if cfg.runtime.expected_episodes and sum(map(len, dataset.train_episodes)) != cfg.runtime.expected_episodes:
            raise ValueError("Unexpected all-demonstration episode count")
        if any(len(episodes) for episodes in dataset.val_episodes):
            raise ValueError("Validation episodes are forbidden in this comparison")
        if float(cfg.policy.full_input_probability) != 1:
            raise ValueError("Sensor and language dropout must remain disabled")
        normalizer = dataset.get_normalizer()
        expected = {"manifest": manifest_fingerprint(dataset.manifest),
                    "normalizer": normalizer_fingerprint(normalizer),
                    "protocol": manifest_fingerprint(training_protocol(cfg)), "source": source_fingerprint()}
        for policy in (self.model, self.ema_model):
            if policy is not None:
                policy.set_normalizer(normalizer)
        self._noise_generator = torch.Generator(device=device).manual_seed(int(cfg.training.seed) + 300000)
        if cfg.resume:
            self.load_checkpoint(path=cfg.resume, map_location="cpu", include_keys=self.include_keys)
            if self.data_provenance != expected:
                raise ValueError("Resume rejected: model/data/normalizer/protocol/source differs")
            if self.probe_indices != select_probe_indices(dataset, cfg.diagnostics.samples_per_task, cfg.diagnostics.seed):
                raise ValueError("Fixed training probes changed")
        else:
            self.data_provenance = expected
            self.probe_indices = select_probe_indices(dataset, cfg.diagnostics.samples_per_task, cfg.diagnostics.seed)
            self.cumulative_task_counts = [0] * len(dataset.task_specs)
        for policy in (self.model, self.ema_model):
            if policy is not None:
                policy.to(device)
                policy.normalizer.to(device)
                policy.normalizer.requires_grad_(False)
                if normalizer_fingerprint(policy.normalizer) != expected["normalizer"]:
                    raise ValueError("Restored normalizer differs")
        optimizer_to(self.optimizer, device)
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if self.ema_model is not None else None
        if ema is not None:
            ema.optimization_step = self.ema_step
        if cfg.resume:
            self._restore_rng()
        self._write_metadata(dataset, expected, output)
        (output / "probe_manifest.json").write_text(json.dumps({
            "task_ids": [spec.task_id for spec in dataset.task_specs], "dataset_indices": self.probe_indices,
            "seed": cfg.diagnostics.seed, "scope": "complete windows from training demonstrations"}, indent=2))
        (output / "initialization.json").write_text(json.dumps({
            "common_initialization_sha256": self.common_initialization,
            "variant": cfg.variant, "precision": "float32", "torch": torch.__version__}, indent=2))
        self._log_progress({"event": "workspace_ready", "variant": cfg.variant, "samples": len(dataset),
                            "episodes": sum(map(len, dataset.train_episodes)), "global_step": self.global_step,
                            "batch_size": cfg.dataloader.batch_size, "device": str(device),
                            "common_initialization_sha256": self.common_initialization})
        epoch_steps = math.ceil(len(dataset) / cfg.dataloader.batch_size)
        total_steps = epoch_steps * cfg.training.num_epochs
        starting_step = self.global_step
        for epoch_index in range(self.epoch, cfg.training.num_epochs):
            if not 0 <= self.batch_in_epoch <= epoch_steps:
                raise ValueError("Invalid checkpoint cursor")
            epoch_started = time.perf_counter()
            self.model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if not self.epoch_metrics_state:
                self.epoch_metrics_state = {
                    "condition_sums": torch.zeros(10, dtype=torch.float64),
                    "condition_counts": torch.zeros(10, dtype=torch.float64),
                    "task_sums": torch.zeros(len(dataset.task_specs), dtype=torch.float64),
                    "task_counts": torch.zeros(len(dataset.task_specs), dtype=torch.float64),
                    "train_seconds": 0.0, "data_seconds": 0.0, "max_grad_norm": 0.0,
                }
            state = self.epoch_metrics_state
            requested = time.perf_counter()
            for batch in self._epoch_batches(dataset, epoch_index):
                requested += self._transport_pause_seconds
                self._transport_pause_seconds = 0.0
                data_seconds = time.perf_counter() - requested
                if self._should_stop():
                    self._finish_stop(output)
                    return
                batch = dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))
                for group in self.optimizer.param_groups:
                    group["lr"] = cfg.optimizer.lr * self._lr_multiplier(total_steps)
                self.optimizer.zero_grad(set_to_none=True)
                condition_ids = torch.zeros(len(batch["action"]), dtype=torch.long, device=device)
                sample = self.model.prepare_denoising_sample(batch, generator=self._noise_generator)
                losses = self.model.compute_loss_per_sample(batch, condition_ids, sample)
                if not torch.isfinite(losses).all():
                    raise FloatingPointError("Nonfinite Action Loss")
                losses.mean().backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.gradient_clip_norm,
                                                     error_if_nonfinite=True)
                self.optimizer.step()
                if ema is not None:
                    ema.step(self.model)
                    self.ema_step = ema.optimization_step
                self.global_step += 1
                self.batch_in_epoch += 1
                count, mean_loss = self._accumulate(state, batch, losses, condition_ids)
                counts = torch.bincount(batch["task_index"].cpu(), minlength=len(dataset.task_specs)).tolist()
                self.cumulative_task_counts = [old + new for old, new in zip(self.cumulative_task_counts, counts)]
                state["max_grad_norm"] = max(state["max_grad_norm"], float(norm))
                state["train_seconds"] += time.perf_counter() - requested
                state["data_seconds"] += data_seconds
                if self.global_step == 1 or self.global_step % cfg.training.log_every_steps == 0:
                    record = {"event": "train_step", "variant": cfg.variant, "epoch": epoch_index + 1,
                              "global_step": self.global_step, "batch_in_epoch": self.batch_in_epoch,
                              "epoch_steps": epoch_steps, "action_loss": mean_loss,
                              "lr": self.optimizer.param_groups[0]["lr"], "grad_norm": float(norm),
                              "samples_per_second": float(state["task_counts"].sum()) / state["train_seconds"],
                              "data_wait_fraction": state["data_seconds"] / state["train_seconds"],
                              "task_counts": state["task_counts"].long().tolist(),
                              "per_task_action_loss": [float(s / n) if n else None
                                                       for s, n in zip(state["task_sums"], state["task_counts"])],
                              "cumulative_task_counts": self.cumulative_task_counts}
                    if device.type == "cuda":
                        record.update(peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                                      peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3)
                    self._log_progress(record)
                limit = cfg.runtime.stop_after_steps
                if limit and self.global_step - starting_step >= limit:
                    self.request_stop("bounded_cpu_test")
                if self._should_stop():
                    self._finish_stop(output)
                    return
                if self.global_step % cfg.training.checkpoint_every_steps == 0:
                    self._save_latest()
                requested = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            if self._should_stop():
                self._finish_stop(output)
                return
            diagnostic_started = time.perf_counter()
            diagnostic = measure_probes(
                self.model, dataset, self.probe_indices,
                gradient=(epoch_index + 1 in (1, 2) or (epoch_index + 1) % cfg.diagnostics.gradient_every == 0),
                noise_repeats=cfg.diagnostics.noise_repeats, gradient_samples=cfg.diagnostics.gradient_samples,
                batch_size=cfg.diagnostics.batch_size, seed=cfg.diagnostics.seed)
            diagnostic.update(epoch=epoch_index + 1, global_step=self.global_step,
                              task_ids=[spec.task_id for spec in dataset.task_specs],
                              seconds=time.perf_counter() - diagnostic_started)
            with (output / "diagnostics.jsonl").open("a") as stream:
                stream.write(json.dumps(diagnostic, allow_nan=False) + "\n")
            metrics = {"epoch": epoch_index + 1, "global_step": self.global_step,
                       "train/action_loss": float(state["task_sums"].sum() / state["task_counts"].sum()),
                       "train/task_counts": state["task_counts"].long().tolist(),
                       "train/per_task_action_loss": (state["task_sums"] / state["task_counts"]).tolist(),
                       "train/samples_per_second": len(dataset) / state["train_seconds"],
                       "runtime/diagnostic_seconds": diagnostic["seconds"],
                       "runtime/epoch_wall_seconds": time.perf_counter() - epoch_started,
                       "cumulative_task_counts": self.cumulative_task_counts}
            self.epoch, self.batch_in_epoch, self.epoch_metrics_state = epoch_index + 1, 0, {}
            self._save_latest()
            if self.epoch in cfg.training.checkpoint_epochs:
                path = output / "checkpoints" / f"epoch{self.epoch}.ckpt"
                if path.exists():
                    raise FileExistsError("Refusing to overwrite a fixed epoch checkpoint")
                # Pin the complete atomic latest inode; no second multi-GB serialization.
                os.link(output / "checkpoints/latest.ckpt", path)
            self._write_metrics(metrics, output)
        (output / "COMPLETE.json").write_text(json.dumps({"epochs": self.epoch, "updates": self.global_step}))
