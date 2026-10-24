"""Auditable, exact-resume training for the from-scratch Stage 1 v2 Teacher."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from adavip.manifeel.full_sensor_workspace import manifest_fingerprint, normalizer_fingerprint
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace


CHECKPOINT_EPOCHS = (1, 2, 5, 10, 20, 50, 100, 200, 300, 400)


def capture_rng() -> dict:
    """Capture all training random generators without advancing them."""
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    """Restore CPU and CUDA generators after model and loader construction."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def source_fingerprint() -> str:
    """Bind checkpoint code to the active v2 and reused infrastructure sources."""
    root = Path(__file__).resolve().parent
    paths = sorted(root.glob("stage1_v2_*.py"))
    paths += [root / name for name in (
        "full_sensor_dataset.py", "full_sensor_schema.py", "decoded_sensor_cache.py",
        "task_protocol.py", "task_embeddings.py", "action_adapter.py",
    )]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class BalancedTaskSampler(Sampler[int]):
    """Equal task draws; persistent per-task permutations exhaust before reshuffle."""

    def __init__(self, dataset: Any, seed: int, epoch: int = 0, offset: int = 0) -> None:
        self.seed, self.epoch, self.offset = seed, epoch, offset
        self.lengths = list(dataset.task_lengths)
        if not self.lengths or min(self.lengths) <= 0:
            raise ValueError("Every task must have at least one complete valid window")
        self.starts = np.cumsum([0] + self.lengths[:-1]).tolist()
        self.draws_per_task = max(self.lengths)

    def __len__(self) -> int:
        return self.draws_per_task * len(self.lengths) - self.offset

    def __iter__(self) -> Iterator[int]:
        pools, cursors, cycles = [], [], []
        for task_index, length in enumerate(self.lengths):
            cycle, cursor = divmod(self.epoch * self.draws_per_task, length)
            generator = np.random.default_rng([self.seed, task_index, cycle])
            pools.append(generator.permutation(length))
            cursors.append(cursor)
            cycles.append(cycle)
        ordinal = 0
        for _ in range(self.draws_per_task):
            for task_index, length in enumerate(self.lengths):
                if cursors[task_index] == length:
                    cycles[task_index] += 1
                    generator = np.random.default_rng([self.seed, task_index, cycles[task_index]])
                    pools[task_index] = generator.permutation(length)
                    cursors[task_index] = 0
                value = self.starts[task_index] + int(pools[task_index][cursors[task_index]])
                cursors[task_index] += 1
                if ordinal >= self.offset:
                    yield value
                ordinal += 1


def gradient_diagnostics(model: torch.nn.Module, losses: dict[str, Tensor]) -> dict:
    """Measure norms and cosines on shared trainable representation parameters."""
    parameters = [parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad and name.startswith((
                      "image_encoders.", "image_temporal.", "proprio_", "language_projection.",
                      "action_history.", "state_tokenizer.",
                  ))]
    names = ("action_loss", "future_vision_loss", "future_tactile_loss")
    gradients = {
        name: torch.autograd.grad(losses[name], parameters, retain_graph=True, allow_unused=True)
        for name in names
    }
    norms = {name: float(torch.sqrt(sum(
        gradient.detach().float().square().sum() for gradient in gradients[name] if gradient is not None
    ))) for name in names}
    cosines = {}
    for first, second in itertools.combinations(names, 2):
        dot = sum((left.detach().float() * right.detach().float()).sum()
                  for left, right in zip(gradients[first], gradients[second])
                  if left is not None and right is not None)
        cosines[f"{first}/{second}"] = float(dot) / max(norms[first] * norms[second], 1e-30)
    if not all(math.isfinite(value) and value > 0 for value in norms.values()):
        raise FloatingPointError(f"Missing/nonfinite shared gradients: {norms}")
    return {"norms": norms, "cosines": cosines}


def choose_future_weights(diagnostics: list[dict], fraction: float) -> tuple[float, float]:
    """Give each auxiliary objective a measured fraction of the action gradient."""
    weights = []
    for name in ("future_vision_loss", "future_tactile_loss"):
        ratios = [item["norms"]["action_loss"] / item["norms"][name] for item in diagnostics]
        weights.append(float(fraction * np.median(ratios)))
    if not all(math.isfinite(value) and value > 0 for value in weights):
        raise ValueError("Invalid calibrated future weights")
    return tuple(weights)


class Stage1V2Workspace(BaseWorkspace):
    """Jointly train all online modules; preserve the entire exact-resume state."""

    def __init__(self, cfg: DictConfig, output_dir: str | None = None) -> None:
        super().__init__(cfg, output_dir)
        seed = int(cfg.training.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model = hydra.utils.instantiate(cfg.policy)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=[parameter for parameter in self.model.parameters() if parameter.requires_grad],
        )
        self.dataset = None
        self.scheduler = None
        self.epoch = self.global_step = self.sample_cursor = 0
        self.lambda_vision = self.lambda_tactile = None
        self.epoch_sums: dict[str, float] = {}
        self.task_counts: list[int] = []
        self.stop_reason = None
        self.provenance: dict = {}

    def request_stop(self, reason: str) -> None:
        """Stop at the next optimizer boundary, preserving the consumed sample cursor."""
        self.stop_reason = reason

    def log(self, record: dict) -> None:
        """Append finite JSON metrics to persistent storage."""
        record.setdefault("global_step", self.global_step)
        record.setdefault("timestamp", time.time())
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        with (Path(self.output_dir) / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    def loader(self, epoch: int, offset: int = 0) -> DataLoader:
        """Use a private loader RNG so recreating workers does not perturb training."""
        options = OmegaConf.to_container(self.cfg.dataloader, resolve=True)
        if options.get("num_workers", 0) == 0:
            options.pop("prefetch_factor", None)
            options["persistent_workers"] = False
        sampler = BalancedTaskSampler(self.dataset, int(self.cfg.training.seed), epoch, offset)
        generator = torch.Generator().manual_seed(int(self.cfg.training.seed) + epoch)
        return DataLoader(self.dataset, sampler=sampler, generator=generator, **options)

    def _protocol(self) -> dict:
        protocol = OmegaConf.to_container(self.cfg, resolve=True)
        protocol.pop("output_dir", None)
        protocol["training"].pop("resume", None)
        protocol.pop("runtime", None)
        return protocol

    def save(self, tag: str) -> Path:
        """Atomically save all state; milestone files are never overwritten."""
        directory = Path(self.output_dir) / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{tag}.ckpt"
        if tag.startswith("epoch") and path.exists():
            raise FileExistsError(f"Preserved milestone already exists: {path}")
        payload = {
            "cfg": self.cfg, "model": self.model.state_dict(),
            "normalizer": self.model.normalizer.state_dict(),
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch, "global_step": self.global_step, "sample_cursor": self.sample_cursor,
            "epoch_sums": self.epoch_sums, "task_counts": self.task_counts,
            "lambda_vision": self.lambda_vision, "lambda_tactile": self.lambda_tactile,
            "rng": capture_rng(), "provenance": self.provenance,
            "sampler": {"seed": int(self.cfg.training.seed), "epoch": self.epoch,
                        "consumed": self.sample_cursor, "version": "balanced_persistent_permutation_v1"},
        }
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        self.log({"event": "checkpoint", "path": str(path), "epoch": self.epoch,
                  "sample_cursor": self.sample_cursor})
        return path

    def restore(self, path: str, device: torch.device) -> dict:
        """Reject protocol/data/source changes and explicitly rebuild dynamic normalizers."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["provenance"] != self.provenance:
            raise ValueError("Resume rejected: code/data/normalizer/protocol changed")
        self.model.normalizer.load_state_dict(payload["normalizer"], strict=True)
        self.model.normalizer.requires_grad_(False)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(device)
        self.optimizer.load_state_dict(payload["optimizer"])
        optimizer_to(self.optimizer, device)
        self.scheduler.load_state_dict(payload["scheduler"])
        for name in ("epoch", "global_step", "sample_cursor", "epoch_sums", "task_counts",
                     "lambda_vision", "lambda_tactile"):
            setattr(self, name, payload[name])
        return payload["rng"]

    @classmethod
    def create_from_checkpoint(cls, path: str, **kwargs: Any) -> "Stage1V2Workspace":
        """Restore the checkpoint's configured inference policy without opening any dataset."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(payload["cfg"], **kwargs)
        instance.model.normalizer.load_state_dict(payload["normalizer"], strict=True)
        instance.model.normalizer.requires_grad_(False)
        instance.model.load_state_dict(payload["model"], strict=True)
        instance.epoch, instance.global_step = payload["epoch"], payload["global_step"]
        return instance

    def run(self) -> None:
        """Require successful GPU preflight before starting up to 400 full epochs."""
        device = torch.device(str(self.cfg.training.device))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Formal Teacher training requires an allocated CUDA GPU")
        if not 1 <= int(self.cfg.training.num_epochs) <= 400:
            raise ValueError("Stage 1 v2 is capped at 400 epochs")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        report = json.loads(Path(self.cfg.runtime.preflight_report).read_text())
        if not report["passed"] or report["source_sha256"] != source_fingerprint():
            raise ValueError("A passing, source-matched GPU preflight report is required")
        if int(self.cfg.dataloader.batch_size) != report["selected_batch"]:
            raise ValueError("Formal batch differs from the measured GPU selection")
        if self.dataset is None:
            self.dataset = hydra.utils.instantiate(self.cfg.dataset)
        if len(self.dataset.task_specs) != 9 or sum(map(len, self.dataset.train_episodes)) != 400:
            raise ValueError("Formal Stage 1 v2 must train on all nine tasks and all 400 demos")
        normalizer = self.dataset.get_normalizer(mode=str(self.cfg.training.normalizer_mode))
        self.model.set_normalizer(normalizer)
        self.model.to(device)
        self.provenance = {
            "source_sha256": source_fingerprint(), "data": manifest_fingerprint(self.dataset.manifest),
            "normalizer": normalizer_fingerprint(normalizer),
            "protocol": manifest_fingerprint(self._protocol()),
        }
        if report["data_sha256"] != self.provenance["data"]:
            raise ValueError("GPU preflight data contract differs")
        epoch_samples = len(BalancedTaskSampler(self.dataset, int(self.cfg.training.seed)))
        steps_per_epoch = math.ceil(epoch_samples / int(self.cfg.dataloader.batch_size))
        warmup = int(self.cfg.training.lr_warmup_steps)
        total_steps = steps_per_epoch * int(self.cfg.training.num_epochs)

        def lr_multiplier(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_multiplier)
        self.lambda_vision, self.lambda_tactile = report["lambda_vision"], report["lambda_tactile"]
        pending_rng = None
        if self.cfg.training.resume:
            pending_rng = self.restore(str(self.cfg.training.resume), device)
        (output / "config.yaml").write_text(OmegaConf.to_yaml(self.cfg, resolve=True))
        (output / "dataset_manifest.json").write_text(json.dumps(self.dataset.manifest, indent=2))
        self.log({"event": "workspace_ready", "device": str(device),
                  "parameters": self.model.parameter_counts(), "task_windows": self.dataset.task_lengths,
                  "windows": len(self.dataset), "epoch_samples": epoch_samples,
                  "steps_per_epoch": steps_per_epoch, "batch_size": int(self.cfg.dataloader.batch_size),
                  "provenance": self.provenance, "lambda_vision": self.lambda_vision,
                  "lambda_tactile": self.lambda_tactile})
        for epoch_index in range(self.epoch, int(self.cfg.training.num_epochs)):
            self.model.train()
            loader = self.loader(epoch_index, self.sample_cursor)
            iterator = iter(loader)
            if pending_rng is not None:
                restore_rng(pending_rng)
                pending_rng = None
            if not self.epoch_sums:
                self.epoch_sums = {name: 0.0 for name in (
                    "action_loss", "future_vision_loss", "future_tactile_loss",
                    "vision_latent_variance", "tactile_latent_variance", "samples",
                )}
                self.task_counts = [0] * len(self.dataset.task_specs)
            previous_finished = time.perf_counter()
            for batch in iterator:
                started = time.perf_counter()
                data_wait_seconds = started - previous_finished
                if self.stop_reason:
                    self.save("latest")
                    self.log({"event": "stopped", "reason": self.stop_reason})
                    return
                batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                self.optimizer.zero_grad(set_to_none=True)
                losses = self.model.compute_loss_with_metrics(batch)
                if self.global_step % int(self.cfg.training.gradient_audit_every) == 0:
                    self.log({"event": "gradient_audit", **gradient_diagnostics(self.model, losses)})
                loss = losses["action_loss"] + self.lambda_vision * losses["future_vision_loss"] \
                    + self.lambda_tactile * losses["future_tactile_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; stopping without overwriting milestones")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg.training.grad_clip_norm), error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.model.update_target_encoder()
                self.scheduler.step()
                self.global_step += 1
                size = batch["action"].shape[0]
                self.sample_cursor += size
                self.epoch_sums["samples"] += size
                for name in self.epoch_sums:
                    if name != "samples":
                        value = float(losses[name].detach())
                        if not math.isfinite(value):
                            raise FloatingPointError(f"Nonfinite {name}")
                        self.epoch_sums[name] += value * size
                counts = np.bincount(batch["task_index"].cpu().numpy(), minlength=len(self.task_counts))
                self.task_counts = (np.asarray(self.task_counts) + counts).tolist()
                if self.global_step % int(self.cfg.training.log_every) == 0:
                    self.log({"event": "train", "epoch": epoch_index + 1,
                              "sample_cursor": self.sample_cursor, "loss": float(loss.detach()),
                              "gradient_norm": float(norm), "step_seconds": time.perf_counter() - started,
                              "data_wait_seconds": data_wait_seconds,
                              "samples_per_second": size / max(time.perf_counter() - previous_finished, 1e-12),
                              "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                              **{name: float(losses[name].detach()) for name in self.epoch_sums if name != "samples"}})
                if self.global_step % int(self.cfg.training.checkpoint_every_steps) == 0:
                    self.save("latest")
                previous_finished = time.perf_counter()
            samples = self.epoch_sums["samples"]
            self.epoch = epoch_index + 1
            order = np.asarray(list(BalancedTaskSampler(self.dataset, int(self.cfg.training.seed), epoch_index)))
            starts = np.cumsum([0] + self.dataset.task_lengths)
            coverage = [int(np.unique(order[(order >= first) & (order < last)]).size)
                        for first, last in zip(starts[:-1], starts[1:])]
            self.log({"event": "epoch_end", "epoch": self.epoch, "task_draws": self.task_counts,
                      "unique_task_windows_covered": coverage,
                      **{name: value / samples for name, value in self.epoch_sums.items() if name != "samples"}})
            self.sample_cursor = 0
            self.epoch_sums = {}
            self.save("latest")
            if self.epoch in CHECKPOINT_EPOCHS:
                self.save(f"epoch{self.epoch}")


__all__ = ["BalancedTaskSampler", "CHECKPOINT_EPOCHS", "Stage1V2Workspace"]
