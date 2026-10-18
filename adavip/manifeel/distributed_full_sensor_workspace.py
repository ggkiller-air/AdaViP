"""Single-node DDP with an unchanged global sampler and shared decoded file pages."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterator

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler, SequentialSampler

from adavip.manifeel.full_sensor_workspace import FullSensorWorkspace, manifest_fingerprint


class GlobalBatchShardSampler(Sampler):
    """Shard each original global batch without extending or truncating its sample budget.

    A negative index is a zero-weight dummy only when a rank has no tail samples.
    Weighted sampling replacement is preserved; no DistributedSampler padding is used.
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, rank: int,
                 world_size: int, start_batch: int = 0, audit: Any = None) -> None:
        if batch_size < world_size or not 0 <= rank < world_size or start_batch < 0:
            raise ValueError("Invalid global batch, rank, world size or sampler cursor")
        self.sampler, self.batch_size = sampler, batch_size
        self.rank, self.world_size, self.start_batch = rank, world_size, start_batch
        self.audit = audit

    def __len__(self) -> int:
        return max(0, math.ceil(len(self.sampler) / self.batch_size) - self.start_batch)

    def __iter__(self) -> Iterator[list[int]]:
        plan = list(self.sampler)
        if self.audit is not None:
            self.audit(plan)
        remaining = itertools.islice(iter(plan), self.start_batch * self.batch_size, None)
        while global_batch := list(itertools.islice(remaining, self.batch_size)):
            yield global_batch[self.rank::self.world_size] or [-global_batch[0] - 1]


class ValidityDataset(Dataset):
    """Attach a validity bit to the rare empty-rank dummy without changing real samples."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        valid = index >= 0
        result = dict(self.dataset[index if valid else -index - 1])
        result["_distributed_valid"] = torch.tensor(valid)
        return result


class PolicyLossModule(nn.Module):
    """Route unreduced policy losses through DDP.forward and its reducer hooks."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict, condition_ids: torch.Tensor) -> torch.Tensor:
        """Return one differentiable loss per local sample."""
        return self.policy.compute_loss_per_sample(batch, condition_ids)


def cache_memory_report(dataset: Any) -> dict:
    """Report mapped NPY identities and physical proportional memory, not summed RSS."""
    identities = []
    nbytes = 0
    for group in dataset.groups:
        for key, array in sorted(group["data"].items()):
            if not isinstance(array.values, np.memmap):
                raise ValueError("DDP requires read-only mmap cache, never four private RAM copies")
            path = Path(array.values.filename)
            metadata = path.stat()
            identities.append((str(path), metadata.st_dev, metadata.st_ino, metadata.st_size))
            nbytes += array.values.nbytes
    totals = {"Rss": 0, "Pss": 0, "Shared_Clean": 0, "Private_Clean": 0}
    selected = False
    for line in Path("/proc/self/smaps").read_text().splitlines():
        if line and "-" in line.split()[0]:
            selected = line.endswith(".npy")
        elif selected and line.split(":", 1)[0] in totals:
            field, amount = line.split(":", 1)
            totals[field] += int(amount.split()[0]) * 1024
    return {"pid": os.getpid(), "host": os.uname().nodename, "nbytes": nbytes,
            "file_identity": manifest_fingerprint(identities), "mapped_bytes": totals}


class DistributedFullSensorWorkspace(FullSensorWorkspace):
    """Keep raw checkpoint keys while synchronizing loss, EMA, validation and RNG."""

    exclude_keys = ("_ddp",)

    def __init__(self, cfg: Any, output_dir: str | None = None) -> None:
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        super().__init__(cfg, output_dir)
        self._ddp: DistributedDataParallel | None = None

    def _device(self) -> torch.device:
        if self.cfg.training.device == "cpu":
            return torch.device("cpu")
        return torch.device("cuda", self.local_rank)

    def _reduce(self, values: torch.Tensor) -> torch.Tensor:
        result = values.to(self._device())
        distributed.all_reduce(result)
        return result.cpu()

    def _make_dataset(self) -> Any:
        if not self.cfg.dataset.get("decoded_cache_dir") or self.cfg.dataset.decoded_cache_mode != "mmap":
            raise ValueError("Single-node DDP requires the verified shared mmap decoded cache")
        dataset = None
        if self.rank == 0:
            dataset = super()._make_dataset()
        distributed.barrier()
        if self.rank != 0:
            dataset = super()._make_dataset()
        distributed.barrier()
        reports = [None] * self.world_size
        distributed.all_gather_object(reports, cache_memory_report(dataset))
        if len({report["file_identity"] for report in reports}) != 1 or len({report["host"] for report in reports}) != 1:
            raise ValueError("DDP ranks must share identical cache file inodes on one node")
        self._log_progress({"event": "shared_cache_verified", "ranks": reports})
        return dataset

    def _loader(self, dataset: Any, training: bool, epoch: int = 0, start_batch: int = 0) -> DataLoader:
        options = OmegaConf.to_container(self.cfg.dataloader if training else self.cfg.val_dataloader, resolve=True)
        if options.pop("shuffle", False) or options.pop("drop_last", False):
            raise ValueError("DDP must preserve the original sampler and partial final batch")
        global_batch = options.pop("batch_size")
        if options["num_workers"]:
            options["multiprocessing_context"] = "spawn"
        else:
            options.pop("prefetch_factor", None)
            options.pop("multiprocessing_context", None)
            options.update(persistent_workers=False, timeout=0)
        generator = torch.Generator().manual_seed(int(self.cfg.training.seed) + epoch)
        sampler = dataset.make_task_balanced_sampler(generator) if training else SequentialSampler(dataset)
        audit = (lambda plan: self._audit_sampler(plan, epoch, start_batch)) if training else None
        sharded = GlobalBatchShardSampler(sampler, global_batch, self.rank, self.world_size, start_batch, audit)
        return DataLoader(ValidityDataset(dataset), batch_sampler=sharded, generator=generator, **options)

    def _audit_sampler(self, plan: list[int], epoch: int, start_batch: int) -> None:
        digest = hashlib.sha256(np.asarray(plan, dtype=np.int64).tobytes()).hexdigest()
        signatures = [None] * self.world_size
        distributed.all_gather_object(signatures, digest)
        if len(set(signatures)) != 1:
            raise ValueError("Ranks generated different global weighted-sampling plans")
        batch_size = int(self.cfg.dataloader.batch_size)
        tail = len(plan) % batch_size or batch_size
        self._log_progress({"event": "global_sampler_verified", "epoch": epoch + 1,
                            "global_samples": len(plan), "global_batches": math.ceil(len(plan) / batch_size),
                            "tail_samples_per_rank": [len(range(rank, tail, self.world_size))
                                                      for rank in range(self.world_size)],
                            "start_batch": start_batch, "sample_order_sha256": digest})

    def _initialize_runtime(self, device: torch.device) -> None:
        """Keep constructor bucket layouts fixed across checkpoint reconstruction."""
        self._ddp = DistributedDataParallel(
            PolicyLossModule(self.model), device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=100,
            find_unused_parameters=True,
        )
        self._check_replicas("initialized")
        if not self.cfg.resume:
            seed = int(self.cfg.training.seed) + self.rank
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def _check_replicas(self, phase: str) -> None:
        signatures = {}
        for name in ("model", "ema_model"):
            model = getattr(self, name)
            if model is not None:
                digest = hashlib.sha256()
                for key, tensor in sorted(model.state_dict().items()):
                    digest.update(key.encode())
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                signatures[name] = digest.hexdigest()
        signatures.update(global_step=self.global_step, ema_step=self.ema_step, batch_in_epoch=self.batch_in_epoch)
        replicas = [None] * self.world_size
        distributed.all_gather_object(replicas, signatures)
        if any(replica != replicas[0] for replica in replicas):
            raise ValueError("Raw/EMA replicas or optimizer cursors diverged")
        if self.ema_model is not None and self.ema_step != self.global_step:
            raise ValueError("EMA must be updated exactly once per global optimizer update")
        self._log_progress({"event": "replicas_verified", "phase": phase, **signatures})

    def _training_losses(self, batch: dict, condition_ids: torch.Tensor) -> torch.Tensor:
        return self._ddp(batch, condition_ids)

    def _backward(self, losses: torch.Tensor, batch: dict) -> None:
        valid = batch["_distributed_valid"].to(losses.dtype)
        count = valid.sum()
        distributed.all_reduce(count)
        (losses.mul(valid).sum() * self.world_size / count).backward()

    def _validation_losses(self, policy: Any, batch: dict, generator: torch.Generator,
                           dataset: Any, batch_index: int) -> torch.Tensor:
        batch_size = int(self.cfg.val_dataloader.batch_size)
        count = min(batch_size, len(dataset) - batch_index * batch_size)
        actions = batch["action"]
        noise = torch.randn((count, *actions.shape[1:]), generator=generator,
                            device=actions.device, dtype=actions.dtype)
        timesteps = torch.randint(policy.noise_scheduler.config.num_train_timesteps, (count,),
                                  generator=generator, device=actions.device)
        indices = torch.arange(count, device=actions.device)[self.rank::self.world_size]
        if not len(indices):
            indices = torch.zeros(1, dtype=torch.long, device=actions.device)
        return policy.paired_denoising_losses(batch, noise=noise[indices], timesteps=timesteps[indices])["losses"]

    def _capture_rng(self) -> None:
        local_state = self._local_rng()
        states = [None] * self.world_size
        distributed.all_gather_object(states, local_state)
        self.rng_state = {"world_size": self.world_size, "ranks": states}

    def _local_rng(self) -> dict:
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": [torch.cuda.get_rng_state(self._device())] if self._device().type == "cuda" else []}

    def _restore_local_rng(self, saved: dict) -> None:
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["torch"].cpu())
        if saved["cuda"] and self._device().type == "cuda":
            torch.cuda.set_rng_state(saved["cuda"][0].cpu(), self._device())

    def _restore_rng(self) -> None:
        saved = self.rng_state
        if "ranks" in saved:
            if saved["world_size"] != self.world_size:
                raise ValueError("Exact DDP resume requires the same world size")
            selected = saved["ranks"][self.rank]
            self._restore_local_rng(selected)
            actual = self._local_rng()
            if (actual["python"] != selected["python"] or actual["numpy"][0] != selected["numpy"][0]
                    or not np.array_equal(actual["numpy"][1], selected["numpy"][1])
                    or actual["numpy"][2:] != selected["numpy"][2:]
                    or not torch.equal(actual["torch"], selected["torch"].cpu())
                    or any(not torch.equal(value, expected.cpu())
                           for value, expected in zip(actual["cuda"], selected["cuda"]))):
                raise ValueError("Per-rank checkpoint RNG restoration failed")
        else:
            self._restore_local_rng(saved)
            if self.rank:
                seed = int(self.cfg.training.seed) + self.global_step * self.world_size + self.rank
                random.seed(seed)
                np.random.seed(seed % 2 ** 32)
                torch.manual_seed(seed)
        self.rng_state = saved
        self._log_progress({"event": "resume_rng_verified", "world_size": self.world_size,
                            "source_world_size": saved.get("world_size", 1), "global_step": self.global_step,
                            "batch_in_epoch": self.batch_in_epoch})

    def _should_stop(self) -> bool:
        value = torch.tensor(int(super()._should_stop()), device=self._device())
        distributed.all_reduce(value, op=distributed.ReduceOp.MAX)
        if value.item() and not self._stop_requested:
            self.request_stop("peer_stop")
        return bool(value.item())

    def _save(self, tag: str) -> str:
        distributed.barrier()
        path = super()._save(tag) if self.rank == 0 else str(Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt")
        distributed.barrier()
        return path

    def _log_progress(self, record: dict) -> None:
        if record["event"] == "train_step":
            limits = torch.tensor([-record["samples_per_second"], record["data_wait_fraction"],
                                   record.get("peak_allocated_gib", 0)], device=self._device(), dtype=torch.float64)
            distributed.all_reduce(limits, op=distributed.ReduceOp.MAX)
            record.update(samples_per_second=-limits[0].item(), data_wait_fraction=limits[1].item(),
                          peak_allocated_gib=limits[2].item(), world_size=self.world_size,
                          global_batch_size=int(self.cfg.dataloader.batch_size),
                          per_rank_batch_size=int(self.cfg.dataloader.batch_size) // self.world_size)
        if self.rank == 0:
            super()._log_progress(record)

    def _write_metadata(self, dataset: Any, expected: dict, output: Path) -> None:
        if self.rank == 0:
            super()._write_metadata(dataset, expected, output)
        distributed.barrier()

    def _write_metrics(self, metrics: dict, output: Path) -> None:
        self._check_replicas("epoch_complete")
        if self.rank == 0:
            super()._write_metrics(metrics, output)
