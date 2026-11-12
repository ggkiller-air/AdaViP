"""Stage2 batch comparisons with a fixed optimizer-update clock and real DDP."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import dill
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from adavip.manifeel.stage2_ablation_workspace import initialize_ablation_worker, tensor_fingerprint
from scripts.manifeel.train_stage2_t2 import (
    atomic_checkpoint, make_optimizer_scheduler, restore_policy_state,
    set_stage_trainability, update_ema,
)

REFERENCE_STEPS = 2040
TOTAL_UPDATES = 1224000
FIXED_EPOCHS = (1, 6, 10, 20, 50, 100, 200, 300, 400, 500, 600)


def stage_for_update(completed: int) -> str:
    """Select the phase of the NEXT update, including both exact boundaries."""
    return 'A' if completed < 2040 else 'B' if completed < 12240 else 'C'


def shard_bounds(size: int, rank: int, world: int) -> tuple[int, int]:
    """Contiguous nonduplicating shards, with correct unequal final batches."""
    if not 0 <= rank < world or size < world:
        raise ValueError('This protocol requires at least one real example per rank')
    base, extra = divmod(size, world)
    start = rank * base + min(rank, extra)
    return start, start + base + int(rank < extra)


def weighted_ddp_loss(action, utility, stage: str, global_size: int, world: int):
    """DDP averages ranks; rescale local sums to the global per-example mean."""
    value = utility.sum() if stage == 'A' else action.sum() + .05 * utility.sum()
    return value * (world / global_size)


class EqualUpdateSampler(Sampler):
    """Replay the original 42+data_epoch balanced draws, then shard global batches."""

    def __init__(self, dataset, batch_size: int, rank: int, world: int):
        self.dataset, self.batch_size, self.rank, self.world = dataset, batch_size, rank, world
        self.epoch, self.cursor = 1, 0

    def plan(self):
        generator = torch.Generator().manual_seed(42 + self.epoch)
        return list(self.dataset.make_task_balanced_sampler(generator))

    def __iter__(self):
        plan = self.plan()
        digest = hashlib.sha256(np.asarray(plan, dtype=np.int64).tobytes()).hexdigest()
        if dist.is_initialized():
            digests = [None] * self.world
            dist.all_gather_object(digests, digest)
            if len(set(digests)) != 1:
                raise RuntimeError('Global sampling plans differ across ranks')
        for first in range(self.cursor * self.batch_size, len(plan), self.batch_size):
            batch = plan[first:first + self.batch_size]
            start, stop = shard_bounds(len(batch), self.rank, self.world)
            yield batch[start:stop]

    def __len__(self):
        return max(0, math.ceil(len(self.dataset) / self.batch_size) - self.cursor)


class EqualUpdateLoss(nn.Module):
    """Expose just the two per-example losses to DDP's autograd traversal."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, stage):
        result = self.policy.training_losses(batch, stage)
        return result['action_loss_per_sample'], result['utility_loss_per_sample']


class EqualUpdateTrainer:
    """Keep data-pass cursors separate from phase, LR, EMA and comparison clocks."""

    def __init__(self, policy, dataset, cfg, contract, output, workers=2):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
        self.policy = policy.to(self.device)
        self.ema = copy.deepcopy(policy).eval().requires_grad_(False)
        self.dataset, self.cfg, self.contract = dataset, cfg, contract
        self.output = Path(output)
        self.output.mkdir(exist_ok=True, parents=True)
        self.contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.batch_size, self.workers = contract['batch_size'], workers
        self.steps_per_data_epoch = math.ceil(len(dataset) / self.batch_size)
        # Scientific schedule ALWAYS uses B32's reference clock.
        self.optimizer, self.scheduler = make_optimizer_scheduler(policy, REFERENCE_STEPS, 600)
        self.global_step, self.data_epoch, self.cursor = 0, 1, 0
        self.stats = torch.zeros(3, 9, dtype=torch.float64)
        self.samples_seen = 0
        self.ddp, self.stage, self.loader = None, None, None
        self.stop_requested = False
        self.sampler = EqualUpdateSampler(dataset, self.batch_size, self.rank, self.world)

    def rng(self):
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.device))

    def restore_rng(self, state):
        random.setstate(state['python']); np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'], self.device)

    def set_stage(self, stage):
        if stage == self.stage:
            return
        state = self.rng()
        self.ddp = None
        set_stage_trainability(self.policy, stage)
        self.policy.train()
        self.ddp = DistributedDataParallel(EqualUpdateLoss(self.policy),
            device_ids=[self.device.index], output_device=self.device.index,
            broadcast_buffers=False, find_unused_parameters=True,
            gradient_as_bucket_view=True, bucket_cap_mb=100)
        self.stage = stage
        self.restore_rng(state)

    def log(self, event, **values):
        if self.rank == 0:
            row = dict(event=event, optimizer_updates=self.global_step,
                       reference_epoch=self.global_step / REFERENCE_STEPS,
                       data_epoch=self.data_epoch, batch_cursor=self.cursor,
                       stage=self.stage, **values)
            with (self.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row, allow_nan=False), flush=True)

    def assert_replicas(self):
        signatures = [None] * self.world
        signature = (tensor_fingerprint(self.policy.state_dict()), tensor_fingerprint(self.ema.state_dict()))
        dist.all_gather_object(signatures, signature)
        if len(set(signatures)) != 1:
            raise RuntimeError('Raw/EMA replicas diverged')

    def save(self, fixed=False):
        self.assert_replicas()
        states = [None] * self.world
        dist.all_gather_object(states, self.rng())
        if self.rank == 0:
            payload = dict(model=self.policy.state_dict(), ema=self.ema.state_dict(),
                optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                cfg=self.cfg, contract=self.contract, contract_hash=self.contract_hash,
                global_step=self.global_step, optimizer_updates=self.global_step,
                epoch=self.global_step // REFERENCE_STEPS, reference_epoch=self.global_step / REFERENCE_STEPS,
                data_epoch=self.data_epoch, batch_cursor=self.cursor, rng_by_rank=states,
                stats=self.stats, samples_seen=self.samples_seen, world_size=self.world,
                stage=stage_for_update(self.global_step))
            path = self.output / 'checkpoints/latest.ckpt'
            atomic_checkpoint(path, payload)
            epoch = self.global_step // REFERENCE_STEPS
            if fixed:
                destination = path.parent / f'epoch{epoch}.ckpt'
                if not destination.exists():
                    os.link(path, destination)
        dist.barrier()

    def restore(self, path):
        saved = torch.load(path, map_location='cpu', pickle_module=dill)
        if saved['contract_hash'] != self.contract_hash or saved['world_size'] != self.world:
            raise ValueError('Equal-update resume contract/world size differs')
        self.ddp, self.stage = None, None
        restore_policy_state(self.policy, saved['model'], self.device)
        restore_policy_state(self.ema, saved['ema'], self.device)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.scheduler.load_state_dict(saved['scheduler'])
        self.global_step, self.data_epoch, self.cursor = saved['global_step'], saved['data_epoch'], saved['batch_cursor']
        self.stats, self.samples_seen = saved['stats'], saved['samples_seen']
        self.restore_rng(saved['rng_by_rank'][self.rank])
        self.set_stage(stage_for_update(self.global_step))
        self.assert_replicas()

    def batches(self):
        self.sampler.epoch, self.sampler.cursor = self.data_epoch, self.cursor
        if self.loader is None:
            options = dict(dataset=self.dataset, batch_sampler=self.sampler,
                num_workers=self.workers, pin_memory=True,
                worker_init_fn=initialize_ablation_worker,
                generator=torch.Generator().manual_seed(700042 + self.rank))
            if self.workers:
                options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
            self.loader = DataLoader(**options)
        yield from self.loader

    def update(self, batch, global_size):
        self.set_stage(stage_for_update(self.global_step))
        batch = {k: ({n:t.to(self.device, non_blocking=True) for n,t in v.items()}
                     if k == 'obs' else v.to(self.device, non_blocking=True)) for k,v in batch.items()}
        self.optimizer.zero_grad(set_to_none=True)
        action, utility = self.ddp(batch, self.stage)
        loss = weighted_ddp_loss(action, utility, self.stage, global_size, self.world)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite equal-update loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.policy.parameters() if p.requires_grad],
                                             1.0, error_if_nonfinite=True)
        self.optimizer.step(); self.scheduler.step(); update_ema(self.ema, self.policy, .9999)
        self.global_step += 1; self.cursor += 1; self.samples_seen += global_size
        stats = torch.zeros(3, 9, dtype=torch.float64, device=self.device)
        tasks = batch['task_index']
        stats[0].index_add_(0, tasks, torch.ones_like(action, dtype=torch.float64))
        stats[1].index_add_(0, tasks, action.detach().double())
        stats[2].index_add_(0, tasks, utility.detach().double())
        dist.all_reduce(stats)
        if int(stats[0].sum()) != global_size:
            raise RuntimeError('Distributed sample counts differ')
        self.stats += stats.cpu()
        return float(norm)

    def run(self, stop_after=0, monitor=None):
        initial = self.global_step
        window_started = time.perf_counter(); window_samples = 0
        while self.global_step < TOTAL_UPDATES:
            for batch in self.batches():
                if monitor is not None:
                    monitor.check()
                global_size = min(self.batch_size, len(self.dataset) - self.cursor * self.batch_size)
                norm = self.update(batch, global_size)
                window_samples += global_size
                if self.global_step % 20 == 0 or self.global_step == 1:
                    counts = self.stats[0].clamp_min(1)
                    self.log('train', samples_seen=self.samples_seen,
                        per_task_action_loss=(self.stats[1] / counts).tolist(),
                        per_task_utility_loss=(self.stats[2] / counts).tolist(),
                        task_counts=self.stats[0].tolist(), gradient_norm=norm,
                        samples_per_second=window_samples / (time.perf_counter()-window_started),
                        lr={g['name']:g['lr'] for g in self.optimizer.param_groups})
                    window_started = time.perf_counter(); window_samples = 0
                stop_file = os.environ.get('ADAVIP_STOP_FILE')
                stop = self.stop_requested or bool(stop_file and Path(stop_file).exists())
                stop |= bool(stop_after and self.global_step - initial >= stop_after)
                flag = torch.tensor(int(stop), device=self.device); dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                done = self.global_step >= TOTAL_UPDATES
                fixed = self.global_step in [epoch * REFERENCE_STEPS for epoch in FIXED_EPOCHS]
                if self.global_step == 1 or self.global_step % REFERENCE_STEPS == 0 or bool(flag) or done:
                    self.save(fixed=fixed)
                if done or bool(flag):
                    if self.rank == 0:
                        name = 'COMPLETE.json' if done else 'SAFE_STOP.json'
                        (self.output/name).write_text(json.dumps(dict(optimizer_updates=self.global_step,
                            samples_seen=self.samples_seen, reference_epoch=self.global_step/REFERENCE_STEPS))+'\n')
                    return
            self.data_epoch += 1; self.cursor = 0
