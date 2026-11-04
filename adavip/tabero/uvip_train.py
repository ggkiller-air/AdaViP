"""Train the Tabero DP S0 Student after complete E150 EMA K64 labels."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from adavip.advanced.common import atomic_json
from .data import TaberoDataset
from .schema import fingerprint
from .train import atomic_save, capture_rng, restore_rng, ModelEMA, move_batch
from .utility import emit, generate, load_teacher, should_stop
from .uvip import TaberoUViPPolicy


class LabeledDataset(torch.utils.data.Dataset):
    """Join labels by the exact native source and instruction indices."""

    def __init__(self, source, labels, teacher_identity):
        self.source = source
        labels = Path(labels)
        self.manifest = json.loads((labels / 'manifest.json').read_text())
        complete = json.loads((labels / 'COMPLETE.json').read_text())
        if (self.manifest['identity'] != complete['identity'] or self.manifest['teacher'] != teacher_identity
                or self.manifest['cache_id'] != source.manifest['cache_id'] or complete['windows'] != len(source)):
            raise ValueError('Teacher/data/label identity mismatch')
        if fingerprint({k: v for k, v in self.manifest.items() if k != 'identity'}) != complete['identity']:
            raise ValueError('Invalid label contract')
        self.utility = np.load(labels / 'utility.npy')
        self.rms = np.load(labels / 'utility_rms.npy')
        np.testing.assert_array_equal(np.load(labels / 'source_index.npy'), np.arange(len(source)))
        tasks = np.load(labels / 'task_index.npy')
        for episode in source.episodes:
            if not np.all(tasks[episode['start']:episode['end']] == episode['task_index']):
                raise ValueError('Label/source instruction index mismatch')
        if self.utility.shape != (len(source), 6) or not np.isfinite(self.utility).all():
            raise ValueError('Invalid signed utility values')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        item = self.source[index]
        item['utility'] = torch.as_tensor(self.utility[index].astype(np.float32))
        return item


def phase_for_epoch(epoch):
    return 'A' if epoch == 0 else 'B' if epoch < 6 else 'C'


def validate_gradient_routes(policy, batch):
    """Check actual native B/C backward while preserving all random streams."""
    saved_rng = capture_rng()
    saved_phase = policy.phase
    reports = {}
    try:
        for phase in ('B', 'C'):
            policy.set_phase(phase)
            policy.zero_grad(set_to_none=True)
            result = policy.training_losses(batch, phase)
            result['utility_loss'].backward(retain_graph=True)
            if not any(p.grad is not None for p in policy.utility_predictor.parameters()):
                raise RuntimeError('Utility predictor lacks its own gradient')
            if any(p.grad is not None for name, p in policy.named_parameters() if not name.startswith('utility_predictor.')):
                raise RuntimeError('Utility loss escaped its detached context')
            policy.zero_grad(set_to_none=True)
            result['action_loss'].backward()
            if any(p.grad is not None for p in policy.utility_predictor.parameters()):
                raise RuntimeError('Action gradient reached Utility predictor')
            for name in ('images', 'tcn', 'state', 'language_head'):
                module = getattr(policy.obs_encoder, name)
                active = any(p.grad is not None and bool(p.grad.count_nonzero()) for p in module.parameters())
                if active != (phase == 'C'):
                    raise RuntimeError(f'Incorrect encoder route: {phase}/{name}')
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
            reports[phase] = dict(action_loss=float(result['action_loss'].detach()),
                                 utility_loss=float(result['utility_loss'].detach()),
                                 trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad))
    finally:
        policy.zero_grad(set_to_none=True)
        policy.set_phase(saved_phase)
        restore_rng(saved_rng)
    return reports


def worker_init(worker_id):
    torch.set_num_threads(1)
    cores = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cores[worker_id % len(cores)]})


@torch.no_grad()
def verify_teacher_context(policy, teacher, observations):
    """Require exact inherited tensors, with FP32 tolerance only for activations."""
    for module_name in ('obs_encoder', 'normalizer'):
        expected_state = getattr(teacher, module_name).state_dict()
        actual_state = getattr(policy, module_name).state_dict()
        if expected_state.keys() != actual_state.keys():
            raise ValueError('Inherited state keys differ: ' + module_name)
        for key, value in expected_state.items():
            torch.testing.assert_close(actual_state[key], value, rtol=0, atol=0)
    tokens = teacher.obs_encoder.encode_tokens(observations, teacher.normalizer)
    expected = teacher.obs_encoder.condition(tokens, observations['language'],
                teacher.condition_presence[:1].expand(len(tokens), -1))
    actual = policy.encode_stage2(observations)['z_t']
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return dict(inherited_encoder_and_normalizer_exact=True, activation_rtol=1e-5,
                activation_atol=1e-6, max_absolute_error=float((actual - expected).abs().max()))


class Trainer:
    """Persist model/EMA/Adam, schedule, phase, data cursor and RNG exactly."""

    def __init__(self, config, output, policy, dataset, teacher_identity, device='cuda:0'):
        self.config, self.output = copy.deepcopy(config), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.dataset = dataset
        self.policy = policy.to(self.device).train()
        self.ema = ModelEMA(self.policy)
        self.steps_per_epoch = math.ceil(len(dataset) / config['batch_size'])
        self.total_steps = config['num_epochs'] * self.steps_per_epoch
        encoder = list(policy.obs_encoder.parameters())
        utility = list(policy.utility_predictor.parameters())
        excluded = {id(p) for p in encoder + utility}
        action = [p for p in policy.parameters() if id(p) not in excluded]
        self.optimizer = torch.optim.AdamW([
            dict(params=encoder, name='encoder', lr=config['encoder_lr']),
            dict(params=utility, name='utility', lr=config['utility_lr']),
            dict(params=action, name='action', lr=config['action_lr'])],
            betas=tuple(config['betas']), weight_decay=config['weight_decay'],
            fused=self.device.type == 'cuda')
        self.contract = dict(config=config, teacher=teacher_identity, label_identity=dataset.manifest['identity'],
                             cache_id=dataset.source.manifest['cache_id'], windows=len(dataset),
                             steps_per_epoch=self.steps_per_epoch, total_steps=self.total_steps,
                             architecture='tabero_l1_s0_uvip_v1', action_unet='fresh',
                             phases={'A': 'E1', 'B': 'E2-E6', 'C': 'E7-E200; inherited encoders unfrozen'})
        self.identity = fingerprint(self.contract)
        self.epoch = self.cursor = self.global_step = 0
        self.routes_checked = False
        self.epoch_sums = dict(action=0., utility=0., position=0., force=0., samples=0)
        self.normalizer = {name: value.detach().clone() for name, value in self.policy.normalizer.state_dict().items()}

    def save(self, snapshot=False):
        for name, value in self.normalizer.items():
            torch.testing.assert_close(self.policy.normalizer.state_dict()[name], value, rtol=0, atol=0)
        payload = dict(config=self.config, contract=self.contract, identity=self.identity,
                       policy=self.policy.state_dict(), ema=self.ema.model.state_dict(), ema_step=self.ema.optimization_step,
                       optimizer=self.optimizer.state_dict(), epoch=self.epoch, cursor=self.cursor,
                       global_step=self.global_step, phase=phase_for_epoch(min(self.epoch, self.config['num_epochs'] - 1)),
                       rng=capture_rng(), epoch_sums=self.epoch_sums,
                       schedule=dict(total_steps=self.total_steps, completed_steps=self.global_step))
        latest = self.output / 'checkpoints/latest.ckpt'
        atomic_save(payload, latest)
        if snapshot:
            retained = latest.parent / f'epoch{self.epoch:04d}.ckpt'
            if retained.exists():
                raise FileExistsError(retained)
            os.link(latest, retained)
        if self.global_step == 1 and self.config.get('archive'):
            import shutil
            archive = Path(self.config['archive'])
            archive.mkdir(parents=True, exist_ok=True)
            target = archive / 'first_update.ckpt'
            if not target.exists():
                shutil.copyfile(latest, target.with_suffix('.partial'))
                os.replace(target.with_suffix('.partial'), target)

    def resume(self, path):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['identity'] != self.identity or fingerprint(saved['contract']) != self.identity:
            raise ValueError('Training resume contract changed')
        if saved['global_step'] != saved['epoch'] * self.steps_per_epoch + saved['cursor'] or saved['ema_step'] != saved['global_step']:
            raise ValueError('Invalid persisted step/cursor/EMA identity')
        if saved['schedule'] != dict(total_steps=self.total_steps, completed_steps=saved['global_step']):
            raise ValueError('Training schedule changed')
        self.policy.load_state_dict(saved['policy'], strict=True)
        self.ema.model.load_state_dict(saved['ema'], strict=True)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.epoch, self.cursor, self.global_step = saved['epoch'], saved['cursor'], saved['global_step']
        self.epoch_sums = saved['epoch_sums']
        self.ema.optimization_step = saved['ema_step']
        restore_rng(saved['rng'])

    def set_learning_rates(self):
        progress = max(0., (self.global_step - self.config['warmup_steps']) / max(1, self.total_steps - self.config['warmup_steps']))
        decay = .5 * (1 + math.cos(math.pi * min(progress, 1.)))
        for group in self.optimizer.param_groups:
            name = group['name']
            start = (6 * self.steps_per_epoch if name == 'encoder' else self.steps_per_epoch if name == 'action' else 0)
            warmup = min(1., max(0., (self.global_step - start + 1) / self.config['warmup_steps']))
            group['lr'] = self.config[name + '_lr'] * warmup * decay

    def run(self, stop_after_steps=0):
        atomic_json(self.output / 'training_contract.json', dict(identity=self.identity, **self.contract))
        while self.epoch < self.config['num_epochs']:
            phase = phase_for_epoch(self.epoch)
            self.policy.set_phase(phase)
            permutation = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.config['seed'] + self.epoch)).tolist()
            loader = DataLoader(self.dataset, sampler=permutation[self.cursor * self.config['batch_size']:],
                                batch_size=self.config['batch_size'], num_workers=self.config['num_workers'],
                                pin_memory=self.device.type == 'cuda', generator=torch.Generator().manual_seed(700042 + self.epoch),
                                worker_init_fn=worker_init,
                                **({'prefetch_factor': 1, 'multiprocessing_context': 'fork'} if self.config['num_workers'] else {}))
            emit(self.output, 'phase', phase=phase, epoch=self.epoch + 1, global_step=self.global_step)
            started, measured_samples = time.monotonic(), 0
            for batch in loader:
                if should_stop() or (stop_after_steps and self.global_step >= stop_after_steps):
                    self.save()
                    atomic_json(self.output / 'SAFE_STOP.json', dict(epoch=self.epoch, cursor=self.cursor, global_step=self.global_step))
                    return
                batch = move_batch(batch, self.device)
                if not self.routes_checked:
                    routes = validate_gradient_routes(self.policy, batch)
                    atomic_json(self.output / 'gradient_routes.json', routes)
                    self.routes_checked = True
                self.set_learning_rates()
                self.optimizer.zero_grad(set_to_none=True)
                result = self.policy.training_losses(batch, phase)
                loss = result['utility_loss'] if phase == 'A' else result['action_loss'] + self.config['utility_weight'] * result['utility_loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite native loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['gradient_clip_norm'], error_if_nonfinite=True)
                self.optimizer.step()
                self.ema.step(self.policy)
                self.cursor += 1
                self.global_step += 1
                count = len(batch['action'])
                measured_samples += count
                self.epoch_sums['samples'] += count
                for key in ('action', 'utility', 'position', 'force'):
                    self.epoch_sums[key] += float(result[key + '_loss'].detach()) * count
                if self.global_step == 1 or self.global_step % self.config['checkpoint_every_steps'] == 0:
                    self.save()
                if self.global_step % 20 == 0:
                    emit(self.output, 'train', epoch=self.epoch + 1, phase=phase, global_step=self.global_step,
                         total_steps=self.total_steps, action_loss=float(result['action_loss'].detach()),
                         utility_loss=float(result['utility_loss'].detach()), gradient_norm=float(norm),
                         samples_per_second=measured_samples / max(time.monotonic() - started, 1e-6))
                if self.global_step == 100:
                    atomic_json(self.output / 'first100_updates.json', dict(finite=True, batch_size=self.config['batch_size'],
                                samples_per_second=measured_samples / max(time.monotonic() - started, 1e-6),
                                affinity=sorted(os.sched_getaffinity(0))))
                if self.epoch == 1 and self.cursor == 100:
                    throughput = measured_samples / max(time.monotonic() - started, 1e-6)
                    atomic_json(self.output / 'B_first100_updates.json', dict(finite=True, batch_size=self.config['batch_size'],
                                samples_per_second=throughput, affinity=sorted(os.sched_getaffinity(0))))
                    if throughput < self.config.get('min_samples_per_second', 0):
                        self.save()
                        raise RuntimeError('B-stage first100-update throughput gate failed; inspect allocation/worker affinities')
            self.epoch += 1
            self.cursor = 0
            emit(self.output, 'epoch_completed', epoch=self.epoch, global_step=self.global_step, phase=phase,
                 **{key + '_loss': self.epoch_sums[key] / self.epoch_sums['samples'] for key in ('action', 'utility', 'position', 'force')})
            self.epoch_sums = dict(action=0., utility=0., position=0., force=0., samples=0)
            self.save(snapshot=self.epoch in self.config['checkpoint_epochs'])
        atomic_json(self.output / 'COMPLETE.json', dict(epoch=self.epoch, global_step=self.global_step, identity=self.identity))
        emit(self.output, 'train_complete', epoch=self.epoch, global_step=self.global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    for signum in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(signum, lambda *_: Path(os.environ['TABERO_UVIP_STOP_FILE']).touch())
    torch.set_num_threads(4)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Formal labels/training require exactly one allocated H200')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    teacher, teacher_identity = load_teacher(config['teacher'], 'cuda:0')
    source = TaberoDataset(config['cache_dir'], preload_ram=config['preload_ram'])
    normalizer_archive = root / 'labels/teacher_normalizer.npz'
    if normalizer_archive.exists():
        with np.load(normalizer_archive) as archive:
            for key, value in teacher.normalizer.state_dict().items():
                np.testing.assert_array_equal(value.cpu().numpy(), archive[key])
    generate(config, root / 'labels', teacher=(teacher, teacher_identity), dataset=source)
    if should_stop() or args.labels_only:
        return
    dataset = LabeledDataset(source, root / 'labels', teacher_identity)
    # Stage 2 initialization is independent of label job interruptions.
    torch.manual_seed(config['seed'])
    policy_config = {**teacher.config, **config, 'architecture': 'tabero_l1_s0_uvip_v1'}
    policy = TaberoUViPPolicy(policy_config, dataset.rms)
    policy.initialize_from_teacher(teacher.to('cpu'))
    probe = torch.utils.data.default_collate([dataset[0], dataset[len(dataset) - 1]])
    initialization = verify_teacher_context(policy, teacher, probe['obs'])
    atomic_json(root / 'stage2_initialization_verified.json', initialization)
    del teacher, probe
    gc.collect()
    torch.cuda.empty_cache()
    trainer = Trainer(policy_config, root / 'train', policy, dataset, teacher_identity)
    latest = root / 'train/checkpoints/latest.ckpt'
    if latest.exists():
        trainer.resume(latest)
    trainer.run()


if __name__ == '__main__':
    main()
