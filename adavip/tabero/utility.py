"""Resumable K64 native-loss labels for the selected Tabero DP Teacher."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adavip.advanced.common import atomic_json
from .data import TaberoDataset
from .model import TaberoL1Policy
from .schema import CONDITION_NAMES, fingerprint
from .train import move_batch

STRATA = ((0, 12), (12, 25), (25, 50), (50, 75), (75, 100))
COUNTS = (4, 3, 3, 3, 3)
WEIGHTS = np.array([.12, .13, .25, .25, .25])


def emit(output, event, **values):
    record = dict(time=time.strftime('%Y-%m-%d %H:%M:%S %z'), event=event, **values)
    line = json.dumps(record, allow_nan=False)
    print(line, flush=True)
    with (Path(output) / 'events.jsonl').open('a') as stream:
        stream.write(line + '\n')


def should_stop():
    name = os.environ.get('TABERO_UVIP_STOP_FILE')
    return bool(name and Path(name).exists())


def load_teacher(path, device='cpu'):
    """Require the user-selected E150 EMA and its original complete protocol."""
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if saved['epoch'] != 150 or saved['cursor'] != 0 or saved['global_step'] != 255750 or saved['ema_step'] != 255750:
        raise ValueError('Expected E150 EMA / 255750 updates at an epoch boundary')
    if fingerprint(saved['protocol']) != saved['protocol_id']:
        raise ValueError('Invalid Teacher protocol')
    policy = TaberoL1Policy(saved['config'])
    policy.load_state_dict(saved['ema'], strict=True)
    policy.eval().requires_grad_(False).to(device)
    stat = Path(path).stat()
    identity = dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    epoch=150, global_step=255750, weights='ema', protocol_id=saved['protocol_id'])
    return policy, identity


def make_draws(indices, horizon=16, seed=20261104):
    """Four independent K16 blocks, reproducible independently of batching/resume."""
    timesteps, noise = [], []
    for index in indices:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(index)]))
        timesteps.append(np.concatenate([np.concatenate([rng.integers(lo, hi, size=count, dtype=np.int64)
                         for (lo, hi), count in zip(STRATA, COUNTS)]) for _ in range(4)]))
        noise.append(rng.standard_normal((64, horizon, 13), dtype=np.float32))
    return np.stack(timesteps), np.stack(noise)


@torch.no_grad()
def paired_losses(teacher, batch, timesteps, noise, microbatch=128):
    """Reuse prefusion tokens; full/masked conditions share native noised actions."""
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Formal labels require a frozen eval Teacher')
    if microbatch < 1:
        raise ValueError('Denoising microbatch must be positive')
    count, draws = timesteps.shape
    if count != len(batch['action']) or noise.shape != (count, draws, teacher.horizon, 13):
        raise ValueError('Draw/batch shape mismatch')
    device = batch['action'].device
    steps = torch.as_tensor(timesteps.reshape(-1), device=device)
    eps = torch.as_tensor(noise.reshape(-1, teacher.horizon, 13), device=device)
    mask = batch['action_loss_mask'].repeat_interleave(draws, 0)
    clean = teacher.normalizer.normalize('action', batch['action']).masked_fill(~batch['action_loss_mask'].bool(), 0)
    noisy = teacher.noise_scheduler.add_noise(clean.repeat_interleave(draws, 0), eps, steps)
    tokens = teacher.obs_encoder.encode_tokens(batch['obs'], teacher.normalizer)
    result = np.empty((count, draws, 7), dtype=np.float32)
    for condition, presence in enumerate(teacher.condition_presence):
        context = teacher.obs_encoder.condition(tokens, batch['obs']['language'], presence[None].expand(count, -1))
        context = context.repeat_interleave(draws, 0)
        values = []
        for start in range(0, len(context), microbatch):
            stop = min(start + microbatch, len(context))
            prediction = teacher.model(noisy[start:stop], steps[start:stop], global_cond=context[start:stop])
            loss = teacher.loss_components((prediction.float() - eps[start:stop]).square(), mask[start:stop])['loss']
            values.append(loss.cpu().numpy())
        result[:, :, condition] = np.concatenate(values).reshape(count, draws)
    if not np.isfinite(result).all() or (result < 0).any():
        raise FloatingPointError('Invalid paired native losses')
    return result


def aggregate(losses, timesteps):
    """Preserve signed utility, analytic stratified variance and two K32 replicas."""
    losses = np.asarray(losses, dtype=np.float64)
    if losses.shape[1:] != (64, 7) or timesteps.shape != losses.shape[:2] or not np.isfinite(losses).all():
        raise ValueError('Expected finite [N,64,7] paired losses')
    means, variances = [], []
    for start, blocks in ((0, 4), (0, 2), (2, 2)):
        total, variance = np.zeros((len(losses), 7)), np.zeros((len(losses), 6))
        offset = 0
        for (low, high), count, weight in zip(STRATA, COUNTS, WEIGHTS):
            selected_steps = timesteps.reshape(-1, 4, 16)[:, start:start + blocks, offset:offset + count]
            if not ((selected_steps >= low) & (selected_steps < high)).all():
                raise ValueError('Invalid K64 stratum ordering')
            selected = losses.reshape(-1, 4, 16, 7)[:, start:start + blocks, offset:offset + count].reshape(len(losses), blocks * count, 7)
            differences = selected[:, :, 1:] - selected[:, :, :1]
            total += weight * selected.mean(1)
            variance += weight ** 2 * differences.var(1, ddof=1) / (blocks * count)
            offset += count
        means.append(total)
        variances.append(variance)
    return dict(utility=means[0][:, 1:] - means[0][:, :1], weighted_losses=means[0],
                utility_variance=variances[0], k32_utility=np.stack([m[:, 1:] - m[:, :1] for m in means[1:]], 1),
                k32_variance=np.stack(variances[1:], 1))


def shard_path(output, start, stop):
    return Path(output) / 'shards' / f'shard_{start:06d}_{stop:06d}.npz'


def read_shard(output, start, stop, identity):
    path = shard_path(output, start, stop)
    marker = path.with_suffix('.json')
    if not marker.exists():
        return None
    record = json.loads(marker.read_text())
    if record != dict(identity=identity, start=start, stop=stop, bytes=path.stat().st_size):
        raise ValueError('Committed label shard identity/size mismatch')
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(values['source_index'], np.arange(start, stop))
    if values['utility'].shape != (stop - start, 6) or not np.isfinite(values['losses']).all():
        raise ValueError('Invalid committed label shard')
    return values


def commit_shard(output, start, stop, identity, arrays):
    path = shard_path(output, start, stop)
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.npz.partial')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    atomic_json(path.with_suffix('.json'), dict(identity=identity, start=start, stop=stop, bytes=path.stat().st_size))


def preflight(teacher, dataset, microbatch):
    """Compare optimized scoring against native forwards, including episode padding."""
    indices = [0, min(len(dataset) - 1, dataset.episodes[0]['end'])]
    batch = move_batch(torch.utils.data.default_collate([dataset[i] for i in indices]), next(teacher.parameters()).device)
    steps, noise = make_draws(indices)
    actual = paired_losses(teacher, batch, steps[:, :2], noise[:, :2], microbatch)
    for draw in range(2):
        for condition in range(7):
            native = teacher.compute_losses(batch, torch.full((len(indices),), condition, device=batch['action'].device),
                        torch.as_tensor(noise[:, draw], device=batch['action'].device),
                        torch.as_tensor(steps[:, draw], device=batch['action'].device))['loss']
            np.testing.assert_allclose(actual[:, draw, condition], native.cpu().numpy(), rtol=5e-5, atol=1e-7)
    return dict(native_paired_loss_equal=True, padded_episode_boundary_checked=True,
                conditions=list(CONDITION_NAMES), no_trainable_teacher_parameters=True)


def generate(config, output, device='cuda:0', teacher=None, dataset=None):
    """Generate every native anchor once, with atomic shards and deterministic draws."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    teacher, identity = load_teacher(config['teacher'], device) if teacher is None else teacher
    dataset = dataset if dataset is not None else TaberoDataset(config['cache_dir'], preload_ram=config['preload_ram'])
    if dataset.manifest['cache_id'] != config['cache_id']:
        raise ValueError('Wrong source data cache')
    contract = dict(schema='uvip.tabero.dp.k64.v1', teacher=identity, cache_id=dataset.manifest['cache_id'],
                    windows=len(dataset), conditions=list(CONDITION_NAMES), k=64,
                    strata=STRATA, counts_per_k16=COUNTS, expectation_weights=WEIGHTS.tolist(), seed=config['label_seed'],
                    horizon=16, history=2, action_dim=13, force_weight=.1, shard_size=config['label_shard_size'],
                    window_selection='All native anchors; episode-bounded padding with native action/time masks',
                    normalization='Raw signed masked-minus-full; no centering or clipping', precision='strict fp32; TF32 disabled')
    contract = json.loads(json.dumps(contract))
    contract['identity'] = fingerprint(contract)
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('Label generation continuation contract changed')
    atomic_json(manifest, contract)
    if (output / 'COMPLETE.json').exists():
        return
    (output / 'shards').mkdir(exist_ok=True)
    np.savez(output / 'teacher_normalizer.npz', **{k: v.cpu().numpy() for k, v in teacher.normalizer.state_dict().items()})
    checked = preflight(teacher, dataset, config['denoise_microbatch'])
    atomic_json(output / 'preflight.json', checked)
    emit(output, 'preflight', **checked)
    completed = 0
    for start in range(0, len(dataset), config['label_shard_size']):
        stop = min(start + config['label_shard_size'], len(dataset))
        if read_shard(output, start, stop, contract['identity']) is None:
            break
        completed = stop
    loader = DataLoader(Subset(dataset, range(completed, len(dataset))), batch_size=config['label_window_batch'],
                        num_workers=config['num_workers'], pin_memory=str(device).startswith('cuda'),
                        generator=torch.Generator().manual_seed(700042),
                        **({'prefetch_factor': 1, 'multiprocessing_context': 'fork'} if config['num_workers'] else {}))
    iterator = iter(loader)
    started = time.monotonic()
    for start in range(completed, len(dataset), config['label_shard_size']):
        stop = min(start + config['label_shard_size'], len(dataset))
        timesteps, noise = make_draws(np.arange(start, stop), seed=config['label_seed'])
        losses = np.empty((stop - start, 64, 7), np.float32)
        cursor = start
        while cursor < stop:
            if should_stop():
                atomic_json(output / 'SAFE_STOP.json', dict(completed_windows=start))
                return
            batch = move_batch(next(iterator), torch.device(device))
            count = len(batch['action'])
            np.testing.assert_array_equal(batch['source_index'].cpu().numpy(), np.arange(cursor, cursor + count))
            if cursor + count > stop:
                raise ValueError('Shard size must be divisible by window batch')
            local = cursor - start
            losses[local:local + count] = paired_losses(teacher, batch, timesteps[local:local + count], noise[local:local + count], config['denoise_microbatch'])
            cursor += count
        arrays = dict(source_index=np.arange(start, stop), timesteps=timesteps, noise=noise, losses=losses,
                      **aggregate(losses, timesteps))
        commit_shard(output, start, stop, contract['identity'], arrays)
        emit(output, 'labels', done=stop, total=len(dataset), elapsed_seconds=time.monotonic() - started)
    finalize(output, contract, dataset)


def finalize(output, contract, dataset):
    """Publish complete aligned arrays and per-instruction training RMS scales."""
    names = dict(utility=(6,), utility_variance=(6,), weighted_losses=(7,), k32_utility=(2, 6), k32_variance=(2, 6))
    arrays = {key: np.lib.format.open_memmap(output / (key + '.npy.partial'), mode='w+', dtype=np.float64,
                                          shape=(len(dataset), *shape)) for key, shape in names.items()}
    for start in range(0, len(dataset), contract['shard_size']):
        stop = min(start + contract['shard_size'], len(dataset))
        saved = read_shard(output, start, stop, contract['identity'])
        if saved is None:
            raise ValueError('Cannot finalize incomplete labels')
        steps, noise = make_draws(np.arange(start, stop), seed=contract['seed'])
        np.testing.assert_array_equal(saved['timesteps'], steps)
        np.testing.assert_array_equal(saved['noise'], noise)
        reduced = aggregate(saved['losses'], saved['timesteps'])
        for key in arrays:
            np.testing.assert_array_equal(saved[key], reduced[key])
            arrays[key][start:stop] = saved[key]
    for key, array in arrays.items():
        if not np.isfinite(array).all():
            raise FloatingPointError('Invalid finalized label values')
        array.flush()
        os.replace(output / (key + '.npy.partial'), output / (key + '.npy'))
    utility = arrays['utility']
    np.testing.assert_allclose(utility, arrays['weighted_losses'][:, 1:] - arrays['weighted_losses'][:, :1], rtol=1e-12, atol=1e-14)
    if (arrays['utility_variance'] < 0).any():
        raise FloatingPointError('Invalid label variance')
    tasks = np.empty(len(dataset), dtype=np.int64)
    for episode in dataset.episodes:
        tasks[episode['start']:episode['end']] = episode['task_index']
    np.save(output / 'source_index.npy', np.arange(len(dataset)))
    np.save(output / 'task_index.npy', tasks)
    pooled = np.maximum(np.sqrt(np.square(utility).mean(0)), 1e-6)
    rms = np.stack([np.maximum(np.sqrt(np.square(utility[tasks == index]).mean(0)), 1e-6)
                    if np.any(tasks == index) else pooled for index in range(len(dataset.manifest['tasks']))]).astype(np.float32)
    np.save(output / 'utility_rms.npy', rms)
    atomic_json(output / 'COMPLETE.json', dict(identity=contract['identity'], windows=len(dataset), k=64,
        dimensions=list(CONDITION_NAMES[1:]), arrays={key: (output / (key + '.npy')).stat().st_size for key in names},
        all_native_anchors_exactly_once=True, raw_draws_reproduced=True,
        rms_method='Per native instruction index and utility dimension; no centering or clipping',
        negative_fraction=(utility < 0).mean(0).tolist()))
    emit(output, 'labels_complete', windows=len(dataset), identity=contract['identity'])
