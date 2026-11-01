"""Generate the full L1 E500 EMA K64 signed label store without training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset

from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
from adavip.manifeel.full_sensor_schema import CONDITION_NAMES
from adavip.manifeel.full_sensor_workspace import manifest_fingerprint, normalizer_fingerprint
from adavip.manifeel.l1_utility_labels import (
    ReferenceK64Draws, mask_conditionings, paired_k64_losses, reference_paired_losses, score_conditionings,
)
from adavip.manifeel.stage1_future_dataset import initialize_future_worker
from adavip.manifeel.utility_label_pilot import aggregate_losses
from adavip.manifeel.utility_label_store import (
    atomic_json, commit_shard, complete_window_index, file_hash, read_shard, shard_ranges,
)
from diffusion_policy.common.pytorch_util import dict_apply
from scripts.manifeel.language_runtime import configure_profile


def finalize(output: Path, contract: dict, index: np.ndarray, draws: ReferenceK64Draws) -> None:
    """Reconstruct every shard, recheck its draws, then publish loader-compatible arrays."""
    arrays = {name: np.lib.format.open_memmap(output/f'{name}.npy.partial', mode='w+',
              dtype=np.float64, shape=(len(index), size)) for name, size in
              (('utility', 9), ('utility_variance', 9), ('weighted_losses', 10))}
    shards = {}
    for start, stop in shard_ranges(len(index), contract['chunk_size']):
        saved = read_shard(output/'shards', start, stop, contract['sha256'], index[start:stop])
        if saved is None:
            raise RuntimeError('Cannot finalize incomplete label coverage')
        steps, noise = draws.get(index['window_id'][start:stop])
        np.testing.assert_array_equal(saved['timesteps'], steps)
        np.testing.assert_array_equal(saved['noise'], noise)
        for name, array in arrays.items():
            array[start:stop] = saved[name]
        marker = output/'shards'/f'shard_{start:06d}_{stop:06d}.json'
        shards[marker.name] = json.loads(marker.read_text())
    for name, array in arrays.items():
        array.flush()
        os.replace(output/f'{name}.npy.partial', output/f'{name}.npy')
    utility, variance = arrays['utility'], arrays['utility_variance']
    if not np.isfinite(utility).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise FloatingPointError('Invalid final utility labels')
    np.testing.assert_allclose(utility, arrays['weighted_losses'][:, 1:] -
                               arrays['weighted_losses'][:, :1], rtol=1e-12, atol=1e-14)
    rows = []
    for task in [-1, *range(9)]:
        selected = np.ones(len(index), dtype=bool) if task == -1 else index['task_index'] == task
        for dimension, name in enumerate(CONDITION_NAMES[1:]):
            values = utility[selected, dimension]
            rows.append({'task_index': task, 'dimension': name, 'windows': len(values),
                         'mean': float(values.mean()), 'std': float(values.std()),
                         'negative_fraction': float((values < 0).mean()),
                         'analytic_noise_mse': float(variance[selected, dimension].mean())})
    atomic_json(output/'label_summary.json', {'rows': rows})
    names = ('window_index.npy', 'task_id.npy', 'utility.npy', 'utility_variance.npy',
             'weighted_losses.npy', 'label_summary.json', 'metadata/teacher_normalizer.npz')
    files = {name: {'bytes': (output/name).stat().st_size, 'sha256': file_hash(output/name)} for name in names}
    atomic_json(output/'COMPLETE.json', {
        'status': 'complete', 'schema': contract['schema'], 'windows': len(index),
        'contract_sha256': contract['sha256'], 'files': files, 'shards': len(shards),
        'shard_records': shards, 'all_windows_exactly_once': True,
        'raw_draws_reproduced': True, 'weighted_labels_reconstructed': True,
        'finite_labels_and_nonnegative_variances': True, 'student_developed_or_trained': False})


def main() -> None:
    """Run a bounded GPU preflight followed by the complete resumable label job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if (cfg.shard_size < 1 or cfg.window_batch_size < 1
            or cfg.shard_size % cfg.window_batch_size or cfg.denoise_batch_size < 1):
        raise ValueError('Shard size must be divisible by the positive window batch size')
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'COMPLETE.json').exists():
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or 'H200' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Full L1 labels require one independently allocated H200')
    stopped = []
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda number, frame: stopped.append(number))
    if cfg.runtime_profile != 'strict':
        raise ValueError('Formal K64 labels retain strict FP32 without TF32 or AMP')
    runtime = configure_profile('strict')
    torch.use_deterministic_algorithms(True)
    checkpoint = Path(cfg.checkpoint).resolve()
    policy, model_cfg, metadata = load_evaluation_policy(checkpoint, 'cuda:0', cfg.weights)
    if (metadata['epoch'] != cfg.expected_epoch or metadata['global_step'] != cfg.expected_step
            or metadata['weights'] != 'ema' or model_cfg.variant != 'l1'
            or policy.horizon != 16 or policy.n_obs_steps != 2
            or policy.noise_scheduler.config.num_train_timesteps != 100):
        raise ValueError('Expected the original L1 E500 EMA, not a dropout/Future checkpoint')
    normalizer = normalizer_fingerprint(policy.normalizer)
    previous = json.loads((Path(cfg.previous_diagnostic)/'protocol.json').read_text())
    if previous['checkpoint'] != str(checkpoint) or previous['normalizer'] != normalizer:
        raise ValueError("Checkpoint/normalizer differs from yesterday's L1 diagnostic")
    dataset_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg.dataset, resolve=True))
    dataset_cfg._target_ = 'adavip.manifeel.language_conditioning_dataset.LanguageConditioningDataset'
    dataset = hydra.utils.instantiate(dataset_cfg)
    index = complete_window_index(dataset.manifest)
    reference = Path(cfg.reference_archive)
    reference_manifest = json.loads((reference/'manifest.json').read_text())
    if len(dataset) != cfg.expected_samples or len(index) != cfg.expected_windows:
        raise ValueError('Unexpected anchor or complete-window count')
    np.testing.assert_array_equal(index, np.load(reference/'window_index.npy'))
    task_ids = np.asarray([spec.task_id for spec in dataset.task_specs])[index['task_index']]
    np.testing.assert_array_equal(task_ids, np.load(reference/'task_id.npy'))
    sources = ('adavip/manifeel/l1_utility_labels.py', 'adavip/manifeel/language_conditioned_obs_encoder.py',
               'adavip/manifeel/reference_diffusion_policy.py', 'adavip/manifeel/full_sensor_evaluation.py',
               'adavip/manifeel/full_sensor_dataset.py', 'adavip/manifeel/full_sensor_schema.py',
               'adavip/manifeel/full_sensor_workspace.py', 'adavip/manifeel/language_conditioning_dataset.py',
               'adavip/manifeel/decoded_sensor_cache.py', 'adavip/manifeel/stage1_future_dataset.py',
               'adavip/manifeel/utility_label_pilot.py', 'adavip/manifeel/utility_k64_precision.py',
               'adavip/manifeel/utility_label_store.py', 'adavip/manifeel/task_protocol.py',
               'adavip/manifeel/task_embeddings.py', 'scripts/manifeel/generate_l1_k64_labels.py')
    root = Path(__file__).resolve().parents[2]
    contract = {'schema': 'adavip.utility_labels.v1', 'windows': len(index),
                'chunk_size': int(cfg.shard_size), 'dimensions': list(CONDITION_NAMES[1:]),
                'conditions': list(CONDITION_NAMES), 'dataset_manifest': dataset.manifest,
                'tasks': reference_manifest['tasks'], 'protocol': reference_manifest['protocol'],
                'draws': reference_manifest['draws'],
                'checkpoint': {'path': str(checkpoint), 'bytes': checkpoint.stat().st_size,
                               'mtime_ns': checkpoint.stat().st_mtime_ns},
                'teacher': {'metadata': metadata, 'normalizer_sha256': normalizer,
                            'policy_config': OmegaConf.to_container(model_cfg.policy, resolve=True)},
                'runtime': {**runtime, 'dtype': 'float32', 'numpy': np.__version__, 'torch': torch.__version__,
                            'window_batch_size': int(cfg.window_batch_size),
                            'denoise_batch_size': int(cfg.denoise_batch_size)},
                'reference_archive': str(reference),
                'reference_manifest_sha256': file_hash(reference/'manifest.json'),
                'source_files': {name: file_hash(root/name) for name in sources},
                'definition': 'signed normalized-action masked denoising MSE minus paired full MSE',
                'mask_semantics': 'L1 zero post-projection sensor+identity tokens, six presence flags, zero late language',
                'validity': 'complete history2/action horizon16; absent action dimension7 masked',
                'label_normalization': 'none; raw signed utility; no clipping or fitting',
                'scope': 'full label generation only; no model/optimizer/Stage2 training'}
    contract['sha256'] = manifest_fingerprint(contract)
    if (output/'manifest.json').exists():
        if json.loads((output/'manifest.json').read_text()) != contract:
            raise ValueError('Label continuation contract differs')
    else:
        atomic_json(output/'manifest.json', contract)
        np.save(output/'window_index.npy', index)
        np.save(output/'task_id.npy', task_ids)
    (output/'metadata').mkdir(exist_ok=True)
    (output/'shards').mkdir(exist_ok=True)
    with (output/'metadata/teacher_normalizer.npz').open('wb') as stream:
        np.savez_compressed(stream, **{name: tensor.cpu().numpy() for name, tensor in policy.normalizer.state_dict().items()})
    OmegaConf.save(cfg, output/'config.yaml', resolve=True)
    draws = ReferenceK64Draws(reference)
    print(json.dumps({'event': 'label_workspace_ready', 'windows': len(index), 'teacher': metadata,
                      'normalizer': normalizer, 'runtime': contract['runtime']}), flush=True)
    started = time.perf_counter()
    dataset.load_ram_cache()
    print(json.dumps({'event': 'full_ram_ready', 'seconds': time.perf_counter()-started}), flush=True)
    # Preflight uses real complete windows across tasks, independent of the formal shard cursor.
    probe_ids = [int(np.flatnonzero(index['task_index'] == task)[0]) for task in (0, 2, 4, 8)]
    probe = torch.utils.data.default_collate([dataset[int(index[i]['dataset_index'])] for i in probe_ids])
    probe = dict_apply(probe, lambda tensor: tensor.to('cuda:0'))
    steps, noise = draws.get(np.asarray(probe_ids))
    conditions = mask_conditionings(policy, probe['obs'])
    expected = reference_paired_losses(policy, probe, conditions, noise, steps)
    actual = score_conditionings(policy, probe, conditions, noise, steps, cfg.denoise_batch_size)
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-7)
    # Check the inherited pilot/deterministic streams against persisted legacy raw draws.
    with np.load(reference/'shards/shard_000000_000256.npz') as archive:
        reference_steps, reference_noise = draws.get(np.arange(8))
        np.testing.assert_array_equal(reference_steps, archive['timesteps'][:8])
        np.testing.assert_array_equal(reference_noise, archive['noise'][:8])
    assert all(not p.requires_grad and p.grad is None for p in policy.parameters())
    assert dataset._normalizer is None
    assert normalizer_fingerprint(policy.normalizer) == normalizer
    assert all(p.device.type == 'cuda' for p in policy.normalizer.parameters())
    atomic_json(output/'preflight.json', {'passed': True, 'teacher': metadata,
                'probe_window_ids': probe_ids, 'draws_per_window_checked': 64,
                'paired_loss_max_abs_error': float(np.abs(actual-expected).max()),
                'legacy_draws_exact': True, 'normalizer_unchanged': normalizer_fingerprint(policy.normalizer)==normalizer,
                'trainable_parameters': 0, 'gpu': torch.cuda.get_device_name(0),
                'trainer_affinity': sorted(os.sched_getaffinity(0))})
    del probe, conditions, steps, noise, actual, expected
    ranges = shard_ranges(len(index), cfg.shard_size)
    completed = 0
    for start, stop in ranges:
        if read_shard(output/'shards', start, stop, contract['sha256'], index[start:stop]) is None:
            break
        completed = stop
    if any((output/'shards'/f'shard_{start:06d}_{stop:06d}.json').exists()
           for start, stop in ranges if start >= completed):
        raise ValueError('Unexpected non-contiguous committed shards')
    loader = DataLoader(Subset(dataset, index['dataset_index'][completed:].tolist()),
                        batch_size=cfg.window_batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, persistent_workers=True, prefetch_factor=1,
                        worker_init_fn=initialize_future_worker,
                        generator=torch.Generator().manual_seed(700042))
    iterator = iter(loader)
    generation_started, session_windows, new_shards = time.perf_counter(), 0, 0
    try:
        for start, stop in ranges:
            if stop <= completed:
                continue
            steps, noise = draws.get(index['window_id'][start:stop])
            losses = np.empty((stop-start, 64, 10), dtype=np.float32)
            position = start
            while position < stop:
                stop_file = os.environ.get('ADAVIP_STOP_FILE')
                if stopped or (stop_file and Path(stop_file).exists()):
                    atomic_json(output/'SAFE_STOP.json', {'completed_windows': completed})
                    return
                cpu_batch = next(iterator)
                size = len(cpu_batch['action'])
                assert position+size <= stop
                np.testing.assert_array_equal(cpu_batch['source_index'].numpy(), index['source_index'][position:position+size])
                assert cpu_batch['observation_valid_mask'].all() and cpu_batch['action_valid_mask'].all()
                batch = dict_apply(cpu_batch, lambda tensor: tensor.to('cuda:0', non_blocking=True))
                local = position-start
                losses[local:local+size] = paired_k64_losses(policy, batch,
                    noise[local:local+size], steps[local:local+size], cfg.denoise_batch_size)
                position += size
            arrays = {'window_index': index[start:stop].copy(), 'losses': losses, 'noise': noise,
                      'timesteps': steps, 'raw_signed_utility': losses[:, :, 1:]-losses[:, :, :1],
                      **aggregate_losses(losses, steps)}
            commit_shard(output/'shards', start, stop, contract['sha256'], arrays)
            completed, new_shards = stop, new_shards+1
            session_windows += stop-start
            elapsed = time.perf_counter()-generation_started
            record = {'event': 'shard_committed', 'completed_windows': completed, 'windows': len(index),
                      'start': start, 'stop': stop, 'seconds': elapsed, 'windows_per_second': session_windows/elapsed,
                      'estimated_remaining_hours': (len(index)-completed)*elapsed/max(session_windows, 1)/3600,
                      'peak_cuda_gib': torch.cuda.max_memory_allocated()/1024**3}
            atomic_json(output/'progress.json', record)
            print(json.dumps(record), flush=True)
            if cfg.stop_after_shards and new_shards >= cfg.stop_after_shards:
                atomic_json(output/'SAFE_STOP.json', {'completed_windows': completed, 'reason': 'bounded check'})
                return
    finally:
        iterator._shutdown_workers()
    if normalizer_fingerprint(policy.normalizer) != normalizer or dataset._normalizer is not None:
        raise ValueError('Teacher normalizer changed or was fitted from the dataset')
    finalize(output, contract, index, draws)
    print(json.dumps({'event': 'full_l1_k64_complete', 'windows': len(index), 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
