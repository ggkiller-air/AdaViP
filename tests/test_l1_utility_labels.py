"""Verify draw reuse and denoising-batch pairing independently of GPU kernels."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from test_language_conditioning import data_args

from adavip.manifeel.l1_utility_labels import (
    ReferenceK64Draws, mask_conditionings, reference_paired_losses, score_conditionings,
)
from adavip.manifeel.utility_label_pilot import aggregate_losses
from adavip.manifeel.utility_label_store import window_draws


class AlgebraicPolicy:
    """Expose the production sample/loss interface with checkable action-dimension masking."""

    horizon = 16

    def prepare_denoising_sample(self, batch, noise, timesteps):
        return SimpleNamespace(action=batch['action'], mask=batch['action_loss_mask'],
                               noise=noise, timesteps=timesteps)

    def denoising_loss_per_sample(self, sample, condition):
        prediction = sample.action + sample.noise * condition[:, :1, None] + sample.timesteps[:, None, None]*0.001
        return (prediction.square()*sample.mask).sum((1, 2))/sample.mask.sum((1, 2))


@pytest.mark.parametrize('denoise_batch', [1, 3, 17, 64, 256, 1024])
def test_batched_scoring_preserves_windows_draws_masks_and_signed_reduction(denoise_batch):
    policy = AlgebraicPolicy()
    generator = torch.Generator().manual_seed(42)
    actions = torch.randn(3, 16, 7, generator=generator)
    masks = torch.ones_like(actions)
    masks[0, :, 6] = 0
    noise = torch.randn(3, 64, 16, 7, generator=generator).numpy()
    timesteps = np.stack([window_draws(i, 20261065, 20261066)[0] for i in range(3)])
    conditions = torch.arange(10).float()[:, None, None].expand(10, 3, 1)*0.1
    conditions[1] = 0  # This removal improves the synthetic prediction; preserve negative utility.
    conditions[0] = 0.1
    batch = dict(action=actions, action_loss_mask=masks)
    reference = reference_paired_losses(policy, batch, conditions, noise, timesteps)
    actual = score_conditionings(policy, batch, conditions, noise, timesteps, denoise_batch)
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)
    estimates = aggregate_losses(actual, timesteps)
    assert estimates['utility'].shape == (3, 9)
    assert (estimates['utility'][:, 0] < 0).all()
    assert (estimates['utility_variance'] >= 0).all()
    np.testing.assert_allclose(estimates['utility'], estimates['weighted_losses'][:, 1:]-
                               estimates['weighted_losses'][:, :1], rtol=0, atol=0)


def test_reference_draws_reuse_pilot_and_ignore_batch_boundaries(tmp_path: Path):
    (tmp_path/'metadata').mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps({'draws': {'timestep_seed': 20261065, 'noise_seed': 20261066}}))
    np.save(tmp_path/'metadata/pilot_reuse_rows.npy', np.array([-1, 0, -1]))
    np.save(tmp_path/'metadata/pilot_timesteps.npy', np.full((1, 64), 7, dtype=np.int64))
    np.save(tmp_path/'metadata/pilot_noise.npy', np.ones((1, 64, 16, 7), dtype=np.float32))
    draws = ReferenceK64Draws(tmp_path)
    whole_steps, whole_noise = draws.get(np.arange(3))
    for index in range(3):
        steps, noise = draws.get(np.array([index]))
        np.testing.assert_array_equal(steps[0], whole_steps[index])
        np.testing.assert_array_equal(noise[0], whole_noise[index])
    assert (whole_steps[1] == 7).all() and (whole_noise[1] == 1).all()
    steps, seed = window_draws(2, 20261065, 20261066)
    np.testing.assert_array_equal(whole_steps[2], steps)
    np.testing.assert_array_equal(whole_noise[2], torch.randn(64, 16, 7, generator=torch.Generator().manual_seed(seed)))
    with pytest.raises(IndexError):
        draws.get(np.array([-1]))


def test_l1_masks_preserve_native_full_path_and_remove_input_influence(tmp_path: Path, data_args):
    from test_language_conditioning import small_config
    from adavip.manifeel.language_conditioning_dataset import LanguageConditioningDataset
    from adavip.manifeel.language_conditioning_workspace import LanguageConditioningWorkspace
    dataset = LanguageConditioningDataset(**data_args)
    workspace = LanguageConditioningWorkspace(small_config(data_args, tmp_path/'mask'))
    policy = workspace.model.eval()
    policy.set_normalizer(dataset.get_normalizer())
    observations = torch.utils.data.default_collate([dataset[3], dataset[36]])['obs']
    original = mask_conditionings(policy, observations)
    changed = {k: v.clone() for k, v in observations.items()}
    for key in ('front', 'side', 'wrist'):
        changed[key].zero_()
    cameras = mask_conditionings(policy, changed)
    torch.testing.assert_close(original[8], cameras[8], rtol=0, atol=0)
    assert not torch.equal(original[0], cameras[0])
    changed = {**observations, 'task_embedding': torch.zeros_like(observations['task_embedding'])}
    language = mask_conditionings(policy, changed)
    torch.testing.assert_close(original[7], language[7], rtol=0, atol=0)
    assert not torch.equal(original[0], language[0])
    assert all(p.grad is None for p in policy.parameters())


def test_final_store_matches_existing_reader_and_preserves_every_signed_label(tmp_path: Path):
    from adavip.manifeel.full_sensor_workspace import manifest_fingerprint
    from adavip.manifeel.stage1_dataset import UtilityLabelStore
    from adavip.manifeel.utility_label_store import commit_shard, complete_window_index, shard_ranges
    from scripts.manifeel.generate_l1_k64_labels import finalize
    manifest = {'horizon': 16, 'n_obs_steps': 2, 'stores': [
        {'episode_ends': [16, 32], 'train_episodes': [0, 1], 'val_episodes': []} for _ in range(9)]}
    index = complete_window_index(manifest)
    contract = {'schema': 'adavip.utility_labels.v1', 'windows': len(index), 'chunk_size': 8,
                'dataset_manifest': manifest, 'tasks': [{'task_index': i, 'task_id': f'task{i}'} for i in range(9)]}
    contract['sha256'] = manifest_fingerprint(contract)
    (tmp_path/'manifest.json').write_text(json.dumps(contract))
    (tmp_path/'metadata').mkdir()
    (tmp_path/'shards').mkdir()
    np.save(tmp_path/'window_index.npy', index)
    np.save(tmp_path/'task_id.npy', np.array([f'task{i}' for i in index['task_index']]))
    np.savez(tmp_path/'metadata/teacher_normalizer.npz', scale=np.ones(7))

    class Draws:
        def get(self, ids):
            return (np.stack([window_draws(int(i), 20261065, 20261066)[0] for i in ids]),
                    np.zeros((len(ids), 64, 16, 7), dtype=np.float32))

    for start, stop in shard_ranges(len(index), 8):
        steps, noise = Draws().get(index['window_id'][start:stop])
        losses = np.random.default_rng(start).random((stop-start, 64, 10), dtype=np.float32)
        arrays = dict(window_index=index[start:stop], losses=losses, noise=noise, timesteps=steps,
                      raw_signed_utility=losses[:, :, 1:]-losses[:, :, :1], **aggregate_losses(losses, steps))
        commit_shard(tmp_path/'shards', start, stop, contract['sha256'], arrays)
    finalize(tmp_path, contract, index, Draws())
    reader = UtilityLabelStore(tmp_path)
    assert reader.utility.shape == (18, 9)
    assert (reader.utility < 0).any()
    assert (reader.variance >= 0).all()
    np.testing.assert_array_equal(reader.index, index)
