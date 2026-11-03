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


@pytest.mark.parametrize('dropout', [False, True])
def test_l1_masks_preserve_native_full_path_and_remove_input_influence(tmp_path: Path, data_args, dropout):
    from test_language_conditioning import small_config
    from adavip.manifeel.language_conditioning_dataset import LanguageConditioningDataset
    from adavip.manifeel.language_conditioning_workspace import LanguageConditioningWorkspace
    dataset = LanguageConditioningDataset(**data_args)
    if dropout:
        from test_l1_dropout import dropout_config
        from adavip.manifeel.l1_dropout import L1DropoutWorkspace
        workspace = L1DropoutWorkspace(dropout_config(data_args, tmp_path/'mask'))
    else:
        workspace = LanguageConditioningWorkspace(small_config(data_args, tmp_path/'mask'))
    policy = workspace.model.eval()
    policy.set_normalizer(dataset.get_normalizer())
    observations = torch.utils.data.default_collate([dataset[3], dataset[36]])['obs']
    original = mask_conditionings(policy, observations)
    if dropout:
        with torch.inference_mode():
            normalized = policy.normalizer.normalize(observations)
            for condition, presence in enumerate(policy.condition_presence):
                expected = policy.obs_encoder(normalized, presence.expand(2, -1))
                torch.testing.assert_close(original[condition], expected, rtol=0, atol=0)
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


@pytest.mark.parametrize('protocol', ['original_snapshot', 'dropout10'])
def test_snapshot_teacher_identity_and_normalizer_are_required(tmp_path: Path, data_args, protocol):
    from omegaconf import OmegaConf
    from test_language_conditioning import small_config
    from test_l1_dropout import dropout_config
    from adavip.manifeel.l1_utility_labels import validate_teacher_protocol

    baseline = small_config(data_args, tmp_path/'baseline')
    baseline_path = tmp_path/'training.yaml'
    OmegaConf.save(baseline, baseline_path)
    model = dropout_config(data_args, tmp_path/'teacher') if protocol == 'dropout10' else baseline
    reference = tmp_path/'labels'
    reference.mkdir()
    (reference/'manifest.json').write_text(json.dumps({'sha256': 'abc', 'teacher': {'normalizer_sha256': 'norm'}}))
    (reference/'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'contract_sha256': 'abc'}))
    cfg = OmegaConf.create({'teacher_protocol': protocol, 'expected_epoch': 50, 'expected_step': 55700,
                           'baseline_training_config': str(baseline_path), 'normalizer_reference_archive': str(reference)})
    metadata = {'epoch': 50, 'global_step': 55700, 'weights': 'ema'}
    validate_teacher_protocol(cfg, model, metadata, tmp_path/'epoch50.ckpt', 'norm')
    with pytest.raises(ValueError, match='epoch/step/variant'):
        validate_teacher_protocol(cfg, model, {**metadata, 'epoch': 200}, tmp_path/'epoch50.ckpt', 'norm')
    with pytest.raises(ValueError, match='normalizer'):
        validate_teacher_protocol(cfg, model, metadata, tmp_path/'epoch50.ckpt', 'different')
    with pytest.raises(ValueError, match='fixed epoch'):
        validate_teacher_protocol(cfg, model, metadata, tmp_path/'latest.ckpt', 'norm')
    model.policy.full_input_probability = 0.8
    with pytest.raises(ValueError):
        validate_teacher_protocol(cfg, model, metadata, tmp_path/'epoch50.ckpt', 'norm')


def test_paired_comparison_preserves_negative_scores_and_constant_correlations():
    from scripts.manifeel.compare_l1_utility_labels import summarize_pair
    result = summarize_pair(np.array([-1., 1., 3.]), np.array([-2., 0., 1.]))
    assert result['original_mean'] == 1.
    assert result['dropout_negative_fraction'] == 1/3
    assert result['sign_agreement'] == 2/3
    np.testing.assert_allclose(result['delta_mean'], -4/3)
    assert summarize_pair(np.ones(3), np.zeros(3))['pearson'] is None


def test_paired_comparison_rejects_epoch_or_draw_mismatches():
    import copy
    from scripts.manifeel.compare_l1_utility_labels import validate_pair
    common = {'dimensions': [], 'conditions': [], 'protocol': {}, 'draws': {'seed': 42},
              'reference_manifest_sha256': 'abc', 'mask_semantics': 'zero', 'validity': 'full',
              'label_normalization': 'none', 'teacher': {'normalizer_sha256': 'norm',
              'metadata': {'epoch': 50, 'global_step': 55700, 'weights': 'ema'}}, 'runtime': {}}
    original = SimpleNamespace(index=np.arange(3), task_ids=np.arange(3),
                               manifest={**copy.deepcopy(common), 'teacher_protocol': 'original_snapshot'})
    dropout = SimpleNamespace(index=np.arange(3), task_ids=np.arange(3),
                              manifest={**copy.deepcopy(common), 'teacher_protocol': 'dropout10'})
    validate_pair(original, dropout)
    dropout.manifest['teacher']['metadata']['epoch'] = 500
    with pytest.raises(ValueError, match='checkpoint differs: epoch'):
        validate_pair(original, dropout)
    dropout.manifest['teacher']['metadata']['epoch'] = 50
    dropout.manifest['draws']['seed'] = 43
    with pytest.raises(ValueError, match='protocol differs: draws'):
        validate_pair(original, dropout)
