"""Native paired labels, gradient isolation and exact Tabero Student continuation."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('diffusers')

from test_tabero_field import batch, make_cache, tiny_policy
from adavip.tabero.utility import aggregate, make_draws, paired_losses, commit_shard, read_shard
from adavip.tabero.uvip import TaberoUViPPolicy
from adavip.tabero.uvip_train import Trainer, validate_gradient_routes


def student():
    teacher = tiny_policy()
    policy = TaberoUViPPolicy({**teacher.config, 'utility_weight': .05}, torch.ones(2, 6),
                            copy.deepcopy(teacher.obs_encoder))
    policy.initialize_from_teacher(teacher)
    return teacher, policy


def labeled_batch():
    value = batch()
    value.update(utility=torch.randn(2, 6), task_index=torch.tensor([0, 1]))
    return value


def test_inherited_context_normalizer_and_fresh_actor():
    torch.set_num_threads(1)
    teacher, policy = student()
    value = labeled_batch()
    tokens = teacher.obs_encoder.encode_tokens(value['obs'], teacher.normalizer)
    expected = teacher.obs_encoder.condition(tokens, value['obs']['language'], teacher.condition_presence[:1].expand(2, -1))
    torch.testing.assert_close(policy.encode_stage2(value['obs'])['z_t'], expected, rtol=0, atol=0)
    assert any(not torch.equal(v, policy.model.state_dict()[k]) for k, v in teacher.model.state_dict().items())
    for head in policy.film_hypernets:
        token = torch.randn(2, 32)
        torch.testing.assert_close(head(token, torch.randn(2, 128)), token, rtol=0, atol=0)


@pytest.mark.parametrize('phase', ['A', 'B', 'C'])
def test_stage_freezing(phase):
    _, policy = student()
    policy.set_phase(phase)
    assert all(p.requires_grad == (phase == 'C') for p in policy.obs_encoder.parameters())
    assert all(p.requires_grad == (phase != 'A') for p in policy.model.parameters())
    assert all(p.requires_grad for p in policy.utility_predictor.parameters())


def test_gradient_isolation_and_native_inference():
    _, policy = student()
    value = labeled_batch()
    assert set(validate_gradient_routes(policy, value)) == {'B', 'C'}
    result = policy.eval().predict_action(value['obs'])
    assert result['action'].shape == (2, 8, 13)
    assert torch.isfinite(result['action']).all()


def test_cached_native_paired_loss_with_padding():
    teacher = tiny_policy().eval().requires_grad_(False)
    value = batch()
    value['action_loss_mask'][0, :2] = False
    steps, noise = make_draws([13, 31])
    steps = steps[:, :2] % teacher.config['num_diffusion_steps']
    actual = paired_losses(teacher, value, steps, noise[:, :2], microbatch=3)
    for draw in range(2):
        for condition in range(7):
            native = teacher.compute_losses(value, torch.full((2,), condition), torch.from_numpy(noise[:, draw]),
                                            torch.from_numpy(steps[:, draw]))['loss']
            np.testing.assert_allclose(actual[:, draw, condition], native.numpy(), rtol=1e-5, atol=1e-7)


def test_k64_weighting_variance_signed_values_and_batch_independence():
    steps, noise = make_draws([4, 9, 21])
    again_steps, again_noise = make_draws([21])
    np.testing.assert_array_equal(steps[2:], again_steps)
    np.testing.assert_array_equal(noise[2:], again_noise)
    losses = np.ones((3, 64, 7))
    losses[:, :, 1] = 0
    losses[:, :, 2] = steps
    reduced = aggregate(losses, steps)
    np.testing.assert_array_equal(reduced['utility'][:, 0], -1)
    assert np.all(reduced['utility_variance'][:, 0] == 0)
    np.testing.assert_allclose(reduced['utility'], reduced['k32_utility'].mean(1), atol=1e-14)
    np.testing.assert_allclose(reduced['utility'], reduced['weighted_losses'][:, 1:] - reduced['weighted_losses'][:, :1])
    broken = steps.copy()
    broken[0, 0] = 99
    with pytest.raises(ValueError, match='ordering'):
        aggregate(losses, broken)


def test_atomic_shard_resume_identity(tmp_path):
    steps, noise = make_draws([0, 1])
    losses = np.ones((2, 64, 7))
    arrays = dict(source_index=np.arange(2), timesteps=steps, noise=noise, losses=losses, **aggregate(losses, steps))
    commit_shard(tmp_path, 0, 2, 'selected-teacher', arrays)
    saved = read_shard(tmp_path, 0, 2, 'selected-teacher')
    for key, value in arrays.items():
        np.testing.assert_array_equal(saved[key], value)
    with pytest.raises(ValueError, match='identity'):
        read_shard(tmp_path, 0, 2, 'different-teacher')


class DiagnosticLabels(torch.utils.data.Dataset):
    def __init__(self, source):
        self.source = source
        self.manifest = {'identity': 'fixture'}

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        value = self.source[index]
        value['utility'] = torch.full((6,), (index - 3) / 10)
        return value


def test_exact_resume_across_a_b_c_boundaries(tmp_path):
    source = make_cache(tmp_path / 'cache')
    source.horizon = 16
    dataset = DiagnosticLabels(source)
    _, original = student()
    config = {**original.config, 'seed': 42, 'batch_size': 6, 'num_epochs': 8, 'num_workers': 0,
              'encoder_lr': 8e-5, 'utility_lr': 1e-4, 'action_lr': 1e-4, 'betas': [.95, .999],
              'weight_decay': 1e-6, 'warmup_steps': 2, 'gradient_clip_norm': 1.,
              'checkpoint_every_steps': 100, 'checkpoint_epochs': [1, 6, 8]}
    identity = {'epoch': 150, 'weights': 'ema'}
    def make(name):
        torch.manual_seed(42)
        return Trainer(config, tmp_path / name, copy.deepcopy(original), dataset, identity, device='cpu')
    continuous = make('continuous')
    continuous.run()
    partial = make('resumed')
    partial.run(stop_after_steps=6)
    resumed = make('resumed')
    resumed.resume(tmp_path / 'resumed/checkpoints/latest.ckpt')
    resumed.run()
    for left, right in ((continuous.policy, resumed.policy), (continuous.ema.model, resumed.ema.model)):
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)
    for key, state in continuous.optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed.optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
    assert resumed.global_step == 8
    assert sorted(p.name for p in (tmp_path / 'resumed/checkpoints').glob('epoch*.ckpt')) == [
        'epoch0001.ckpt', 'epoch0006.ckpt', 'epoch0008.ckpt']


def test_label_pipeline_resume_finalize_and_join(tmp_path):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from adavip.tabero.utility import generate
    from adavip.tabero.uvip_train import LabeledDataset
    source = make_cache(tmp_path / 'cache')
    source.horizon = 16
    teacher = tiny_policy().eval().requires_grad_(False)
    teacher.config['num_diffusion_steps'] = 100
    teacher.noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2')
    identity = {'epoch': 150, 'weights': 'ema'}
    config = dict(cache_id=source.manifest['cache_id'], preload_ram=False, label_seed=20261104,
                  label_shard_size=4, label_window_batch=2, denoise_microbatch=128, num_workers=0)
    output = tmp_path / 'labels'
    generate(config, output, device='cpu', teacher=(teacher, identity), dataset=source)
    expected = np.load(output / 'utility.npy').copy()
    first_mtime = (output / 'shards/shard_000000_000004.npz').stat().st_mtime_ns
    (output / 'COMPLETE.json').unlink()
    (output / 'shards/shard_000004_000006.json').unlink()
    generate(config, output, device='cpu', teacher=(teacher, identity), dataset=source)
    np.testing.assert_array_equal(expected, np.load(output / 'utility.npy'))
    assert (output / 'shards/shard_000000_000004.npz').stat().st_mtime_ns == first_mtime
    joined = LabeledDataset(source, output, identity)
    assert len(joined) == 6 and joined[0]['utility'].shape == (6,)
    with pytest.raises(ValueError, match='identity'):
        LabeledDataset(source, output, {'epoch': 200})


def test_teacher_context_fp32_roundoff_and_exact_weights(monkeypatch):
    from adavip.tabero.uvip_train import verify_teacher_context
    teacher, policy = student()
    observations = labeled_batch()['obs']
    original = policy.encode_stage2
    def rounded(obs):
        result = original(obs)
        result['z_t'] = result['z_t'] + 1.2e-7
        return result
    monkeypatch.setattr(policy, 'encode_stage2', rounded)
    assert verify_teacher_context(policy, teacher, observations)['max_absolute_error'] < 2e-7
    with torch.no_grad():
        next(policy.obs_encoder.parameters()).flatten()[0].add_(1e-6)
    with pytest.raises(AssertionError):
        verify_teacher_context(policy, teacher, observations)
