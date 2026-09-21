from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adavip.manifeel.action_adapter import action_loss_mask, pad_action, truncate_action_for_task
from adavip.manifeel.success_metrics import aggregate_success_rate
from adavip.manifeel.task_embeddings import (
    embedding_dim,
    expected_embedding_dim,
    load_task_embeddings,
    validate_task_embedding_dim,
)
from adavip.manifeel.task_protocol import DEFAULT_TASK_SPECS
from adavip.manifeel.training_state import (
    next_training_state,
    remaining_epochs,
    validation_improved,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_task_specs_cover_nine_tasks() -> None:
    task_ids = [spec.task_id for spec in DEFAULT_TASK_SPECS]
    assert len(task_ids) == 9
    assert len(set(task_ids)) == 9
    assert sum(spec.action_dim == 6 for spec in DEFAULT_TASK_SPECS) == 5
    assert sum(spec.action_dim == 7 for spec in DEFAULT_TASK_SPECS) == 4


def test_action_padding_mask_and_truncation() -> None:
    action = np.ones((3, 6), dtype=np.float32)
    padded = pad_action(action)
    assert padded.shape == (3, 7)
    np.testing.assert_allclose(padded[:, :6], 1.0)
    np.testing.assert_allclose(padded[:, 6], 0.0)

    mask = action_loss_mask(action_dim=6, horizon=3)
    np.testing.assert_allclose(mask[:, :6], 1.0)
    np.testing.assert_allclose(mask[:, 6], 0.0)

    full_action = np.ones((2, 4, 7), dtype=np.float32)
    assert truncate_action_for_task(full_action, action_dim=6).shape == (2, 4, 6)


def test_task_embedding_archive_validation(tmp_path) -> None:
    path = tmp_path / "task_embeddings.npz"
    task_ids = np.array([spec.task_id for spec in DEFAULT_TASK_SPECS])
    embeddings = np.arange(len(task_ids) * 4, dtype=np.float32).reshape(len(task_ids), 4)
    np.savez(path, task_ids=task_ids, embeddings=embeddings)

    loaded = load_task_embeddings(path, list(DEFAULT_TASK_SPECS))
    assert list(loaded) == task_ids.tolist()
    assert embedding_dim(loaded) == 4

    bad_path = tmp_path / "bad_embeddings.npz"
    np.savez(bad_path, task_ids=task_ids[:-1], embeddings=embeddings[:-1])
    with pytest.raises(KeyError):
        load_task_embeddings(bad_path, list(DEFAULT_TASK_SPECS))


def test_task_embedding_shape_meta_validation(tmp_path) -> None:
    path = tmp_path / "task_embeddings.npz"
    task_ids = np.array([spec.task_id for spec in DEFAULT_TASK_SPECS])
    embeddings = np.ones((len(task_ids), 4), dtype=np.float32)
    np.savez(path, task_ids=task_ids, embeddings=embeddings)

    loaded = load_task_embeddings(path, list(DEFAULT_TASK_SPECS))
    shape_meta = {"obs": {"task_embedding": {"shape": [4], "type": "low_dim"}}}
    assert expected_embedding_dim(shape_meta) == 4
    assert validate_task_embedding_dim(shape_meta, loaded) == 4

    bad_shape_meta = {"obs": {"task_embedding": {"shape": [8], "type": "low_dim"}}}
    with pytest.raises(ValueError):
        validate_task_embedding_dim(bad_shape_meta, loaded)


def test_success_rate_aggregation_shapes() -> None:
    assert aggregate_success_rate({"success": np.array(1.0)}) == 1.0
    assert aggregate_success_rate({"success": np.array([1, 0, 0, 0])}) == 0.25
    env_time = np.array([[0, 1, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    assert math.isclose(aggregate_success_rate({"success": env_time}), 2 / 3)
    env_time_channels = env_time[:, :, None]
    assert math.isclose(aggregate_success_rate({"success": env_time_channels}), 2 / 3)
    env_time_with_scalar_fallback = {
        "success": np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
        "successes": np.array([1.0], dtype=np.float32),
    }
    assert aggregate_success_rate(env_time_with_scalar_fallback) == 0.0
    assert math.isclose(
        aggregate_success_rate({"successes": np.array([0.1, 0.4, 0.6])}),
        0.6,
        abs_tol=1e-6,
    )


def test_multitask_rgb_preprocessing_skips_matching_resize(monkeypatch) -> None:
    pytest.importorskip("torch")
    dataset_module = pytest.importorskip("adavip.manifeel.multitask_dataset")
    dataset = object.__new__(dataset_module._SingleTaskZarrDataset)
    dataset.n_obs_steps = 2
    dataset.rgb_keys = ["front"]
    dataset.lowdim_keys = ["state"]
    dataset.shape_meta = {
        "obs": {"front": {"shape": [3, 4, 5], "type": "rgb"}}
    }
    images = np.arange(3 * 4 * 5 * 3, dtype=np.float32).reshape(3, 4, 5, 3)
    sample = {
        "front": images,
        "state": np.ones((3, 7), dtype=np.float32),
        "action": np.ones((3, 7), dtype=np.float32),
    }

    def fail_resize(*args, **kwargs):
        raise AssertionError("matching images must not be resized")

    monkeypatch.setattr(dataset_module.cv2, "resize", fail_resize)
    result = dataset._sample_to_data(sample)

    assert result["obs"]["front"].shape == (2, 3, 4, 5)
    assert np.shares_memory(result["obs"]["front"], images)


def test_multitask_config_declares_protocol() -> None:
    config_path = REPO_ROOT / "configs/manifeel/train_multitask_diffusion_workspace.yaml"
    text = config_path.read_text()
    assert "_target_: adavip.manifeel.multitask_dataset.MultiTaskManiFeelDataset" in text
    assert "_target_: adavip.manifeel.multitask_runner.MultiTaskManiFeelRunner" in text
    assert "front:" in text
    assert "wrist:" in text
    assert "task_embedding:" in text
    assert "shape:\n          - ${task_embedding_dim}" in text
    assert "action:" in text
    assert "shape: [7]" in text
    for spec in DEFAULT_TASK_SPECS:
        assert f"task_id: {spec.task_id}" in text
        assert f"dataset: {spec.dataset}" in text
        assert f"isaacgym_cfg_name: {spec.isaacgym_cfg_name}" in text


def test_table1_multitask_configs_have_training_defaults() -> None:
    configs = sorted((REPO_ROOT / "configs/manifeel").glob("train_multitask_*.yaml"))
    assert [path.name for path in configs] == ["train_multitask_diffusion_workspace.yaml"]
    text = configs[0].read_text()
    assert "exp_name: table1_dp_b416_w12_ep300_seed42" in text
    for expected in (
        "batch_size: 416", "num_workers: 12", "num_epochs: 300",
        "checkpoint_every: 30", "val_every: 30", "sample_every: 30",
        "rollout_every: 0", "project: manifeel_multitask",
        "resume: allow", "name: ${exp_name}", "id: ${exp_name}",
        "mode: offline", "save_best_val_ckpt: true",
    ):
        assert expected in text


def test_total_epoch_target_for_fresh_training() -> None:
    assert next_training_state(0, 0, checkpoint_loaded=False) == (0, 0)
    assert remaining_epochs(total_epochs=300, next_epoch=0) == 300


def test_total_epoch_target_after_epoch25_checkpoint() -> None:
    epoch, global_step = next_training_state(25, 4159, checkpoint_loaded=True)
    assert (epoch, global_step) == (26, 4160)
    assert remaining_epochs(total_epochs=300, next_epoch=epoch) == 274


def test_total_epoch_target_after_epoch299_checkpoint() -> None:
    epoch, global_step = next_training_state(299, 47999, checkpoint_loaded=True)
    assert (epoch, global_step) == (300, 48000)
    assert remaining_epochs(total_epochs=300, next_epoch=epoch) == 0


def test_failed_resume_does_not_advance_training_state() -> None:
    assert next_training_state(0, 0, checkpoint_loaded=False) == (0, 0)


def test_workspace_reports_only_successful_checkpoint_loads() -> None:
    workspace_module = pytest.importorskip(
        "manifeel.workspace.train_diffusion_unet_image_workspace"
    )
    workspace = object.__new__(workspace_module.TrainDiffusionUnetImageWorkspace)
    cfg = SimpleNamespace(training=SimpleNamespace(resume=True))
    checkpoint_path = Path("latest_epoch25.ckpt")
    workspace.get_latest_checkpoint_path = lambda: checkpoint_path
    workspace.get_previous_checkpoint_path = lambda path: None
    workspace.load_checkpoint = lambda path: None
    assert workspace.resume_training(cfg) is True

    def fail_load(path):
        raise RuntimeError("corrupt checkpoint")

    workspace.load_checkpoint = fail_load
    assert workspace.resume_training(cfg) is False


def test_validation_improvement_requires_a_finite_new_minimum() -> None:
    assert validation_improved(0.5, float("inf"))
    assert validation_improved(0.4, 0.5)
    assert not validation_improved(0.5, 0.5)
    assert not validation_improved(float("nan"), 0.5)


def test_periodic_checkpoint_pruning_preserves_best_val(tmp_path) -> None:
    workspace_module = pytest.importorskip(
        "manifeel.workspace.train_diffusion_unet_image_workspace"
    )
    for epoch in (30, 60, 90, 120):
        (tmp_path / f"latest_epoch{epoch}.ckpt").touch()
    best_path = tmp_path / "best_val.ckpt"
    best_path.touch()

    workspace_module._prune_checkpoints(tmp_path)

    assert sorted(path.name for path in tmp_path.glob("latest_epoch*.ckpt")) == [
        "latest_epoch120.ckpt",
        "latest_epoch60.ckpt",
        "latest_epoch90.ckpt",
    ]
    assert best_path.is_file()


def test_fm_embedding_eval_uses_retained_checkpoint_and_dynamic_output() -> None:
    text = (REPO_ROOT / "slurm/manifeel/eval_fm_embedding_ablation.sbatch").read_text()
    assert "latest_epoch900.ckpt" in text
    assert "CHECKPOINT_EPOCH=\"${BASH_REMATCH[1]}\"" in text
    assert "table1_fm_epoch${CHECKPOINT_EPOCH}_power_plug" in text


def test_fm_all_task_eval_runs_full_sequential_suite() -> None:
    text = (REPO_ROOT / "slurm/manifeel/eval_fm_all_tasks.sbatch").read_text()
    assert "eval_multitask_checkpoint.py" in text
    assert "--task-id" not in text
    assert "MANIFEEL_TASK_EMBEDDING_MODE=correct" in text
    assert 'N_TEST="${MANIFEEL_EVAL_N_TEST:-10}"' in text
    assert 'MAX_STEPS="${MANIFEEL_EVAL_MAX_STEPS:-500}"' in text
    assert "camera_preflight=passed" in text


def test_fm_task_eval_isolates_one_full_rollout() -> None:
    text = (REPO_ROOT / "slurm/manifeel/eval_fm_task.sbatch").read_text()
    assert 'TASK_ID="${MANIFEEL_EVAL_TASK_ID:?Set MANIFEEL_EVAL_TASK_ID}"' in text
    assert '--task-id "${TASK_ID}"' in text
    assert "MANIFEEL_TASK_EMBEDDING_MODE=correct" in text
    assert 'N_TEST="${MANIFEEL_EVAL_N_TEST:-10}"' in text
    assert 'MAX_STEPS="${MANIFEEL_EVAL_MAX_STEPS:-500}"' in text
    assert "camera_preflight=passed" in text


def test_adavip_fm_upload_uses_complete_workspace_checkpoints() -> None:
    text = (REPO_ROOT / "scripts/manifeel/upload_adavip_fm_modelscope.sh").read_text()
    assert 'EPOCHS="${ADAVIP_FM_EPOCHS:-120 150 210}"' in text
    assert "MIN_FULL_CHECKPOINT_BYTES" in text
    assert "archive/data.pkl" in text
    assert 'adavip-fm/checkpoints/latest_epoch${epoch}.ckpt' in text
    assert "hyper_adavip_obs_encoder.py" in text
    assert "export_deployable_checkpoint.py" not in text
    assert ".deploy.ckpt" not in text


def test_masked_flow_matching_loss_ignores_padded_actions() -> None:
    torch = pytest.importorskip("torch")
    policy_module = pytest.importorskip("adavip.manifeel.multitask_policy")
    if policy_module.FMDP is None:
        pytest.skip("torchcfm is unavailable")

    class IdentityNormalizer:
        def normalize(self, value):
            return value

        def __getitem__(self, key):
            assert key == "action"
            return self

    class ObsEncoder(torch.nn.Module):
        def forward(self, obs):
            return obs["state"]

    class ZeroVelocity(torch.nn.Module):
        def forward(self, sample, timestep, global_cond):
            return torch.zeros_like(sample)

    class DeterministicFlowMatcher:
        def sample_location_and_conditional_flow(self, x0, trajectory):
            timestep = torch.zeros(trajectory.shape[0], dtype=trajectory.dtype)
            return timestep, trajectory, trajectory

    policy = object.__new__(policy_module.MaskedFMDP)
    torch.nn.Module.__init__(policy)
    policy.normalizer = IdentityNormalizer()
    policy.obs_encoder = ObsEncoder()
    policy.model = ZeroVelocity()
    policy.FM = DeterministicFlowMatcher()
    policy.obs_as_global_cond = True
    policy.n_obs_steps = 1
    policy.action_dim = 2

    batch = {
        "obs": {"state": torch.zeros((1, 1, 1))},
        "action": torch.tensor([[[1.0, 100.0], [1.0, 100.0]]]),
        "action_loss_mask": torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    }
    assert torch.isclose(policy.compute_loss(batch), torch.tensor(1.0))


def test_masked_flow_matching_sampling_uses_gaussian_prior(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    policy_module = pytest.importorskip("adavip.manifeel.multitask_policy")
    if policy_module.FMDP is None:
        pytest.skip("torchcfm is unavailable")

    class ZeroVelocity(torch.nn.Module):
        def forward(self, sample, timestep, local_cond=None, global_cond=None):
            return torch.zeros_like(sample)

    expected = torch.tensor(
        [[[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]]], dtype=torch.float32
    )
    calls = []

    def fake_randn(shape, **kwargs):
        calls.append((shape, kwargs))
        return expected.clone()

    monkeypatch.setattr(torch, "randn", fake_randn)
    policy = object.__new__(policy_module.MaskedFMDP)
    torch.nn.Module.__init__(policy)
    policy.model = ZeroVelocity()
    policy.num_inference_steps = 2

    condition_data = torch.zeros_like(expected)
    sample = policy.conditional_sample(
        condition_data=condition_data,
        condition_mask=torch.zeros_like(condition_data, dtype=torch.bool),
    )

    assert len(calls) == 1
    assert calls[0][0] == condition_data.shape
    assert torch.equal(sample, expected)


def test_ball_sorting_isaacgym_config_uses_existing_task() -> None:
    config_path = (
        REPO_ROOT
        / "third_party/manifeel/manifeel/config/isaacgym_config_ball_sorting.yaml"
    )
    assert "- task: TacSLTaskBallSorting" in config_path.read_text()
