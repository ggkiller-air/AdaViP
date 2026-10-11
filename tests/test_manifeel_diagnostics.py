"""CPU regression tests for Power Plug diagnostic recording."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from adavip.manifeel.diagnostics import PolicyRecorder, RolloutRecorder, attach_diagnostics


class StubEnv:
    """Two-environment simulator with per-step successes and poses."""

    def __init__(self) -> None:
        self.index = 0

    def reset(self):
        """Return an initial end-effector pose."""
        return {"state": np.zeros((2, 7), dtype=np.float32)}

    def step(self, action):
        """Mark the first environment successful only on the second step."""
        self.index += 1
        success = np.array([self.index == 2, False])
        observation = {"state": np.ones((2, 7), dtype=np.float32) * self.index}
        return observation, np.zeros(2), np.zeros(2), {"success": success}


class StubEncoder(nn.Module):
    """Tiny baseline observation encoder with a tracked image module."""

    def __init__(self):
        super().__init__()
        self.key_model_map = nn.ModuleDict({"wrist": nn.Identity()})

    def forward(self, observation):
        """Return a feature vector for all observations."""
        return self.key_model_map["wrist"](observation["wrist"]).flatten(1)


class StubNormalizer:
    """Provide action-range metadata without loading a checkpoint."""

    def get_input_stats(self):
        """Return the training action limits."""
        return {"min": torch.zeros(6), "max": torch.ones(6)}


class StubPolicy(nn.Module):
    """Predict deterministic chunks while exercising the observation encoder."""

    def __init__(self):
        super().__init__()
        self.obs_encoder = StubEncoder()
        self.normalizer = {"action": StubNormalizer()}

    def predict_action(self, observation):
        """Encode the latest camera input and return eight zero actions."""
        self.obs_encoder({"wrist": observation["wrist"][:, -1]})
        return {"action": torch.zeros((observation["wrist"].shape[0], 8, 6))}


def test_rollout_recorder_persists_per_environment_success(tmp_path):
    """A transient success must survive as a per-environment outcome."""
    recorder = RolloutRecorder(StubEnv(), tmp_path)
    recorder.reset()
    recorder.step(np.zeros((2, 6), dtype=np.float32))
    recorder.step(np.ones((2, 6), dtype=np.float32))
    summary = recorder.save()
    assert summary["success_by_env"] == [True, False]
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["executed_steps"] == 2
    with np.load(tmp_path / "rollout_trace.npz") as trace:
        assert trace["executed_actions"].shape == (2, 2, 6)
        assert trace["ee_pose"].shape == (2, 2, 7)
        assert trace["initial_state"].shape == (2, 7)


def test_rollout_records_pairwise_target_contact_without_changing_observations(tmp_path):
    """Target contact is distinct from finger contact and is read without simulator edits."""
    from types import SimpleNamespace

    env = StubEnv()
    force = torch.zeros(2, 3, 3, 3)
    force[:, 0, 1, 2] = 2.0
    env.envs = SimpleNamespace(contact_force_pairwise=force, plug_body_id_env=0,
                               socket_body_id_env=1, plug_pos=torch.zeros(2, 3),
                               keypoints_plug=torch.zeros(2, 4, 3),
                               keypoints_socket=torch.ones(2, 4, 3))
    recorder = RolloutRecorder(env, tmp_path)
    recorder.reset()
    recorder.step(np.zeros((2, 6)))
    recorder.save()
    with np.load(tmp_path / "rollout_trace.npz") as trace:
        np.testing.assert_array_equal(trace["plug_socket_force"][0, :, 2], [2, 2])
        assert trace["keypoints_plug"].shape == (1, 2, 4, 3)
        assert trace["initial_plug_socket_force"].shape == (2, 3)
        assert trace["reward_by_step"].shape == (1, 2)


def test_policy_recorder_captures_inputs_features_and_chunks(tmp_path):
    """Baseline diagnostics should capture raw camera values and feature norms."""
    policy = StubPolicy()
    recorder = PolicyRecorder(policy, tmp_path)
    observations = {"wrist": torch.ones((2, 2, 3, 4, 4)), "state": torch.zeros((2, 2, 7))}
    policy.predict_action(observations)
    summary = recorder.save()
    assert summary["predicted_chunks"] == 1
    assert summary["nonfinite_actions"] == 0
    assert summary["action_training_stats"]["max"] == [1.0] * 6
    assert json.loads((tmp_path / "input_stats.json").read_text())[0]["wrist"]["max"] == 1.0
    assert len(json.loads((tmp_path / "feature_stats.json").read_text())["resnet_wrist"]) == 1
    assert (tmp_path / "images/chunk000_wrist_0.png").is_file()
    with np.load(tmp_path / "policy_chunks.npz") as trace:
        assert trace["action_chunks"].shape == (1, 2, 8, 6)


def test_policy_recorder_supports_cs_fusion_without_hypernet(tmp_path):
    """Record frozen fusion checkpoints without assuming FiLM modules."""
    class StubCSFusionEncoder(StubEncoder):
        def __init__(self):
            super().__init__()
            self.clip_encoder = nn.Identity()
            self.sparsh_encoder = nn.Identity()
            self.vision_projection = nn.Identity()
            self.tactile_projection = nn.Identity()
            self.cross_attention = nn.MultiheadAttention(4, 1)

    policy = StubPolicy()
    policy.obs_encoder = StubCSFusionEncoder()
    recorder = PolicyRecorder(policy, tmp_path)
    assert "hypernet" not in recorder.feature_stats
    observations = {"wrist": torch.ones((2, 2, 3, 4, 4)), "state": torch.zeros((2, 2, 7))}
    policy.predict_action(observations)
    assert recorder.save()["predicted_chunks"] == 1


def test_ablation_rejects_baseline_policy(tmp_path):
    """Inference-only FiLM interventions must not silently alter a baseline."""
    with pytest.raises(ValueError, match="FilmFusionObsEncoder"):
        PolicyRecorder(StubPolicy(), tmp_path, ablation="no_film")


def test_attach_diagnostics_supports_baseline_and_film_wrappers(tmp_path):
    """Both runners must instrument the video wrapper's actual simulator."""
    class Container:
        pass

    for film in (False, True):
        video = Container()
        video.env = StubEnv()
        multistep = Container()
        multistep.env = video
        underlying = Container()
        underlying.env = multistep
        runner = underlying
        if film:
            adapter = Container()
            adapter.env = multistep
            underlying.env = adapter
            runner = Container()
            runner._runner = underlying
        rollout, policy = attach_diagnostics(runner, StubPolicy(), tmp_path)
        assert video.env is rollout
        assert isinstance(video.env.env, StubEnv)
        assert isinstance(policy, PolicyRecorder)


def test_full_info_wrapper_retains_early_success():
    """A success outside the last observation window must count for the episode."""
    gym = pytest.importorskip("gym")
    pytest.importorskip("manifeel.gym_util.multistep_wrapper")
    from adavip.manifeel.full_info_wrapper import FullInfoMultiStepWrapper
    from adavip.manifeel.success_metrics import aggregate_success_rate

    class TimeLimitedEnv(gym.Env):
        """Emit a single early success in a five-step rollout."""

        def __init__(self):
            self.num_envs = 2
            self.index = 0
            self.action_space = gym.spaces.Box(-1, 1, shape=(2, 6))
            self.observation_space = gym.spaces.Dict({
                "state": gym.spaces.Box(-1, 1, shape=(2, 7))})

        def reset(self):
            """Reset the step counter and observations."""
            self.index = 0
            return {"state": np.zeros((2, 7), dtype=np.float32)}

        def step(self, action):
            """Mark success only at the second step."""
            self.index += 1
            return ({"state": np.zeros((2, 7), dtype=np.float32)},
                    np.zeros(2), np.zeros(2, dtype=bool),
                    {"success": np.array([self.index == 2, False])})

    env = FullInfoMultiStepWrapper(TimeLimitedEnv(), n_obs_steps=2, n_action_steps=1)
    env.reset()
    for _ in range(5):
        env.step(np.zeros((2, 1, 6), dtype=np.float32))
    infos = env.get_infos()
    assert infos["success"].shape == (2, 5)
    assert aggregate_success_rate(infos) == pytest.approx(0.5)
    env.reset()
    assert env.get_infos() == {}


def test_quaternion_canonicalization_preserves_rotation(tmp_path):
    """Flip all quaternion components before the policy receives negative w."""
    policy = StubPolicy()
    recorded = []
    original_predict_action = policy.predict_action

    def capture_state(observation):
        recorded.append(observation["state"].clone())
        return original_predict_action(observation)

    policy.predict_action = capture_state
    recorder = PolicyRecorder(policy, tmp_path, canonicalize_quaternion=True)
    observations = {"wrist": torch.ones((2, 2, 3, 4, 4)),
                    "state": torch.zeros((2, 2, 7))}
    observations["state"][0, :, 3:7] = torch.tensor([-0.9, 0.1, -0.2, 0.3])
    policy.predict_action(observations)
    summary = recorder.save()
    torch.testing.assert_close(recorded[0][0, 0, 3:7],
                               torch.tensor([0.9, -0.1, 0.2, -0.3]))
    torch.testing.assert_close(observations["state"][0, 0, 3:7],
                               torch.tensor([-0.9, 0.1, -0.2, 0.3]))
    assert summary["quaternion_sign_corrections"] == 2
def test_contact_phase_selection_uses_only_executed_history(tmp_path) -> None:
    """Phase-gated intervention must not use future contact or post-success samples."""
    recorder = RolloutRecorder(object(), tmp_path)
    recorder.initial_task_state["plug_socket_force"] = np.zeros((2, 3))
    np.testing.assert_array_equal(recorder.current_contact_phase("pre_contact"), [True, True])
    recorder.rewards.append(np.zeros(2))
    recorder.task_states["plug_socket_force"].append(np.array([[1., 0, 0], [0, 0, 0]]))
    np.testing.assert_array_equal(recorder.current_contact_phase("post_contact_pre_success"), [True, False])
    recorder.rewards.append(np.array([1, 0]))
    recorder.task_states["plug_socket_force"].append(np.zeros((2, 3)))
    np.testing.assert_array_equal(recorder.current_contact_phase("post_contact_pre_success"), [False, False])
    np.testing.assert_array_equal(recorder.current_contact_phase("pre_contact"), [False, True])
