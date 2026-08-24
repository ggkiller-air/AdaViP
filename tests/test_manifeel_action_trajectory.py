from __future__ import annotations

import numpy as np
import pytest


def test_task_conditioned_policy_saves_action_trajectory(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    runner_module = pytest.importorskip("adavip.manifeel.multitask_runner")

    class FakePolicy:
        device = torch.device("cpu")
        dtype = torch.float32

        def reset(self):
            return None

        def predict_action(self, obs_dict):
            batch_size = obs_dict["state"].shape[0]
            return {
                "action": torch.arange(
                    batch_size * 2 * 7, dtype=torch.float32
                ).reshape(batch_size, 2, 7),
                "action_pred": torch.ones((batch_size, 4, 7)),
            }

    policy = runner_module._TaskConditionedPolicy(
        policy=FakePolicy(),
        task_embedding=np.arange(4, dtype=np.float32),
        action_dim=6,
        embedding_mode="correct",
        record_actions=True,
    )
    obs = {"state": torch.zeros((2, 2, 7))}
    policy.predict_action(obs)
    policy.predict_action(obs)

    output = tmp_path / "action_trajectory.npz"
    policy.save_action_trajectory(output)
    archive = np.load(output, allow_pickle=False)

    assert archive["action_chunks"].shape == (2, 2, 2, 6)
    assert archive["executed_actions"].shape == (2, 4, 6)
    assert archive["action_predictions"].shape == (2, 2, 4, 7)
    assert archive["embedding_mode"].item() == "correct"
    assert archive["embedding_source_task_id"].item() == "correct"
    np.testing.assert_array_equal(
        archive["task_embedding"], np.arange(4, dtype=np.float32)
    )


def test_task_conditioned_policy_uses_other_trained_task_embedding() -> None:
    torch = pytest.importorskip("torch")
    runner_module = pytest.importorskip("adavip.manifeel.multitask_runner")

    class FakePolicy:
        device = torch.device("cpu")
        dtype = torch.float32

        def reset(self):
            return None

        def predict_action(self, obs_dict):
            return {"action": torch.zeros((1, 2, 7))}

    alternate = np.full(4, 3.0, dtype=np.float32)
    policy = runner_module._TaskConditionedPolicy(
        policy=FakePolicy(),
        task_embedding=np.arange(4, dtype=np.float32),
        action_dim=6,
        embedding_mode="other_task",
        alternate_task_embedding=alternate,
        embedding_source_task_id="usb_insertion",
    )

    np.testing.assert_array_equal(policy.task_embedding.numpy(), alternate)
    assert policy.embedding_source_task_id == "usb_insertion"
