"""Shape and conditioning tests for the data-format-independent AdaViP core."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from adavip.model import AdaViPPerception
from adavip.policy import AdaViPPolicy


class DummyBackbone(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.projection = nn.Linear(feature_dim, action_dim)

    def forward(self, fused):
        return self.projection(fused)


def make_policy():
    perception = AdaViPPerception(
        modality_dims={"vision": 8, "tactile": 6, "proprio": 4},
        task_dim=5,
        latent_dim=12,
        progress_dim=2,
        fusion_heads=3,
    )
    backbone = DummyBackbone(feature_dim=12, action_dim=7)
    return AdaViPPolicy(perception=perception, backbone=backbone)


def test_policy_preserves_batch_time_and_action_shapes():
    policy = make_policy()
    modalities = {
        "vision": torch.randn(2, 5, 8),
        "tactile": torch.randn(2, 5, 6),
        "proprio": torch.randn(2, 5, 4),
    }
    action = policy(
        modalities,
        task_embedding=torch.randn(2, 5),
        progress=torch.randn(2, 5, 2),
    )
    assert action.shape == (2, 5, 7)


def test_task_condition_changes_generated_perception_parameters():
    policy = make_policy()
    modalities = {
        "vision": torch.randn(2, 3, 8),
        "tactile": torch.randn(2, 3, 6),
        "proprio": torch.randn(2, 3, 4),
    }
    first = policy(
        modalities, task_embedding=torch.zeros(2, 5), return_perception=True
    )["perception"]
    second = policy(
        modalities, task_embedding=torch.ones(2, 5), return_perception=True
    )["perception"]
    assert not torch.allclose(first.context, second.context)
    assert not torch.allclose(first.fused, second.fused)


def test_base_encoders_are_frozen():
    perception = AdaViPPerception(
        modality_dims={"vision": 4}, task_dim=3, latent_dim=8, fusion_heads=2
    )
    encoder = nn.Linear(10, 4)
    policy = AdaViPPolicy(
        perception=perception,
        backbone=DummyBackbone(feature_dim=8, action_dim=2),
        base_encoders={"vision": encoder},
    )
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    policy.train()
    assert not encoder.training
