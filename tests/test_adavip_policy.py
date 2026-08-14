"""Shape and conditioning tests for the data-format-independent AdaViP core."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from adavip.model import AdaViPPerception
from adavip.manifeel.adavip_obs_encoder import AdaViPObsEncoder
from adavip.policy import AdaViPPolicy


class DummyBackbone(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int):
        super().__init__()
        self.projection = nn.Linear(feature_dim, action_dim)

    def forward(self, fused):
        return self.projection(fused)


class DummyRgbEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, image):
        batch_size = image.shape[0]
        pooled = image.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        return pooled.repeat(1, self.output_dim)


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


def test_adavip_obs_encoder_groups_bilateral_tactile_before_dp_condition():
    shape_meta = {
        "obs": {
            "front": {"shape": [3, 4, 4], "type": "rgb"},
            "wrist": {"shape": [3, 4, 4], "type": "rgb"},
            "left_tactile_camera_taxim": {"shape": [3, 4, 4], "type": "rgb"},
            "right_tactile_camera_taxim": {"shape": [3, 4, 4], "type": "rgb"},
            "state": {"shape": [7], "type": "low_dim"},
            "task_embedding": {"shape": [5], "type": "low_dim"},
        }
    }
    encoder = AdaViPObsEncoder(
        shape_meta=shape_meta,
        rgb_model=DummyRgbEncoder(output_dim=4),
        state_feature_dim=3,
        latent_dim=8,
        fusion_heads=2,
        hypernet_hidden_dim=16,
        adaptive_rank=2,
        append_task_embedding=True,
    )
    obs = {
        "front": torch.randn(2, 3, 4, 4),
        "wrist": torch.randn(2, 3, 4, 4),
        "left_tactile_camera_taxim": torch.randn(2, 3, 4, 4),
        "right_tactile_camera_taxim": torch.randn(2, 3, 4, 4),
        "state": torch.randn(2, 7),
        "task_embedding": torch.randn(2, 5),
    }

    assert encoder.output_shape() == (13,)
    assert encoder(obs).shape == (2, 13)


def test_adavip_obs_encoder_can_freeze_rgb_backbones():
    shape_meta = {
        "obs": {
            "front": {"shape": [3, 4, 4], "type": "rgb"},
            "wrist": {"shape": [3, 4, 4], "type": "rgb"},
            "left_tactile_camera_taxim": {"shape": [3, 4, 4], "type": "rgb"},
            "right_tactile_camera_taxim": {"shape": [3, 4, 4], "type": "rgb"},
            "state": {"shape": [7], "type": "low_dim"},
            "task_embedding": {"shape": [5], "type": "low_dim"},
        }
    }
    encoder = AdaViPObsEncoder(
        shape_meta=shape_meta,
        rgb_model=nn.Sequential(nn.Flatten(), nn.Linear(48, 4)),
        state_feature_dim=3,
        latent_dim=8,
        fusion_heads=2,
        hypernet_hidden_dim=16,
        adaptive_rank=2,
        freeze_rgb_model=True,
    )

    assert all(
        not parameter.requires_grad
        for model in encoder.key_model_map.values()
        for parameter in model.parameters()
    )
    encoder.train()
    assert all(not model.training for model in encoder.key_model_map.values())
