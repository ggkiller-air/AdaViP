"""CPU regression checks for read-only HyperNet demonstration probes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from adavip.manifeel.hyper_resnet_obs_encoder import HyperResNetObsEncoder
from adavip.manifeel.hypernet_probe import (
    HyperNetProbeRecorder,
    HyperNetRolloutRecorder,
    select_episode_frames,
    summarize_probe,
    temporal_modulation_ratio,
)
from adavip.manifeel.action_probe import reconstruct_condition


def make_encoder(use_fusion: bool = True, use_fusion_hypernet: bool = False) -> HyperResNetObsEncoder:
    """Build a small deterministic encoder with real HyperNet and attention modules."""
    encoder = HyperResNetObsEncoder(
        shape_meta={"obs": {"wrist": {"shape": [3, 8, 8], "type": "rgb"},
                            "right_tactile_camera_taxim": {"shape": [3, 8, 8], "type": "rgb"},
                            "state": {"shape": [7], "type": "low_dim"}}},
        rgb_model=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 512)),
        use_fusion=use_fusion, use_fusion_hypernet=use_fusion_hypernet).eval()
    with torch.no_grad():
        encoder.hypernet[-1].weight.normal_(std=0.01)
        encoder.fusion_gate.fill_(0.3)
        if use_fusion_hypernet:
            encoder.fusion_hypernet[-1].weight.normal_(std=0.01)
    return encoder


@pytest.mark.parametrize("use_fusion,use_fusion_hypernet", [(False, False), (True, False), (True, True)])
def test_recorder_preserves_outputs_and_measures_actual_residual(
        use_fusion: bool, use_fusion_hypernet: bool) -> None:
    """Hooks capture learned modulation without altering the original inference."""
    torch.manual_seed(17)
    encoder = make_encoder(use_fusion, use_fusion_hypernet)
    observations = {"wrist": torch.rand(3, 3, 8, 8),
                    "right_tactile_camera_taxim": torch.rand(3, 3, 8, 8),
                    "state": torch.rand(3, 7)}
    with torch.inference_mode():
        expected = encoder(observations)
        recorder = HyperNetProbeRecorder(encoder)
        actual = encoder(observations)
        arrays = recorder.arrays()
        recorder.close()
    assert torch.equal(expected, actual)
    assert arrays["gamma"].shape == (3, 2, 512)
    delta = arrays["modulated_features"] - arrays["base_features"]
    expected_ratio = np.linalg.norm(delta, axis=-1) / np.linalg.norm(arrays["base_features"], axis=-1)
    np.testing.assert_allclose(arrays["residual_ratio"], expected_ratio, rtol=1e-3, atol=1e-7)
    assert ("attention" in arrays) == use_fusion
    assert ("fusion_gamma" in arrays) == use_fusion_hypernet
    if use_fusion:
        assert arrays["attention"].shape == (3, 4, 2)
        np.testing.assert_allclose(arrays["attention"].sum(axis=-1), 1, atol=1e-6)
    assert not encoder._forward_hooks
    assert not encoder.hypernet._forward_hooks
    assert summarize_probe(arrays)["frames"] == 3
    with torch.inference_mode():
        reconstructed = reconstruct_condition(encoder, torch.from_numpy(arrays["base_features"]),
            torch.from_numpy(arrays["gamma"]), torch.from_numpy(arrays["beta"]), observations["state"])
    torch.testing.assert_close(reconstructed, expected, rtol=1e-6, atol=1e-6)


def test_frame_selection_preserves_episode_boundaries() -> None:
    """Sampling cannot cross episode boundaries or duplicate short-episode frames."""
    indices, episodes, steps = select_episode_frames(np.array([3, 8]), [1, 0], 4)
    np.testing.assert_array_equal(indices, [3, 4, 5, 7, 0, 1, 2])
    np.testing.assert_array_equal(episodes, [1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(steps, [0, 1, 2, 4, 0, 1, 2])
    for requested in ([0, 0], [-1], [2], []):
        with pytest.raises(ValueError):
            select_episode_frames(np.array([3, 8]), requested, 4)
    with pytest.raises(ValueError):
        select_episode_frames(np.array([3, 2]), [0], 4)


def test_recorder_requires_eval_and_frames() -> None:
    """Missing data and training-mode dropout are rejected explicitly."""
    encoder = make_encoder()
    recorder = HyperNetProbeRecorder(encoder)
    with pytest.raises(ValueError, match="no encoder frames"):
        recorder.arrays()
    recorder.close()
    with pytest.raises(ValueError, match="evaluation-mode"):
        HyperNetProbeRecorder(encoder.train())


def test_temporal_probe_holds_features_fixed_and_uses_episode_means() -> None:
    """Feature variation alone cannot create apparent dynamic modulation."""
    arrays = {"base_features": np.ones((4, 2, 3)),
              "gamma": np.array([0.1, 0.3, 0.7, 0.7])[:, None, None] * np.ones((4, 2, 3)),
              "beta": np.zeros((4, 2, 3)), "episode": np.array([0, 0, 1, 1])}
    arrays["base_features"][3] *= 2
    ratio = temporal_modulation_ratio(arrays, 0.1)
    np.testing.assert_allclose(ratio[:2], 0.01)
    np.testing.assert_allclose(ratio[2:], 0, atol=1e-8)


def test_offline_reader_resizes_and_applies_saved_normalization(tmp_path: Path) -> None:
    """CPU probes must normalize RGB and state exactly once using saved scales."""
    import zarr

    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/probe_hypernet_demos.py"
    spec = importlib.util.spec_from_file_location("probe_hypernet_demos", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = make_encoder()
    data = zarr.open_group(str(tmp_path / "data.zarr"), mode="w")
    for key in encoder.rgb_keys:
        data.create_dataset(key, data=np.full((2, 12, 10, 3), 0.25, dtype=np.float32))
    data.create_dataset("state", data=np.full((2, 7), 3, dtype=np.float32))
    normalizers = {key: (torch.tensor([2.0]), torch.tensor([-1.0]))
                   for key in encoder.key_shape_map}
    observations = module.read_observations(data, np.array([1]), encoder, normalizers)
    assert observations["wrist"].shape == (1, 3, 8, 8)
    assert torch.all(observations["wrist"] == -0.5)
    assert torch.all(observations["state"] == 5)


def test_demo_plotter_writes_internal_signal_and_modulation_maps(tmp_path: Path) -> None:
    """Recorded arrays must produce actual plots, including the extra fusion branch."""
    path = Path(__file__).resolve().parents[1] / "scripts/manifeel/probe_hypernet_demos.py"
    spec = importlib.util.spec_from_file_location("probe_hypernet_demos_plot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoder = make_encoder(use_fusion_hypernet=True)
    recorder = HyperNetProbeRecorder(encoder)
    with torch.inference_mode():
        encoder({"wrist": torch.rand(3, 3, 8, 8),
                 "right_tactile_camera_taxim": torch.rand(3, 3, 8, 8),
                 "state": torch.rand(3, 7)})
    arrays = recorder.arrays()
    recorder.close()
    arrays.update({"episode": np.zeros(3, dtype=int), "episode_step": np.arange(3),
                   "normalized_state_norm": np.ones(3), "force_field_norm_mean": np.ones(3)})
    arrays["dynamic_residual_ratio"] = temporal_modulation_ratio(arrays, float(encoder.alpha.detach()))
    module.plot_episode(arrays, 0, tmp_path)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "episode0_signals.png", "episode0_modulation.png",
        "episode0_modulation_centered.png", "episode0_fusion_hypernet.png"}


def test_rollout_probe_saves_explicit_chunk_environment_history_axes(tmp_path: Path) -> None:
    """Flattened encoder frames must retain their original rollout layout."""
    from types import SimpleNamespace

    encoder = make_encoder()
    policy = SimpleNamespace(obs_encoder=encoder, n_obs_steps=2, n_action_steps=8)
    recorder = HyperNetRolloutRecorder(policy, tmp_path)
    with torch.inference_mode():
        for chunk_index in range(2):
            encoder({"wrist": torch.rand(6, 3, 8, 8),
                     "right_tactile_camera_taxim": torch.rand(6, 3, 8, 8),
                     "state": torch.rand(6, 7)})
    summary = recorder.save()
    recorder.close()
    assert summary["environments"] == 3
    with np.load(tmp_path / "hypernet_rollout_trace.npz") as trace:
        assert trace["gamma"].shape == (2, 3, 2, 2, 512)
        np.testing.assert_array_equal(trace["chunk_start_sim_step"], [0, 8])
    with np.load(tmp_path / "attention_trace.npz") as trace:
        assert trace["weights"].shape == (2, 3, 2, 4, 2)
