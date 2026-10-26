"""CPU regression checks for portable legacy Stage2 checkpoint evaluation."""

from __future__ import annotations

from pathlib import Path

import dill
import pytest
import torch
from omegaconf import OmegaConf

from adavip.manifeel.full_sensor_evaluation import load_evaluation_policy
from adavip.manifeel.full_sensor_schema import make_shape_meta
from test_stage2_policy import make_batch, make_policy
from test_stage2_mlp_policy import make_mlp_policy


def small_config() -> object:
    """Retain real Stage2 fusion while using a small image backbone and U-Net."""
    shape_meta = make_shape_meta((3, 8, 8), (3, 8, 8), language_dim=8)
    return OmegaConf.create({
        "shape_meta": shape_meta, "n_obs_steps": 2, "n_action_steps": 8,
        "task_embedding_path": None,
        "policy": {
            "_target_": "unused",
            "obs_encoder": {
                "_target_": "adavip.manifeel.full_sensor_obs_encoder.FullSensorObsEncoder",
                "shape_meta": shape_meta, "image_feature_dim": 8, "use_group_norm": False,
                "rgb_model": {"_target_": "test_stage2_policy.TinyImageEncoder"}},
            "noise_scheduler": {"_target_": "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
                                "num_train_timesteps": 8, "prediction_type": "epsilon"},
            "horizon": 16, "n_obs_steps": 2, "n_action_steps": 8, "num_inference_steps": 8,
            "down_dims": [16, 32], "diffusion_step_embed_dim": 16, "n_groups": 8}})


@pytest.mark.parametrize("variant", ["transformer", "mlp"])
@pytest.mark.parametrize("weights", ["ema", "raw"])
def test_legacy_checkpoint_restores_exact_policy_without_teacher(
    tmp_path: Path, variant: str, weights: str,
) -> None:
    """Restore real fusion, RMS and normalizers with no dataset or Teacher access."""
    original = (make_policy() if variant == "transformer" else make_mlp_policy()[0]).eval()
    original.utility_rms.copy_(torch.arange(1, 10))
    state = original.state_dict()
    other = {key: value.clone() for key, value in state.items()}
    other["utility_predictor.0.bias"].add_(1)
    contract = {"teacher": "/unavailable/teacher.ckpt", "cache": "/unavailable/cache"}
    if variant == "mlp":
        contract["variant"] = "stage2_mlp_fusion"
    checkpoint = tmp_path / "epoch20.ckpt"
    torch.save({"model": state if weights == "raw" else other,
                "ema": state if weights == "ema" else other,
                "contract": contract, "epoch": 21, "global_step": 100},
               checkpoint, pickle_module=dill)
    config = tmp_path / "architecture.yaml"
    OmegaConf.save(small_config(), config)
    restored, cfg, metadata = load_evaluation_policy(checkpoint, weights=weights, config_path=config)
    assert type(restored) is type(original)
    assert not restored.training
    assert not any(parameter.requires_grad for parameter in restored.parameters())
    assert metadata["weights"] == weights
    assert metadata["epoch"] == 21
    assert cfg.n_obs_steps == 2
    for key, value in state.items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    batch = make_batch()
    with torch.no_grad():
        old = original.predict_action(batch["obs"], generator=torch.Generator().manual_seed(12))
        new = restored.predict_action(batch["obs"], generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(new["action"], old["action"])


def test_stage2_rejects_missing_ema_and_conflicting_variant(tmp_path: Path) -> None:
    """Never silently use raw weights or instantiate the wrong architecture."""
    checkpoint = tmp_path / "bad.ckpt"
    state = {"cross_attention.in_proj_weight": torch.ones(1)}
    torch.save({"model": state, "contract": {}}, checkpoint, pickle_module=dill)
    with pytest.raises(ValueError, match="silent weights fallback"):
        load_evaluation_policy(checkpoint)
    torch.save({"model": state, "ema": state, "contract": {"variant": "stage2_mlp_fusion"}},
               checkpoint, pickle_module=dill)
    with pytest.raises(ValueError, match="disagrees"):
        load_evaluation_policy(checkpoint)


def test_batch_evaluates_both_variants_and_forwards_rollout_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run both model types in isolation and retain their official score reports."""
    import json
    import sys

    from scripts.manifeel import eval_stage2_checkpoints as batch

    checkpoints = tmp_path / "checkpoints"
    for variant in ("transformer", "mlp"):
        path = checkpoints / variant / "checkpoints/epoch20.ckpt"
        path.parent.mkdir(parents=True)
        path.touch()
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["batch", "--checkpoint-root", str(checkpoints),
                        "--output-dir", str(output), "--epochs", "20",
                        "--n-test", "50", "--task-embedding-path", "/portable/embeddings.npz"])
    calls = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        folder = Path(command[command.index("--output-dir") + 1])
        folder.mkdir()
        (folder / "evaluation.json").write_text(json.dumps({"test/macro_mean_score": 0.5}))

    monkeypatch.setattr(batch.subprocess, "run", run)
    batch.main()
    reports = json.loads((output / "summary.json").read_text())
    assert set(reports) == {"transformer_epoch20", "mlp_epoch20"}
    assert len(calls) == 2
    assert all(command[command.index("--n-test") + 1] == "50" for command in calls)
    assert all(command[command.index("--task-embedding-path") + 1] == "/portable/embeddings.npz"
               for command in calls)
