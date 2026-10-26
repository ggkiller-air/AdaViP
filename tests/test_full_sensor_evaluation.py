"""CPU checks for Stage0 evaluation restoration and observation contracts."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("hydra")
import dill
from omegaconf import OmegaConf

from adavip.manifeel.full_sensor_evaluation import FullSensorEvaluationPolicy, load_evaluation_policy
from adavip.manifeel.full_sensor_schema import IMAGE_KEYS, make_shape_meta


class DummyPolicy(torch.nn.Module):
    """Expose normalizer state and capture simulator inputs without diffusion."""

    def __init__(self) -> None:
        super().__init__()
        self.normalizer = torch.nn.Linear(1, 1, bias=False)
        self.n_obs_steps = 2
        self.observations = None

    @property
    def device(self) -> torch.device:
        return self.normalizer.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.normalizer.weight.dtype

    def reset(self) -> None:
        """Reset a stateless dummy."""

    def predict_action(self, observations: dict) -> dict:
        """Record untouched images and return unified actions."""
        self.observations = observations
        return {"action": torch.ones(3, 8, 7)}


def test_adapter_preserves_images_and_injects_temporal_language() -> None:
    shape_meta = make_shape_meta(camera_shape=(3, 4, 4), tactile_shape=(3, 6, 4))
    policy = DummyPolicy()
    adapter = FullSensorEvaluationPolicy(policy, shape_meta, np.ones(512), 6)
    observations = {key: torch.rand(3, 2, *shape_meta["obs"][key]["shape"]) for key in IMAGE_KEYS}
    observations["state"] = torch.zeros(3, 2, 7)
    result = adapter.predict_action(observations)
    assert result["action"].shape == (3, 8, 6)
    assert policy.observations["task_embedding"].shape == (3, 2, 512)
    assert "task_embedding" not in observations
    assert policy.observations["front"] is observations["front"]
    del observations[IMAGE_KEYS[-1]]
    with pytest.raises(ValueError, match="missing"):
        adapter.predict_action(observations)


@pytest.mark.parametrize("weights,expected", [("ema", 2.0), ("raw", 1.0)])
def test_loader_selects_saved_normalizer_and_does_not_need_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights: str, expected: float,
) -> None:
    cfg = OmegaConf.create({"shape_meta": make_shape_meta(), "policy": {"_target_": "unused"}})
    payload = {"cfg": cfg, "state_dicts": {
        "model": {"normalizer.weight": torch.tensor([[1.0]])},
        "ema_model": {"normalizer.weight": torch.tensor([[2.0]])}},
        "pickles": {"epoch": dill.dumps(18)}}
    checkpoint = tmp_path / "trusted.ckpt"
    torch.save(payload, str(checkpoint), pickle_module=dill)
    monkeypatch.setattr("hydra.utils.instantiate", lambda settings: DummyPolicy())
    policy, restored_cfg, metadata = load_evaluation_policy(checkpoint, weights=weights)
    assert policy.normalizer.weight.item() == expected
    assert not policy.training
    assert not policy.normalizer.weight.requires_grad
    assert metadata == {"epoch": 18, "weights": weights}
    assert restored_cfg == cfg


def test_loader_refuses_missing_ema(tmp_path: Path) -> None:
    payload = {"cfg": OmegaConf.create({"shape_meta": make_shape_meta()}),
               "state_dicts": {"model": {}}}
    checkpoint = tmp_path / "raw_only.ckpt"
    torch.save(payload, str(checkpoint), pickle_module=dill)
    with pytest.raises(ValueError, match="silent weights fallback"):
        load_evaluation_policy(checkpoint)


def test_multitask_entrypoint_aggregates_mean_score_not_success_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import argparse
    import json

    from scripts.manifeel import eval_full_sensor_checkpoint as entrypoint
    from adavip.manifeel import task_protocol

    args = argparse.Namespace(
        checkpoint=tmp_path / "model.ckpt", output_dir=tmp_path / "evaluation",
        task_embedding_path=tmp_path / "embeddings.npz", device="cuda:0", weights="ema",
        task_id=None, n_test=50, n_test_vis=0, max_steps=500,
        test_start_seed=100000, policy_seed=42,
        config=None, check_only=False,
    )
    calls = []

    def run_task(command: list, **kwargs: object) -> None:
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        calls.append(command)
        (output / "evaluation.json").write_text(json.dumps(
            {"test/mean_score": 0.2 * len(calls), "test/success_rate": 0.0}))

    monkeypatch.setattr(entrypoint, "parse_args", lambda: args)
    monkeypatch.setattr(task_protocol, "DEFAULT_TASK_SPECS", task_protocol.DEFAULT_TASK_SPECS[:2])
    monkeypatch.setattr(entrypoint.subprocess, "run", run_task)
    entrypoint.main()
    report = json.loads((args.output_dir / "evaluation.json").read_text())
    assert report["test/macro_mean_score"] == pytest.approx(0.3)
    assert len(calls) == 2
    assert all(command[command.index("--test-start-seed") + 1] == "100000" for command in calls)
    assert all("--task-embedding-path" in command for command in calls)


def test_entrypoint_does_not_import_torch_before_isaacgym() -> None:
    import subprocess
    import sys

    entrypoint = Path(__file__).resolve().parents[1] / "scripts/manifeel/eval_full_sensor_checkpoint.py"
    subprocess.run([sys.executable, "-c",
                    "import runpy,sys; runpy.run_path(sys.argv[1], run_name='import_check'); "
                    "assert 'torch' not in sys.modules", str(entrypoint)], check=True, timeout=20)
