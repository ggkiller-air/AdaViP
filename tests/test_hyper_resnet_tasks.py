"""Checks for the eight single-task HyperResNet fusion profiles."""

from __future__ import annotations

from pathlib import Path
import subprocess

from scripts.manifeel.train_hyper_resnet_tasks import TASKS, load_task


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python"


def test_all_non_power_plug_profiles_use_the_shared_protocol() -> None:
    assert len(TASKS) == 8
    for task_id in TASKS:
        cfg = load_task(task_id)
        assert cfg.task == "vistac_wrist"
        assert cfg.use_fusion is True
        assert cfg.use_fusion_hypernet is False
        assert cfg.num_epochs == 301
        assert cfg.checkpoint_every == 100
        assert cfg.batch_size == cfg.val_batch_size == 8
        assert "batch8_ep301" in cfg.run_name


def test_task_profile_dry_runs_include_encoder_and_action_overrides() -> None:
    for task_id, task in TASKS.items():
        result = subprocess.run(
            [PYTHON, str(REPO_ROOT / "scripts/manifeel/train_hyper_resnet_tasks.py"), task_id, "--dry-run"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "HyperResNetObsEncoder" in result.stdout
        assert "+policy.obs_encoder.use_fusion=true" in result.stdout
        assert "+policy.obs_encoder.use_fusion_hypernet=false" in result.stdout
        assert f"task.shape_meta.action.shape=[{task['action_dim']}]" in result.stdout
        assert "MANIFEEL_NUM_EPOCHS=301" in result.stdout
        assert "MANIFEEL_CHECKPOINT_EVERY=100" in result.stdout


def test_shared_task_slurm_script_is_resource_only() -> None:
    script = (REPO_ROOT / "slurm/manifeel/train_hyper_resnet_task.sbatch").read_text()
    assert "--gres=gpu:1" in script
    assert "--cpus-per-task=64" in script
    assert "--mem=512G" in script
    for option in ("--nodelist", "--exclude", "--constraint"):
        assert option not in script
