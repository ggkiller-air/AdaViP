"""Check that the three Power Plug baselines retain the upstream DP protocol."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs/manifeel"
TASK_KEYS = {
    "vision": ("vision_wrist", {"wrist", "state"}),
    "tacrgb": ("vistac_wrist", {"wrist", "right_tactile_camera_taxim", "state"}),
    "tacff": ("visff_wrist", {"wrist", "tactile_force_field_right", "state"}),
}
EXPECTED_EPOCHS = {"vision": 1000, "tacrgb": 400, "tacff": 1000}
EXPECTED_CHECKPOINT_EVERY = {"vision": 100, "tacrgb": 50, "tacff": 100}


@pytest.mark.parametrize("method", TASK_KEYS)
def test_power_plug_upstream_config_composes(method: str) -> None:
    hydra = pytest.importorskip("hydra")
    profile = OmegaConf.load(CONFIG_ROOT / f"power_plug_{method}.yaml")
    task, keys = TASK_KEYS[method]
    assert profile.task == task
    assert profile.isaacgym_cfg_name == "isaacgym_config_power_plug.yaml"
    assert profile.dataset_path == "/data/wangzihao/datasets/manifeel/plug_quan_Aug02"
    assert profile.num_demos == 50
    assert profile.num_epochs == EXPECTED_EPOCHS[method]
    assert profile.checkpoint_every == EXPECTED_CHECKPOINT_EVERY[method]
    assert profile.rollout_every == 0
    expected_suffix = "_ep400" if method == "tacrgb" else ""
    assert profile.run_name == f"dp_power_plug_{method}_batch8{expected_suffix}_seed42"
    assert profile.batch_size == 8
    assert profile.num_workers == 8
    assert profile.val_batch_size == 8
    assert profile.val_num_workers == 2

    config_dir = str(REPO_ROOT / "third_party/manifeel/manifeel/config")
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(
            config_name="train_diffusion_workspace",
            overrides=[
                f"task={profile.task}",
                f"dataset_path={profile.dataset_path}",
                f"isaacgym_cfg_name={profile.isaacgym_cfg_name}",
                f"task.dataset.max_train_episodes={profile.num_demos}",
            ],
        )
    assert cfg.policy._target_ == (
        "diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy"
    )
    assert set(cfg.shape_meta.obs) == keys
    assert cfg.shape_meta.action.shape == [6]
    assert cfg.task.dataset.zarr_path == profile.dataset_path
    assert cfg.task.env_runner.isaacgym_cfg_name == profile.isaacgym_cfg_name


def test_power_plug_batch_script_avoids_node_selection() -> None:
    script = (REPO_ROOT / "slurm/manifeel/train_power_plug_baseline.sbatch").read_text()
    assert "--gres=gpu:1" in script
    assert "--cpus-per-task=64" in script
    assert "--mem=512G" in script
    for option in ("--nodelist", "--exclude", "--constraint"):
        assert option not in script


def test_power_plug_tacrgb_ep400_script_avoids_node_selection() -> None:
    script = (
        REPO_ROOT / "slurm/manifeel/train_power_plug_tacrgb_ep400.sbatch"
    ).read_text()
    assert "--gres=gpu:1" in script
    assert "train_power_plug_baseline.py tacrgb" in script
    for option in ("--nodelist", "--exclude", "--constraint"):
        assert option not in script


def test_power_plug_tacrgb_ep400_launcher_dry_run() -> None:
    result = subprocess.run(
        [
            "/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python",
            str(REPO_ROOT / "scripts/manifeel/train_power_plug_baseline.py"),
            "tacrgb",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "NUM_EPOCHS=400" in result.stdout
    assert "CHECKPOINT_EVERY=50" in result.stdout
    assert "batch_size=8" in result.stdout
    assert "RetainedDiffusionUnetImageWorkspace" in result.stdout


def test_power_plug_workspace_keeps_periodic_and_final_checkpoints(monkeypatch) -> None:
    workspace_module = pytest.importorskip("adavip.manifeel.retained_dp_workspace")
    workspace = object.__new__(workspace_module.RetainedDiffusionUnetImageWorkspace)
    workspace.cfg = OmegaConf.create({"training": {"num_epochs": 1000}})
    workspace.epoch = 0
    saved = []

    def upstream_run(self):
        self.epoch = 1000

    def upstream_save(self, *args, **kwargs):
        saved.append(kwargs)
        return "checkpoint"

    monkeypatch.setattr(workspace_module.TrainDiffusionUnetImageWorkspace, "run", upstream_run)
    monkeypatch.setattr(
        workspace_module.TrainDiffusionUnetImageWorkspace,
        "save_checkpoint",
        upstream_save,
    )
    workspace.run()
    assert saved == [{"epoch": 1000, "use_thread": False, "prune": False}]
    workspace.save_checkpoint(epoch=100)
    assert saved[-1] == {"epoch": 100, "prune": False}
