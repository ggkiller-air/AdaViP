import importlib.util
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "scripts" / "real_world" / "train_table2.py"
SPEC = importlib.util.spec_from_file_location("train_table2", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_TABLE2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_TABLE2)

SANITY_LAUNCHER_PATH = REPO_ROOT / "scripts" / "real_world" / "train_dp_sanity.py"
SANITY_SPEC = importlib.util.spec_from_file_location(
    "train_dp_sanity", SANITY_LAUNCHER_PATH
)
assert SANITY_SPEC is not None and SANITY_SPEC.loader is not None
TRAIN_DP_SANITY = importlib.util.module_from_spec(SANITY_SPEC)
SANITY_SPEC.loader.exec_module(TRAIN_DP_SANITY)


class Table2TrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = TRAIN_TABLE2.load_manifest(TRAIN_TABLE2.DEFAULT_MANIFEST)

    def test_manifest_blocks_incomplete_tasks(self) -> None:
        with self.assertRaisesRegex(ValueError, "not runnable"):
            TRAIN_TABLE2.require_entry(
                self.manifest, "vtdp", "puzzle_insertion", False
            )

    def test_manifest_requires_opt_in_for_reference_configs(self) -> None:
        with self.assertRaisesRegex(ValueError, "allow-reference-config"):
            TRAIN_TABLE2.require_entry(self.manifest, "rdp", "peeling", False)

        entry = TRAIN_TABLE2.require_entry(
            self.manifest, "rdp", "peeling", True
        )
        self.assertEqual(entry["rdp_at_profile"], "at_peel")

    def test_offline_overrides_remove_real_runner(self) -> None:
        overrides = TRAIN_TABLE2.common_overrides(
            dataset=Path("dataset.zarr"),
            output_dir=Path("output"),
            seed=7,
            wandb_mode="disabled",
            run_name="test_run",
            remove_env_runner=True,
        )

        self.assertIn("~task.env_runner", overrides)
        self.assertIn("training.rollout_every=0", overrides)

    def test_at_overrides_do_not_delete_absent_runner(self) -> None:
        overrides = TRAIN_TABLE2.common_overrides(
            dataset=Path("dataset.zarr"),
            output_dir=Path("output"),
            seed=7,
            wandb_mode="disabled",
            run_name="test_run",
            remove_env_runner=False,
        )

        self.assertNotIn("~task.env_runner", overrides)


class DpSanityTrainingTest(unittest.TestCase):
    def test_command_uses_plain_dp_and_no_tactile_config(self) -> None:
        with mock.patch.object(
            TRAIN_DP_SANITY.shutil, "which", return_value="/env/bin/accelerate"
        ):
            command = TRAIN_DP_SANITY.build_command(
                python=Path("/env/bin/python"),
                rdp_root=REPO_ROOT / "third_party" / "reactive_diffusion_policy",
                dataset=Path("/data/dataset"),
                output=Path("/data/output"),
                seed=7,
                wandb_mode="disabled",
                overrides=[],
            )

        self.assertIn("task=dp_sanity_no_tactile_30hz", command)
        self.assertNotIn("~task.env_runner", command)
        self.assertFalse(any("rdp" in value.lower() for value in command[4:]))

    def test_command_selects_left_arm_piper_config(self) -> None:
        with mock.patch.object(
            TRAIN_DP_SANITY.shutil, "which", return_value="/env/bin/accelerate"
        ):
            command = TRAIN_DP_SANITY.build_command(
                python=Path("/env/bin/python"),
                rdp_root=REPO_ROOT / "third_party" / "reactive_diffusion_policy",
                dataset=Path("/data/piper"),
                output=Path("/data/output"),
                seed=42,
                wandb_mode="disabled",
                overrides=[],
                task_config="piper_pick_cup_left_30hz",
            )

        self.assertIn("task=piper_pick_cup_left_30hz", command)
        self.assertIn("task.name=piper_pick_cup_left_seed42", command)
        self.assertIn("logging.project=piper_pick_cup_left", command)


if __name__ == "__main__":
    unittest.main()
