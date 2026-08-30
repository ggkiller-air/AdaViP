#!/usr/bin/env python3
"""Launch the no-tactile, single-episode DP sanity training run."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RDP_ROOT = REPO_ROOT / "third_party" / "reactive_diffusion_policy"
DEFAULT_DATASET = Path(
    "/data/wangzihao/datasets/real_world/dp_sanity_no_tactile/processed_30hz"
)
DEFAULT_OUTPUT = Path(
    "/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/seed_42"
)
CONFIG_SEARCH_PATH = REPO_ROOT / "configs" / "real_world" / "dp_sanity"
DEFAULT_TASK_CONFIG = "dp_sanity_no_tactile_30hz"


def build_command(
    python: Path,
    rdp_root: Path,
    dataset: Path,
    output: Path,
    seed: int,
    wandb_mode: str,
    overrides: Sequence[str],
    task_config: str = DEFAULT_TASK_CONFIG,
    num_processes: int = 1,
) -> list[str]:
    """Build the upstream DP command with the local offline task config."""
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")
    environment_accelerate = python.parent / "accelerate"
    accelerate = (
        str(environment_accelerate)
        if environment_accelerate.is_file()
        else shutil.which("accelerate")
    )
    if accelerate is None:
        raise FileNotFoundError("accelerate is not available in the DP environment")

    task_name = task_config[:-5] if task_config.endswith("_30hz") else task_config
    run_name = f"{task_name}_seed{seed}"
    command = [accelerate, "launch"]
    if num_processes > 1:
        command.extend(["--multi_gpu", "--num_processes", str(num_processes)])
    command.extend([
        str(rdp_root / "train.py"),
        "--config-name=train_diffusion_unet_real_image_workspace",
        f"hydra.searchpath=[file://{CONFIG_SEARCH_PATH.resolve()}]",
        f"task={task_config}",
        f"task.dataset_path={dataset.resolve()}",
        f"task.name={run_name}",
        f"training.seed={seed}",
        "training.rollout_every=0",
        "training.num_epochs=100",
        "training.checkpoint_every=10",
        "training.sample_every=10",
        "dataloader.batch_size=32",
        "dataloader.num_workers=4",
        "val_dataloader.batch_size=32",
        "val_dataloader.num_workers=1",
        f"logging.mode={wandb_mode}",
        f"logging.project={task_name}",
        f"logging.name={run_name}",
        f"hydra.run.dir={output.resolve()}",
        *overrides,
    ])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="offline"
    )
    parser.add_argument("--rdp-root", type=Path, default=DEFAULT_RDP_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and not (args.dataset / "replay_buffer.zarr").is_dir():
        print(f"error: converted dataset is unavailable: {args.dataset}", file=sys.stderr)
        return 2
    if not (args.rdp_root / "train.py").is_file():
        print(f"error: RDP submodule is unavailable: {args.rdp_root}", file=sys.stderr)
        return 2
    try:
        command = build_command(
            python=args.python,
            rdp_root=args.rdp_root,
            dataset=args.dataset,
            output=args.output,
            seed=args.seed,
            wandb_mode=args.wandb_mode,
            overrides=args.overrides,
            task_config=args.task_config,
            num_processes=args.num_processes,
        )
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=args.rdp_root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
