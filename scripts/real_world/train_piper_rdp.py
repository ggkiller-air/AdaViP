#!/usr/bin/env python3
"""Launch local Piper AT and latent diffusion RDP training stages."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RDP_ROOT = REPO_ROOT / "third_party" / "reactive_diffusion_policy"
CONFIG_SEARCH_PATH = REPO_ROOT / "configs" / "real_world" / "piper_rdp"
DEFAULT_DATASET = Path(
    os.environ.get("ADAVIP_RDP_DATASET", REPO_ROOT / "datasets" / "piper_rdp")
)
DEFAULT_OUTPUT = Path(
    os.environ.get("ADAVIP_RDP_OUTPUT", REPO_ROOT / "outputs" / "piper_rdp")
)


def _accelerate(python: Path) -> str:
    environment_binary = python.parent / "accelerate"
    binary = (
        str(environment_binary)
        if environment_binary.is_file()
        else shutil.which("accelerate")
    )
    if binary is None:
        raise FileNotFoundError("accelerate is unavailable in the selected environment")
    return binary


def hydra_path_override(key: str, path: Path) -> str:
    """Quote a path so Hydra accepts filenames containing override syntax."""
    value = str(path.resolve())
    if "'" in value:
        raise ValueError(f"Hydra path values cannot contain a single quote: {value}")
    return f"{key}='{value}'"


def build_stage_command(
    stage: str,
    python: Path,
    rdp_root: Path,
    dataset: Path,
    output: Path,
    seed: int,
    wandb_mode: str,
    at_checkpoint: Path | None,
    overrides: Sequence[str],
) -> list[str]:
    """Build one upstream workspace command with local task/profile configs."""
    if stage not in {"at", "ldp"}:
        raise ValueError(f"Unsupported stage: {stage}")
    if stage == "ldp" and at_checkpoint is None:
        raise ValueError("LDP requires an AT checkpoint")
    config_name = (
        "train_at_workspace"
        if stage == "at"
        else "train_latent_diffusion_unet_real_image_workspace"
    )
    task_name = (
        "piper_pick_and_place_at_18_75hz"
        if stage == "at"
        else "piper_pick_and_place_ldp_18_75hz"
    )
    command = [
        _accelerate(python),
        "launch",
        str(rdp_root / "train.py"),
        f"--config-name={config_name}",
        f"hydra.searchpath=[file://{CONFIG_SEARCH_PATH.resolve()}]",
        f"task={task_name}",
        "at=piper_rdp_18_75hz",
        f"task.dataset_path={dataset.resolve()}",
        f"training.seed={seed}",
        "training.rollout_every=-1",
        f"logging.mode={wandb_mode}",
        f"logging.project=piper_rdp_{stage}",
        f"logging.name=piper_pick_and_place_{stage}_seed{seed}",
        f"hydra.run.dir={output.resolve()}",
    ]
    if stage == "ldp":
        command.extend(
            [
                hydra_path_override("at_load_dir", at_checkpoint),
            ]
        )
    command.extend(overrides)
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "at", "ldp"), default="all")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--at-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="offline"
    )
    parser.add_argument("--rdp-root", type=Path, default=DEFAULT_RDP_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and not (args.dataset / "replay_buffer.zarr").is_dir():
        raise FileNotFoundError(f"Prepared dataset is unavailable: {args.dataset}")
    if not (args.rdp_root / "train.py").is_file():
        raise FileNotFoundError(f"RDP source is unavailable: {args.rdp_root}")

    stages = ("at", "ldp") if args.stage == "all" else (args.stage,)
    at_checkpoint = args.at_checkpoint
    if args.stage == "all" and at_checkpoint is None:
        at_checkpoint = args.output / "at" / "checkpoints" / "latest.ckpt"
    environment = os.environ.copy()
    python_path = [str(REPO_ROOT), str(args.rdp_root)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)

    for stage in stages:
        stage_output = args.output / stage
        command = build_stage_command(
            stage=stage,
            python=args.python,
            rdp_root=args.rdp_root,
            dataset=args.dataset,
            output=stage_output,
            seed=args.seed,
            wandb_mode=args.wandb_mode,
            at_checkpoint=at_checkpoint,
            overrides=args.overrides,
        )
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        if stage == "ldp" and (at_checkpoint is None or not at_checkpoint.is_file()):
            raise FileNotFoundError(f"AT checkpoint is unavailable: {at_checkpoint}")
        subprocess.run(command, cwd=args.rdp_root, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
