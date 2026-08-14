#!/usr/bin/env python3
"""Launch offline Table 2 DP or RDP training from the pinned RDP codebase."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "real_world" / "table2" / "manifest.json"
DEFAULT_RDP_ROOT = REPO_ROOT / "third_party" / "reactive_diffusion_policy"
DEFAULT_OUTPUT_ROOT = Path("/data/wangzihao/outputs/real_world/table2")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate the Table 2 training manifest."""
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    if not isinstance(manifest.get("methods"), dict) or not isinstance(
        manifest.get("tasks"), dict
    ):
        raise ValueError(f"Manifest must define methods and tasks: {path}")
    return manifest


def require_entry(
    manifest: dict[str, Any], method: str, task: str, allow_reference: bool
) -> dict[str, Any]:
    """Return a runnable task entry or explain why training is blocked."""
    methods = manifest["methods"]
    tasks = manifest["tasks"]
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(methods)}")
    if task not in tasks:
        raise ValueError(f"Unknown task {task!r}; choose from {', '.join(tasks)}")

    method_status = methods[method]["status"]
    if method_status == "pending_adavip_adapter":
        raise ValueError(
            f"{method} is not implemented yet; the AdaViP adapter must be added "
            "without changing the pinned RDP submodule"
        )

    entry = tasks[task]
    supported_methods = entry.get("supported_methods")
    if supported_methods is not None and method not in supported_methods:
        raise ValueError(
            f"{task} supports only {', '.join(supported_methods)}, not {method}"
        )
    protocol_status = entry["protocol_status"]
    if protocol_status == "pending_task_config":
        raise ValueError(f"{task} is not runnable: {entry['formal_blocker']}")
    if protocol_status == "upstream_reference" and not allow_reference:
        raise ValueError(
            f"{task} currently has only an upstream reference config: "
            f"{entry['formal_blocker']} Pass --allow-reference-config only for "
            "preflight/reference training."
        )
    return entry


def common_overrides(
    dataset: Path,
    output_dir: Path,
    seed: int,
    wandb_mode: str,
    run_name: str,
    remove_env_runner: bool,
) -> list[str]:
    """Build overrides shared by all offline training stages."""
    overrides = [
        f"task.dataset_path={dataset}",
        f"task.name={run_name}",
        f"training.seed={seed}",
        "training.rollout_every=0",
        f"logging.mode={wandb_mode}",
        f"logging.name={run_name}",
        f"hydra.run.dir={output_dir}",
    ]
    if remove_env_runner:
        overrides.append("~task.env_runner")
    return overrides


def task_overrides(task_entry: dict[str, Any], task_config: str) -> list[str]:
    """Build Hydra overrides needed to locate and select a task config."""
    overrides = []
    search_path = task_entry.get("config_search_path")
    if search_path:
        absolute_path = (REPO_ROOT / search_path).resolve()
        overrides.append(f"hydra.searchpath=[file://{absolute_path}]")
    overrides.append(f"task={task_config}")
    return overrides


def training_command(
    python: Path,
    rdp_root: Path,
    config_name: str,
    overrides: Sequence[str],
    use_accelerate: bool,
) -> list[str]:
    """Build one upstream training command."""
    train_py = rdp_root / "train.py"
    if use_accelerate:
        environment_accelerate = python.parent / "accelerate"
        accelerate = (
            str(environment_accelerate)
            if environment_accelerate.is_file()
            else shutil.which("accelerate")
        )
        if accelerate is None:
            raise FileNotFoundError(
                f"accelerate was not found beside {python} or on PATH"
            )
        return [
            accelerate,
            "launch",
            str(train_py),
            f"--config-name={config_name}",
            *overrides,
        ]
    return [str(python), str(train_py), f"--config-name={config_name}", *overrides]


def run(command: Sequence[str], cwd: Path, dry_run: bool) -> None:
    """Print and optionally execute a training command."""
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "at", "ldp"), default="all")
    parser.add_argument("--at-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rdp-root", type=Path, default=DEFAULT_RDP_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--allow-reference-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest.resolve())
        task_entry = require_entry(
            manifest, args.method, args.task, args.allow_reference_config
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.method == "vtdp" and args.stage != "all":
        print("error: --stage is only valid for RDP", file=sys.stderr)
        return 2
    if not args.dry_run and not args.dataset.exists():
        print(f"error: dataset does not exist: {args.dataset}", file=sys.stderr)
        return 2
    if not (args.rdp_root / "train.py").is_file():
        print(f"error: RDP submodule is unavailable: {args.rdp_root}", file=sys.stderr)
        return 2

    dataset = args.dataset.resolve()
    output_base = args.output_root.resolve() / args.task / args.method / f"seed_{args.seed}"

    if args.method == "vtdp":
        run_name = f"table2_{args.task}_vtdp_seed{args.seed}"
        overrides = common_overrides(
            dataset,
            output_base,
            args.seed,
            args.wandb_mode,
            run_name,
            remove_env_runner=task_entry.get("remove_env_runner", True),
        )
        overrides.extend(task_entry.get("training_overrides", []))
        overrides.extend(args.overrides)
        command = training_command(
            args.python,
            args.rdp_root,
            "train_diffusion_unet_real_image_workspace",
            [*task_overrides(task_entry, task_entry["dp_task_config"]), *overrides],
            use_accelerate=True,
        )
        run(command, args.rdp_root, args.dry_run)
        return 0

    at_output = output_base / "at"
    ldp_output = output_base / "ldp"
    at_checkpoint = args.at_checkpoint.resolve() if args.at_checkpoint else at_output / "checkpoints" / "latest.ckpt"

    if args.stage in ("all", "at"):
        at_run_name = f"table2_{args.task}_rdp_at_seed{args.seed}"
        overrides = common_overrides(
            dataset,
            at_output,
            args.seed,
            args.wandb_mode,
            at_run_name,
            remove_env_runner=False,
        )
        overrides.extend(args.overrides)
        command = training_command(
            args.python,
            args.rdp_root,
            "train_at_workspace",
            [
                f"task={task_entry['rdp_at_task_config']}",
                f"at={task_entry['rdp_at_profile']}",
                *overrides,
            ],
            use_accelerate=False,
        )
        run(command, args.rdp_root, args.dry_run)

    if args.stage in ("all", "ldp"):
        if not args.dry_run and not at_checkpoint.is_file():
            print(f"error: AT checkpoint does not exist: {at_checkpoint}", file=sys.stderr)
            return 2
        ldp_run_name = f"table2_{args.task}_rdp_ldp_seed{args.seed}"
        overrides = common_overrides(
            dataset,
            ldp_output,
            args.seed,
            args.wandb_mode,
            ldp_run_name,
            remove_env_runner=True,
        )
        overrides.extend(args.overrides)
        command = training_command(
            args.python,
            args.rdp_root,
            "train_latent_diffusion_unet_real_image_workspace",
            [
                f"task={task_entry['rdp_ldp_task_config']}",
                f"at={task_entry['rdp_at_profile']}",
                f"at_load_dir={at_checkpoint}",
                *overrides,
            ],
            use_accelerate=True,
        )
        run(command, args.rdp_root, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
