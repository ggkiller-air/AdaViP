#!/usr/bin/env python3
"""Export a training workspace checkpoint as a compact inference artifact."""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
from typing import Any

import dill
import torch
from omegaconf import OmegaConf


ARTIFACT_FORMAT = "adavip.dp_sanity.inference.v1"
DEFAULT_CHECKPOINT = Path(
    "/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/seed_42/"
    "checkpoints/epoch=0190-train_loss=0.022.ckpt"
)
DEFAULT_OUTPUT = Path(
    "/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/inference/"
    "dp_sanity_no_tactile_ema.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--control-frequency-hz", type=float, default=30.0)
    parser.add_argument(
        "--action-semantics",
        default="policy action in configured dataset order; verify robot joint order and units",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolved_container(config: Any) -> Any:
    """Convert an OmegaConf node to plain, self-contained Python values."""
    return OmegaConf.to_container(config, resolve=True, enum_to_str=True)


def build_contract(
    cfg: Any, control_frequency_hz: float, action_semantics: str
) -> dict[str, Any]:
    """Build the deployment contract from the resolved training config."""
    shape_meta = resolved_container(cfg.task.shape_meta)
    observations = {
        key: list(attributes["shape"])
        for key, attributes in shape_meta["obs"].items()
    }
    action_dim = int(shape_meta["action"]["shape"][0])
    return {
        "control_frequency_hz": control_frequency_hz,
        "observation_history": int(cfg.n_obs_steps),
        "observations": observations,
        "image_color": "RGB",
        "image_dtype": "float32",
        "image_range": [0.0, 1.0],
        "action_shape": [int(cfg.n_action_steps), action_dim],
        "action_pred_shape": [int(cfg.horizon), action_dim],
        "action_semantics": action_semantics,
    }


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}")

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    print(f"[export] loading={args.checkpoint}", flush=True)
    with args.checkpoint.open("rb") as checkpoint_file:
        workspace_payload = torch.load(
            checkpoint_file,
            map_location="cpu",
            pickle_module=dill,
        )

    cfg = workspace_payload["cfg"]
    state_dicts = workspace_payload["state_dicts"]
    state_key = "ema_model" if bool(cfg.training.use_ema) else "model"
    if state_key not in state_dicts:
        raise KeyError(f"Checkpoint does not contain {state_key!r} weights")
    policy_state_dict = state_dicts[state_key]
    if not any(key.startswith("normalizer.") for key in policy_state_dict):
        raise RuntimeError("Policy state dict does not contain its normalizer")

    artifact = {
        "format": ARTIFACT_FORMAT,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_bytes": args.checkpoint.stat().st_size,
        "weights": state_key,
        "policy_config": resolved_container(cfg.policy),
        "policy_state_dict": policy_state_dict,
        "offline_validation_dataset_config": resolved_container(cfg.task.dataset),
        "contract": build_contract(
            cfg,
            control_frequency_hz=args.control_frequency_hz,
            action_semantics=args.action_semantics,
        ),
        "runtime": {
            "num_inference_steps": 8,
            "use_ema": state_key == "ema_model",
            "warmup_required": True,
        },
        "torch_version": torch.__version__,
    }
    state_bytes = sum(
        value.numel() * value.element_size()
        for value in policy_state_dict.values()
        if isinstance(value, torch.Tensor)
    )
    del workspace_payload, state_dicts, cfg

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.name}.{os.getpid()}.tmp"
    )
    try:
        torch.save(artifact, temporary_output)
        os.replace(temporary_output, args.output)
    finally:
        temporary_output.unlink(missing_ok=True)

    print(f"[export] output={args.output}", flush=True)
    print(f"[export] tensor_bytes={state_bytes}", flush=True)
    print(f"[export] artifact_bytes={args.output.stat().st_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
