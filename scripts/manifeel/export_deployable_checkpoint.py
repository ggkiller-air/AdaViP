#!/usr/bin/env python3
"""Convert a full training workspace checkpoint to a deployment artifact.

The exported file keeps the EMA policy, its normalizer, and resolved configs,
while dropping optimizer, scheduler, and other training-only state.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import dill
import torch
from omegaconf import OmegaConf


# Preserve the established DP inference artifact identifier for compatibility
# with existing deployment-side loaders.
ARTIFACT_FORMAT = "adavip.dp_sanity.inference.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--control-frequency-hz", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_container(value: Any) -> Any:
    """Convert an OmegaConf value into plain resolved Python containers."""
    return OmegaConf.to_container(value, resolve=True, enum_to_str=True)


def tensor_bytes(state_dict: dict[str, Any]) -> int:
    """Return the serialized tensor storage size represented by a state dict."""
    return sum(
        value.numel() * value.element_size()
        for value in state_dict.values()
        if isinstance(value, torch.Tensor)
    )


def build_contract(cfg: Any, label: str, control_frequency_hz: float | None) -> dict[str, Any]:
    """Describe the observation and action tensors expected by the policy."""
    shape_meta = resolved_container(cfg.shape_meta)
    return {
        "label": label,
        "control_frequency_hz": control_frequency_hz,
        "observation_steps": int(cfg.n_obs_steps),
        "horizon": int(cfg.horizon),
        "action_steps": int(cfg.n_action_steps),
        "observations": {
            key: {
                "shape": list(attributes["shape"]),
                "type": attributes.get("type", "low_dim"),
            }
            for key, attributes in shape_meta["obs"].items()
        },
        "action_shape": list(shape_meta["action"]["shape"]),
    }


def export_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Load, sanitize, and atomically write one deployment checkpoint."""
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    print(f"[export] loading={checkpoint}", flush=True)
    with checkpoint.open("rb") as stream:
        payload = torch.load(stream, map_location="cpu", pickle_module=dill)

    cfg = payload["cfg"]
    state_dicts = payload["state_dicts"]
    use_ema = bool(OmegaConf.select(cfg, "training.use_ema", default=False))
    state_key = "ema_model" if use_ema and "ema_model" in state_dicts else "model"
    if state_key not in state_dicts:
        raise KeyError(f"Checkpoint does not contain {state_key!r} weights")

    policy_state = dict(state_dicts[state_key])
    # Some upstream EMA implementations create the EMA copy before normalizer
    # propagation. Preserve the trained normalizer when that happens.
    normalizer_source = state_key
    if not any(key.startswith("normalizer.") for key in policy_state):
        model_state = state_dicts.get("model", {})
        normalizer = {
            key: value
            for key, value in model_state.items()
            if key.startswith("normalizer.")
        }
        if not normalizer:
            raise RuntimeError("Checkpoint policy state does not contain a normalizer")
        policy_state.update(normalizer)
        normalizer_source = "model_fallback"

    policy_config = resolved_container(cfg.policy)
    dataset_config = resolved_container(cfg.task.dataset)
    artifact = {
        "format": ARTIFACT_FORMAT,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_bytes": checkpoint.stat().st_size,
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "label": args.label,
        "weights": state_key,
        "normalizer_source": normalizer_source,
        "policy_config": policy_config,
        "policy_state_dict": policy_state,
        "offline_validation_dataset_config": dataset_config,
        "contract": build_contract(cfg, args.label, args.control_frequency_hz),
        "runtime": {
            "use_ema": state_key == "ema_model",
            "warmup_required": True,
            "num_inference_steps": int(
                policy_config.get("num_inference_steps", 8)
            ),
        },
        "training_state": {
            "epoch": int(payload.get("epoch", -1)),
            "global_step": int(payload.get("global_step", -1)),
        },
        "torch_version": torch.__version__,
    }
    artifact_bytes = tensor_bytes(policy_state)
    del payload, state_dicts, cfg, policy_state

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(artifact, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "artifact": output.name,
        "artifact_bytes": output.stat().st_size,
        "artifact_sha256": sha256_file(output),
        "format": ARTIFACT_FORMAT,
        "label": args.label,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": artifact["source_checkpoint_sha256"],
        "weights": state_key,
        "normalizer_source": normalizer_source,
        "tensor_bytes": artifact_bytes,
        "contract": artifact["contract"],
    }
    if args.manifest_output:
        manifest_output = args.manifest_output.resolve()
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"[export] output={output} artifact_bytes={output.stat().st_size} "
        f"tensor_bytes={artifact_bytes}",
        flush=True,
    )
    return manifest


def main() -> int:
    args = parse_args()
    export_checkpoint(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
