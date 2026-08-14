#!/usr/bin/env python3
"""Upload the Piper left-arm EMA artifact and metadata in one Hub commit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi


DEFAULT_ARTIFACT = Path(
    "/data/wangzihao/outputs/real_world/piper-pick-cup-left/inference/"
    "piper_pick_cup_left_ema.pt"
)
DEFAULT_REPO_ID = "ggkiller-air/piper-pick-cup-dp"
ARTIFACT_FORMAT = "adavip.dp_sanity.inference.v1"
CONTRACT = {
    "control_frequency_hz": 30.0,
    "observation_history": 2,
    "observations": {
        "cam_high": [3, 480, 640],
        "cam_left_wrist": [3, 480, 640],
        "qpos": [7],
    },
    "action_shape": [15, 7],
    "action_pred_shape": [16, 7],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(artifact: Path, sha256: str) -> dict[str, Any]:
    """Build machine-readable metadata for the exported policy."""
    return {
        "artifact": artifact.name,
        "artifact_bytes": artifact.stat().st_size,
        "format": ARTIFACT_FORMAT,
        "sha256": sha256,
        "weights": "ema_model",
        "training": {"epochs": 200, "seed": 42},
        "contract": CONTRACT,
    }


def build_model_card(manifest: dict[str, Any]) -> str:
    """Build the public model card stored beside the artifact."""
    return f"""---
tags:
- robotics
- diffusion-policy
- pytorch
---

# Piper Pick-Cup Left-Arm Diffusion Policy

Compact EMA inference artifact for the 30 Hz Piper pick-cup policy.

- File: `{manifest['artifact']}`
- Format: `{manifest['format']}`
- SHA-256: `{manifest['sha256']}`
- Inputs: two frames of `cam_high`, `cam_left_wrist`, and 7D left-arm `qpos`
- Output: 15 actions with six joints and one gripper dimension

Robot communication, synchronization, limits, and safety handling are not
included in this artifact.
"""


def main() -> int:
    args = parse_args()
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"EMA artifact does not exist: {artifact}")

    sha256 = sha256_file(artifact)
    manifest = build_manifest(artifact, sha256)
    model_card = build_model_card(manifest)

    api = HfApi(endpoint="https://huggingface.co")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    result = api.create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Publish Piper left-arm EMA policy",
        operations=[
            CommitOperationAdd(
                path_in_repo=artifact.name,
                path_or_fileobj=artifact,
            ),
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=model_card.encode("utf-8"),
            ),
            CommitOperationAdd(
                path_in_repo="artifact_manifest.json",
                path_or_fileobj=(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8"),
            ),
        ],
    )
    print(result.commit_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
