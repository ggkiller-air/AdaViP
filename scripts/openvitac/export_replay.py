#!/usr/bin/env python3
"""Export small numeric traces for independent controller/clock checks on the 3090."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = (args.root / "manifest.json").read_text()
    manifest = json.loads(content)
    marker = json.loads((args.root / "COMPLETE.json").read_text())
    if hashlib.sha256(content.encode()).hexdigest() != marker["manifest_sha256"]:
        raise ValueError("Converted manifest is incomplete or changed")
    task = next(t for t in manifest["tasks"] if t["task_id"] == args.task)
    directory = args.root / task["directory"]
    episodes = json.loads((directory / "episodes.json").read_text())
    episode = episodes[args.episode]
    selection = slice(episode["start"], episode["stop"])
    values = {key: np.load(directory / (key + ".npy"), mmap_mode="r")[selection]
              for key in ("state", "action", "action_valid", "timestamp", "source_time", "eef_pose", "finger_position")}
    values["instruction"] = np.array(manifest["language_texts"][episode["language_id"]])
    values["contract"] = np.array(json.dumps({"source_revision": manifest["source_revision"],
        "manifest_sha256": marker["manifest_sha256"], "task": args.task, "episode": args.episode,
        "action_codec": manifest["action_codec"], "source_step_gaps": episode["source_step_gaps"],
        "warning": "Derived state targets. Scene seeds/object initial poses are not supplied here. Validate controller timing before contact replay."}))
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **values)
    print(json.dumps({"output": str(args.output), "frames": len(values["state"]),
                      "valid_targets": int(values["action_valid"].sum())}))


if __name__ == "__main__":
    main()
