#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy

RUN_ROOT="${FULL_SENSOR_RUN_ROOT:-$(cat "${MANIFEEL_OUTPUT_ROOT}/slurm/full_sensor_batch64_current_run.txt")}"
ARCHIVE_ROOT="${FULL_SENSOR_UPLOAD_ROOT:-${MANIFEEL_OUTPUT_ROOT}/checkpoint_archive/full_sensor_$(date +%Y%m%d_%H%M%S)}"
REPO_ID="${MODELSCOPE_REPO_ID:-ggkiller/Manifeel-ckpts1018}"
REMOTE_PATH="${MODELSCOPE_REMOTE_PATH:-full_sensor_stage0_batch64_20261018}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$(command -v modelscope)}"

"${MODELSCOPE_BIN}" whoami
"${MODELSCOPE_BIN}" info --repo-type model "${REPO_ID}" >/dev/null
mkdir -p "${ARCHIVE_ROOT}/checkpoints"
for name in latest epoch18; do
    source_name="${name}"
    if [[ "${name}" == epoch18 ]]; then
        source_name=best
    fi
    if [[ -e "${ARCHIVE_ROOT}/checkpoints/${name}.ckpt" ]]; then
        echo "Refusing to overwrite frozen checkpoint: ${ARCHIVE_ROOT}/checkpoints/${name}.ckpt" >&2
        exit 2
    fi
    cp --reflink=auto -- "${RUN_ROOT}/checkpoints/${source_name}.ckpt" \
        "${ARCHIVE_ROOT}/checkpoints/${name}.ckpt"
done

"${MANIFEEL_PYTHON}" - "${RUN_ROOT}" "${ARCHIVE_ROOT}" <<'PY'
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import dill
from omegaconf import OmegaConf
import torch

run, archive = map(Path, sys.argv[1:])
manifest = {"frozen_at": datetime.now().astimezone().isoformat(),
            "source_run": str(run), "checkpoints": {}}
for name in ["latest", "epoch18"]:
    path = archive / "checkpoints" / f"{name}.ckpt"
    payload = torch.load(str(path), map_location="cpu", pickle_module=dill, mmap=True)
    counters = {key: dill.loads(payload["pickles"][key])
                for key in ["epoch", "global_step", "batch_in_epoch", "best_validation_loss"]}
    if name == "epoch18" and counters["epoch"] != 18:
        raise ValueError(f"best.ckpt is not epoch18: {counters}")
    if not all(key in payload["state_dicts"] for key in ["model", "ema_model", "optimizer"]):
        raise ValueError(f"Not a full EMA workspace checkpoint: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    counters.update(size_bytes=path.stat().st_size, sha256=digest.hexdigest())
    manifest["checkpoints"][name] = counters
    if name == "latest":
        OmegaConf.save(payload["cfg"], archive / "training_config.yaml", resolve=True)
        shutil.copyfile(payload["cfg"].task_embedding_path, archive / "task_embeddings.npz")
    del payload
(archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(archive / "README.md").write_text(
    "# Stage0 full-sensor DP comparison\n\n"
    "3 cameras + 2 TacRGB + proprioception + language; batch64, seed42.\n"
    "`checkpoints/latest.ckpt` is a frozen snapshot, not a moving remote pointer.\n"
    "`checkpoints/epoch18.ckpt` is the best full-input EMA validation checkpoint.\n"
    "See manifest.json for exact counters, byte sizes and SHA256.\n"
    "Both are full workspaces with raw/EMA weights, optimizer and normalizer.\n"
    "Load trusted checkpoints only. Use EMA for simulator evaluation.\n"
    "No dataset or simulator assets are included. Validation loss is not success rate.\n"
    "\nCode: ggkiller-air/AdaViP, branch main.\n"
)
print(json.dumps(manifest, indent=2), flush=True)
PY

echo "Frozen upload directory: ${ARCHIVE_ROOT}"
"${MODELSCOPE_BIN}" upload --repo-type model --revision master \
    --commit-message "Upload frozen Stage0 latest and epoch18 full workspaces" \
    --max-workers 2 --use-cache --disable-tqdm \
    "${REPO_ID}" "${ARCHIVE_ROOT}" "${REMOTE_PATH}"
echo "Upload complete: ${REPO_ID}/${REMOTE_PATH}"
