#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

print_manifeel_context
echo "[manifeel] branch: $(git -C "${MANIFEEL_REPO_ROOT}" branch --show-current)"
echo "[manifeel] git status:"
git -C "${MANIFEEL_REPO_ROOT}" status --short --branch --ignore-submodules=dirty
echo "[manifeel] submodules:"
git -C "${MANIFEEL_REPO_ROOT}" submodule status
echo "[manifeel] upstream commits:"
git -C "${MANIFEEL_ROOT}" rev-parse HEAD
git -C "${MANIFEEL_IGE_ROOT}" rev-parse HEAD
git -C "${MANIFEEL_DP_ROOT}" rev-parse HEAD

echo "[manifeel] proxy variables:"
env | grep -Ei '^(http|https|all|no)_proxy=' || true
echo "[manifeel] storage:"
df -h "${MANIFEEL_REPO_ROOT}" "${MANIFEEL_DATA_ROOT}" 2>/dev/null || true
echo "[manifeel] manual artifact: ${MANIFEEL_ISAACGYM_ROOT}"
if [[ -d "${MANIFEEL_ISAACGYM_ROOT}/isaacgym/python" ]]; then
    echo "[manifeel] Isaac Gym Python tree: present"
else
    echo "[manifeel] Isaac Gym Python tree: MISSING"
fi
echo "[manifeel] manual archive: ${MANIFEEL_ISAACGYM_ARCHIVE}"
if [[ -f "${MANIFEEL_ISAACGYM_ARCHIVE}" ]]; then
    echo "[manifeel] Isaac Gym archive: present ($(du -h "${MANIFEEL_ISAACGYM_ARCHIVE}" | cut -f1))"
else
    echo "[manifeel] Isaac Gym archive: MISSING"
fi
echo "[manifeel] dataset: ${MANIFEEL_DATASET_PATH}"
if [[ -d "${MANIFEEL_DATASET_PATH}" ]]; then
    echo "[manifeel] dataset directory: present"
else
    echo "[manifeel] dataset directory: MISSING"
fi

if [[ ! -x "${MANIFEEL_PYTHON}" ]]; then
    echo "[manifeel] isolated Python is not installed; imports skipped"
    exit 0
fi

activate_manifeel
"${MANIFEEL_PYTHON}" - <<'PY'
import importlib.util
import os
import sys

print("python", sys.version)
print("prefix", sys.prefix)
for name in ("torch", "hydra", "diffusion_policy", "manifeel", "isaacgym", "isaacgymenvs"):
    spec = importlib.util.find_spec(name)
    print(f"import {name}: {'present' if spec else 'MISSING'}")
if importlib.util.find_spec("torch"):
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
PY
