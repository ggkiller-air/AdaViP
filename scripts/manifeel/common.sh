#!/usr/bin/env bash
set -euo pipefail

# Shared, explicit paths for the ManiFeel smoke scripts.
MANIFEEL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEEL_REPO_ROOT="$(cd -- "${MANIFEEL_SCRIPT_DIR}/../.." && pwd)"
MANIFEEL_ROOT="${MANIFEEL_REPO_ROOT}/third_party/manifeel"
MANIFEEL_IGE_ROOT="${MANIFEEL_REPO_ROOT}/third_party/manifeel-isaacgymenvs"
MANIFEEL_DP_ROOT="${MANIFEEL_REPO_ROOT}/third_party/diffusion_policy"

MANIFEEL_ENV_PREFIX="${MANIFEEL_ENV_PREFIX:-/public/home/wangzihao/.local/miniforge3/envs/manifeel}"
MANIFEEL_PYTHON="${MANIFEEL_PYTHON:-${MANIFEEL_ENV_PREFIX}/bin/python}"
MANIFEEL_DATA_ROOT="${MANIFEEL_DATA_ROOT:-/data/wangzihao/datasets/manifeel}"
MANIFEEL_CHECKPOINT_ROOT="${MANIFEEL_CHECKPOINT_ROOT:-/data/wangzihao/checkpoints/manifeel}"
MANIFEEL_OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"
MANIFEEL_ISAACGYM_ROOT="${MANIFEEL_ISAACGYM_ROOT:-${MANIFEEL_REPO_ROOT}/third_party/IsaacGym_Preview_TacSL_Package}"
MANIFEEL_ISAACGYM_ARCHIVE="${MANIFEEL_ISAACGYM_ARCHIVE:-${MANIFEEL_REPO_ROOT}/third_party/IsaacGym_Preview_TacSL_Package.tar.gz}"
MANIFEEL_DATASET_PATH="${MANIFEEL_DATASET_PATH:-${MANIFEEL_DATA_ROOT}/usb_quan_Aug05}"
MANIFEEL_DEVICE="${MANIFEEL_DEVICE:-cuda:0}"
MANIFEEL_HF_ENDPOINT="${MANIFEEL_HF_ENDPOINT:-https://hf-mirror.com}"

export MANIFEEL_REPO_ROOT MANIFEEL_ROOT MANIFEEL_IGE_ROOT MANIFEEL_DP_ROOT
export MANIFEEL_ENV_PREFIX MANIFEEL_PYTHON MANIFEEL_DATA_ROOT
export MANIFEEL_CHECKPOINT_ROOT MANIFEEL_OUTPUT_ROOT MANIFEEL_ISAACGYM_ROOT
export MANIFEEL_ISAACGYM_ARCHIVE
export MANIFEEL_DATASET_PATH MANIFEEL_DEVICE
export MANIFEEL_HF_ENDPOINT

export PYTHONUNBUFFERED=1

if [[ "${MANIFEEL_UNSET_PROXY:-0}" == "1" ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
fi

if [[ -n "${MANIFEEL_PROXY:-}" ]]; then
    export http_proxy="${MANIFEEL_PROXY}"
    export https_proxy="${MANIFEEL_PROXY}"
    export HTTP_PROXY="${MANIFEEL_PROXY}"
    export HTTPS_PROXY="${MANIFEEL_PROXY}"
fi

if [[ "${MANIFEEL_USE_TUNA:-0}" == "1" ]]; then
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
    export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
fi

# HF Hub data is downloaded through the configured mirror by default. Set
# MANIFEEL_HF_ENDPOINT=https://huggingface.co to use the upstream endpoint.
export HF_ENDPOINT="${HF_ENDPOINT:-${MANIFEEL_HF_ENDPOINT}}"

activate_manifeel() {
    if [[ ! -x "${MANIFEEL_PYTHON}" ]]; then
        echo "ManiFeel Python not found: ${MANIFEEL_PYTHON}" >&2
        echo "Run scripts/manifeel/setup_environment.sh first or set MANIFEEL_ENV_PREFIX." >&2
        return 1
    fi
    export PATH="$(dirname -- "${MANIFEEL_PYTHON}"):${PATH}"
    export CONDA_PREFIX="${MANIFEEL_ENV_PREFIX}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${MANIFEEL_ENV_PREFIX}/torch_extensions}"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    export PYTHONPATH="${MANIFEEL_ROOT}:${MANIFEEL_IGE_ROOT}:${MANIFEEL_DP_ROOT}:${PYTHONPATH:-}"
}

print_manifeel_context() {
    echo "[manifeel] repo=${MANIFEEL_REPO_ROOT}"
    echo "[manifeel] env=${MANIFEEL_ENV_PREFIX}"
    echo "[manifeel] python=${MANIFEEL_PYTHON}"
    echo "[manifeel] dataset=${MANIFEEL_DATASET_PATH}"
    echo "[manifeel] device=${MANIFEEL_DEVICE}"
    echo "[manifeel] hostname=$(hostname)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
    else
        echo "[manifeel] nvidia-smi not found"
    fi
}
