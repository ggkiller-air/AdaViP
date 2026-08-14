#!/usr/bin/env bash
set -euo pipefail

REAL_WORLD_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_WORLD_REPO_ROOT="$(cd -- "${REAL_WORLD_SCRIPT_DIR}/../.." && pwd)"
RDP_ROOT="${REAL_WORLD_REPO_ROOT}/third_party/reactive_diffusion_policy"

RDP_ENV_PREFIX="${RDP_ENV_PREFIX:-/public/home/wangzihao/envs/rdp-baseline-py310}"
RDP_PYTHON="${RDP_PYTHON:-${RDP_ENV_PREFIX}/bin/python}"
TABLE2_OUTPUT_ROOT="${TABLE2_OUTPUT_ROOT:-/data/wangzihao/outputs/real_world/table2}"

export REAL_WORLD_REPO_ROOT RDP_ROOT RDP_ENV_PREFIX RDP_PYTHON
export TABLE2_OUTPUT_ROOT PYTHONUNBUFFERED=1

activate_rdp() {
    if [[ ! -x "${RDP_PYTHON}" ]]; then
        echo "RDP Python not found: ${RDP_PYTHON}" >&2
        echo "Set RDP_ENV_PREFIX or RDP_PYTHON to the existing RDP environment." >&2
        return 1
    fi

    export PATH="$(dirname -- "${RDP_PYTHON}"):${PATH}"
    export CONDA_PREFIX="${RDP_ENV_PREFIX}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${REAL_WORLD_REPO_ROOT}:${RDP_ROOT}:${PYTHONPATH:-}"
}

print_rdp_training_context() {
    echo "[table2] repo=${REAL_WORLD_REPO_ROOT}"
    echo "[table2] upstream=${RDP_ROOT}"
    echo "[table2] env=${RDP_ENV_PREFIX}"
    echo "[table2] output_root=${TABLE2_OUTPUT_ROOT}"
    echo "[table2] hostname=$(hostname)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader || true
    else
        echo "[table2] nvidia-smi not found"
    fi
}

print_dp_sanity_context() {
    echo "[dp-sanity] repo=${REAL_WORLD_REPO_ROOT}"
    echo "[dp-sanity] upstream_dp=${RDP_ROOT}"
    echo "[dp-sanity] env=${RDP_ENV_PREFIX}"
    echo "[dp-sanity] hostname=$(hostname)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader || true
    else
        echo "[dp-sanity] nvidia-smi not found"
    fi
}
