#!/usr/bin/env bash
set -euo pipefail
OPENVITAC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPENVITAC_PYTHON="${OPENVITAC_PYTHON:-/public/home/wangzihao/.venvs/openvitac/bin/python}"
export OPENVITAC_DP_ROOT="${OPENVITAC_DP_ROOT:-/public/home/wangzihao/projects/AdaViP-main/third_party/diffusion_policy}"
export PYTHONPATH="${OPENVITAC_ROOT}:${OPENVITAC_DP_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
