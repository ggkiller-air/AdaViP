#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

CONDA_EXE="${CONDA_EXE:-$(command -v mamba || command -v conda || true)}"
CONDA_PROFILE_EXE="${CONDA_PROFILE_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_EXE}" || -z "${CONDA_PROFILE_EXE}" ]]; then
    echo "conda is required; install Miniforge/Conda first." >&2
    exit 1
fi
CONDA_BASE="$("${CONDA_PROFILE_EXE}" info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Keep this environment's package cache separate from any shared/base Conda
# process. This avoids lock contention on shared Miniforge installations.
MANIFEEL_CONDA_PKGS_DIR="${MANIFEEL_CONDA_PKGS_DIR:-/data/wangzihao/cache/manifeel-conda-pkgs}"
mkdir -p "${MANIFEEL_CONDA_PKGS_DIR}"
export CONDA_PKGS_DIRS="${MANIFEEL_CONDA_PKGS_DIR}"

TORCH_VERSION="${MANIFEEL_TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${MANIFEEL_TORCHVISION_VERSION:-0.19.1}"
TORCH_INDEX_URL="${MANIFEEL_TORCH_INDEX_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu121}"
TORCH_WHEEL="${MANIFEEL_TORCH_WHEEL:-/public/home/wangzihao/.cache/manifeel-pip/wheels/torch-2.4.1+cu121-cp38-cp38-linux_x86_64.whl}"
PIP_INDEX="${MANIFEEL_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
CONDA_CHANNEL="${MANIFEEL_CONDA_CHANNEL:-conda-forge}"

print_manifeel_context

if [[ ! -d "${MANIFEEL_ISAACGYM_ROOT}/isaacgym/python" ]]; then
    if [[ -f "${MANIFEEL_ISAACGYM_ARCHIVE}" ]]; then
        echo "[manifeel] extracting TacSL Isaac Gym package from ${MANIFEEL_ISAACGYM_ARCHIVE}"
        tar -xzf "${MANIFEEL_ISAACGYM_ARCHIVE}" -C "$(dirname -- "${MANIFEEL_ISAACGYM_ROOT}")"
    fi
fi
if [[ ! -d "${MANIFEEL_ISAACGYM_ROOT}/isaacgym/python" ]]; then
    cat >&2 <<EOF
Isaac Gym TacSL package is missing:
  ${MANIFEEL_ISAACGYM_ROOT}/isaacgym/python

Download the licensed TacSL-specific archive from the official ManiFeel README
to ${MANIFEEL_ISAACGYM_ARCHIVE}, then rerun this script. The archive and its
extracted directory are intentionally ignored by Git. Official URL:
https://drive.google.com/file/d/13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v/view
EOF
    exit 2
fi

echo "[manifeel] creating isolated environment: ${MANIFEEL_ENV_PREFIX}"
if [[ ! -x "${MANIFEEL_PYTHON}" ]]; then
    CREATE_ARGS=(create --prefix "${MANIFEEL_ENV_PREFIX}" -c "${CONDA_CHANNEL}" python=3.8 pip=23.3.2 -y)
    if [[ "$(basename -- "${CONDA_EXE}")" == "mamba" ]]; then
        CREATE_ARGS=(create --no-rc --override-channels --lock-timeout "${MANIFEEL_LOCK_TIMEOUT:-60}" -y
            --prefix "${MANIFEEL_ENV_PREFIX}" -c "${CONDA_CHANNEL}" python=3.8 pip=23.3.2)
    fi
    "${CONDA_EXE}" "${CREATE_ARGS[@]}"
fi
conda activate "${MANIFEEL_ENV_PREFIX}"
export CONDA_PREFIX="${MANIFEEL_ENV_PREFIX}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${MANIFEEL_ENV_PREFIX}/torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

python -m pip install --index-url "${PIP_INDEX}" --upgrade "pip<25" setuptools wheel

echo "[manifeel] installing Hopper-capable PyTorch wheel"
if [[ -f "${TORCH_WHEEL}" ]]; then
    python -m pip install --index-url "${PIP_INDEX}" "${TORCH_WHEEL}"
else
    python -m pip install \
        --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}+cu121"
fi
python -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    "torchvision==${TORCHVISION_VERSION}+cu121"

echo "[manifeel] installing public ManiFeel dependencies"
# torchcfm depends on POT but newer POT sdists can fail Cython builds on this
# Python 3.8 stack. Preinstall the latest Py3.8 manylinux wheel available.
python -m pip install --index-url "${PIP_INDEX}" "POT==0.9.5"
python -m pip install --index-url "${PIP_INDEX}" -r "${MANIFEEL_ROOT}/requirements.txt"
# zarr 2.16 imports symbols removed by newer numcodecs releases. SciPy 1.11+
# dropped Python 3.8 while ManiFeel imports scipy.spatial during DP startup.
python -m pip install --index-url "${PIP_INDEX}" "numcodecs==0.12.1" "scipy==1.10.1"

echo "[manifeel] installing Isaac Gym Python package"
python -m pip install -e "${MANIFEEL_ISAACGYM_ROOT}/isaacgym/python"
python -m pip install -e "${MANIFEEL_DP_ROOT}"
python -m pip install -e "${MANIFEEL_IGE_ROOT}"
python -m pip install -e "${MANIFEEL_ROOT}"

echo "[manifeel] import smoke"
python - <<'PY'
# Isaac Gym must be imported before PyTorch in a fresh Python process.
import isaacgym
from isaacgym import gymtorch
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
cuda_available = torch.cuda.is_available()
print("cuda_available", cuda_available)
if cuda_available:
    print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    x = torch.randn((1024, 1024), device="cuda")
    print("cuda_matmul", (x @ x).mean().item())
else:
    print("cuda smoke skipped; use an interactive GPU allocation or sbatch")
import isaacgymenvs
import diffusion_policy
import manifeel
import cv2
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
print("official ManiFeel imports: OK")
PY
