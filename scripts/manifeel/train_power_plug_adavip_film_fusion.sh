#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
CONFIG_DIR="${MANIFEEL_REPO_ROOT}/configs/manifeel"
CONFIG_NAME="power_plug_adavip_film_fusion"
RUN_NAME="${MANIFEEL_RUN_NAME:-dp_power_plug_adavip_film_fusion_batch8_ep400_seed42}"
OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}"
POWER_PLUG_DATASET="${MANIFEEL_POWER_PLUG_DATASET_PATH:-/data/wangzihao/datasets/manifeel/plug_quan_Aug02}"
MANIFEEL_DATASET_PATH="${POWER_PLUG_DATASET}"
export MANIFEEL_DATASET_PATH
print_manifeel_context
COMMAND=(
    "${MANIFEEL_PYTHON}" train.py
    "--config-dir=${CONFIG_DIR}"
    "--config-name=${CONFIG_NAME}"
    "exp_name=${RUN_NAME}"
    "dataset_path=${POWER_PLUG_DATASET}"
    "training.device=${MANIFEEL_DEVICE}"
    "hydra.run.dir=${OUTPUT_DIR}"
    "$@"
)

if [[ ! -d "${POWER_PLUG_DATASET}" ]]; then
    echo "Power Plug dataset missing: ${POWER_PLUG_DATASET}" >&2
    exit 2
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
if ! "${MANIFEEL_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
    echo "GPU is required for AdaViP film_fusion training." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${MANIFEEL_ROOT}"
exec "${COMMAND[@]}"
