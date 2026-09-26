#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel

METHOD="${1:?Choose power_plug or ball_sorting}"
shift
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi
case "${METHOD}" in
    power_plug)
        CONFIG_NAME="power_plug_cs_fusion"
        DEFAULT_RUN_NAME="dp_power_plug_cs_fusion_batch8_ep400_seed42"
        DEFAULT_DATASET="${MANIFEEL_DATA_ROOT}/plug_quan_Aug02"
        ;;
    ball_sorting)
        CONFIG_NAME="ball_sorting_cs_fusion"
        DEFAULT_RUN_NAME="dp_ball_sorting_cs_fusion_batch8_ep400_seed42"
        DEFAULT_DATASET="${MANIFEEL_DATA_ROOT}/sorting_quan_Aug8"
        ;;
    *)
        echo "Unknown CS fusion method: ${METHOD}" >&2
        exit 2
        ;;
esac

CONFIG_DIR="${MANIFEEL_REPO_ROOT}/configs/manifeel"
RUN_NAME="${MANIFEEL_RUN_NAME:-${DEFAULT_RUN_NAME}}"
DATASET_PATH="${MANIFEEL_CS_DATASET_PATH:-${DEFAULT_DATASET}}"
OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}"
MANIFEEL_DATASET_PATH="${DATASET_PATH}"
export MANIFEEL_DATASET_PATH
COMMAND=(
    "${MANIFEEL_PYTHON}" train.py
    "--config-dir=${CONFIG_DIR}"
    "--config-name=${CONFIG_NAME}"
    "exp_name=${RUN_NAME}"
    "dataset_path=${DATASET_PATH}"
    "training.device=${MANIFEEL_DEVICE}"
    "hydra.run.dir=${OUTPUT_DIR}"
    "$@"
)

print_manifeel_context
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Dataset missing: ${DATASET_PATH}" >&2
    exit 2
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi
if ! "${MANIFEEL_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
    echo "GPU is required for CS fusion training." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${MANIFEEL_ROOT}"
exec "${COMMAND[@]}"
