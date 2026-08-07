#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel
print_manifeel_context

CHECKPOINT="${1:-${MANIFEEL_OUTPUT_ROOT}/dp_usb_vision_wrist_smoke/checkpoints/latest_epoch0.ckpt}"
OUTPUT_DIR="${2:-${MANIFEEL_OUTPUT_ROOT}/dp_usb_vision_wrist_smoke/eval}"
CFG_NAME="${MANIFEEL_EVAL_CFG_NAME:-}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint missing: ${CHECKPOINT}" >&2
    exit 2
fi
if ! "${MANIFEEL_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
    echo "GPU is required for the official evaluation smoke test." >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Evaluation output already exists; choose a new path: ${OUTPUT_DIR}" >&2
    exit 2
fi

cd "${MANIFEEL_ROOT}"
EVAL_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --output_dir "${OUTPUT_DIR}"
    --device "${MANIFEEL_DEVICE}"
)
if [[ -n "${CFG_NAME}" ]]; then
    EVAL_ARGS+=(--cfg_name "${CFG_NAME}")
fi
exec "${MANIFEEL_PYTHON}" eval.py "${EVAL_ARGS[@]}"
