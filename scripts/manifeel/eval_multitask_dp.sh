#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel
print_manifeel_context

CHECKPOINT="${MANIFEEL_CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
    echo "Set MANIFEEL_CHECKPOINT to a multi-task DP checkpoint." >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi

RUN_NAME="${MANIFEEL_RUN_NAME:-dp_multitask_front_wrist_eval_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${MANIFEEL_EVAL_OUTPUT_DIR:-${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}}"
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Eval output already exists, choose MANIFEEL_EVAL_OUTPUT_DIR: ${OUTPUT_DIR}" >&2
    exit 2
fi

export MANIFEEL_EVAL_RETAIN_OUTCOME_VIDEOS="${MANIFEEL_EVAL_RETAIN_OUTCOME_VIDEOS:-1}"

exec "${MANIFEEL_PYTHON}" "${MANIFEEL_REPO_ROOT}/scripts/manifeel/eval_multitask_checkpoint.py" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${MANIFEEL_DEVICE}" \
    --n-test "${MANIFEEL_EVAL_N_TEST:-50}" \
    --n-test-vis "${MANIFEEL_EVAL_N_TEST_VIS:-0}" \
    --max-steps "${MANIFEEL_EVAL_MAX_STEPS:-500}"
