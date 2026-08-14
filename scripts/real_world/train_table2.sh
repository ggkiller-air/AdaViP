#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_rdp
print_rdp_training_context

METHOD="${TABLE2_METHOD:-}"
TASK="${TABLE2_TASK:-}"
DATASET_PATH="${TABLE2_DATASET_PATH:-}"
STAGE="${TABLE2_STAGE:-all}"
SEED="${TABLE2_SEED:-42}"
WANDB_MODE="${TABLE2_WANDB_MODE:-online}"

if [[ -z "${METHOD}" || -z "${TASK}" || -z "${DATASET_PATH}" ]]; then
    echo "Set TABLE2_METHOD, TABLE2_TASK, and TABLE2_DATASET_PATH." >&2
    exit 2
fi

ARGS=(
    --method "${METHOD}"
    --task "${TASK}"
    --dataset "${DATASET_PATH}"
    --stage "${STAGE}"
    --seed "${SEED}"
    --wandb-mode "${WANDB_MODE}"
    --output-root "${TABLE2_OUTPUT_ROOT}"
    --rdp-root "${RDP_ROOT}"
    --python "${RDP_PYTHON}"
)

if [[ "${TABLE2_ALLOW_REFERENCE_CONFIG:-0}" == "1" ]]; then
    ARGS+=(--allow-reference-config)
fi
if [[ -n "${TABLE2_AT_CHECKPOINT:-}" ]]; then
    ARGS+=(--at-checkpoint "${TABLE2_AT_CHECKPOINT}")
fi

EXTRA_OVERRIDES=()
if [[ -n "${TABLE2_NUM_EPOCHS:-}" ]]; then
    EXTRA_OVERRIDES+=(training.num_epochs="${TABLE2_NUM_EPOCHS}")
fi
if [[ -n "${TABLE2_BATCH_SIZE:-}" ]]; then
    EXTRA_OVERRIDES+=(dataloader.batch_size="${TABLE2_BATCH_SIZE}")
fi
if [[ -n "${TABLE2_NUM_WORKERS:-}" ]]; then
    EXTRA_OVERRIDES+=(dataloader.num_workers="${TABLE2_NUM_WORKERS}")
fi
if [[ -n "${TABLE2_MAX_TRAIN_STEPS:-}" ]]; then
    EXTRA_OVERRIDES+=(training.max_train_steps="${TABLE2_MAX_TRAIN_STEPS}")
fi
if [[ -n "${TABLE2_MAX_VAL_STEPS:-}" ]]; then
    EXTRA_OVERRIDES+=(training.max_val_steps="${TABLE2_MAX_VAL_STEPS}")
fi

exec "${RDP_PYTHON}" "${SCRIPT_DIR}/train_table2.py" \
    "${ARGS[@]}" "${EXTRA_OVERRIDES[@]}" "$@"
