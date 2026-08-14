#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_rdp
print_dp_sanity_context

DATASET_PATH="${DP_SANITY_DATASET_PATH:-/data/wangzihao/datasets/real_world/dp_sanity_no_tactile/processed_30hz}"
OUTPUT_DIR="${DP_SANITY_OUTPUT_DIR:-/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/seed_${DP_SANITY_SEED:-42}}"
TASK_CONFIG="${DP_SANITY_TASK_CONFIG:-dp_sanity_no_tactile_30hz}"
SEED="${DP_SANITY_SEED:-42}"
WANDB_MODE="${DP_SANITY_WANDB_MODE:-online}"

OVERRIDES=()
if [[ -n "${DP_SANITY_NUM_EPOCHS:-}" ]]; then
    OVERRIDES+=(training.num_epochs="${DP_SANITY_NUM_EPOCHS}")
fi
if [[ -n "${DP_SANITY_BATCH_SIZE:-}" ]]; then
    OVERRIDES+=(dataloader.batch_size="${DP_SANITY_BATCH_SIZE}")
fi
if [[ -n "${DP_SANITY_NUM_WORKERS:-}" ]]; then
    OVERRIDES+=(dataloader.num_workers="${DP_SANITY_NUM_WORKERS}")
fi
if [[ -n "${DP_SANITY_MAX_TRAIN_STEPS:-}" ]]; then
    OVERRIDES+=(training.max_train_steps="${DP_SANITY_MAX_TRAIN_STEPS}")
fi

exec "${RDP_PYTHON}" "${SCRIPT_DIR}/train_dp_sanity.py" \
    --dataset "${DATASET_PATH}" \
    --output "${OUTPUT_DIR}" \
    --task-config "${TASK_CONFIG}" \
    --seed "${SEED}" \
    --wandb-mode "${WANDB_MODE}" \
    --rdp-root "${RDP_ROOT}" \
    --python "${RDP_PYTHON}" \
    "${OVERRIDES[@]}" "$@"
