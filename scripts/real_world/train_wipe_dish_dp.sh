#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_rdp

MODALITY="${WIPE_DISH_DP_MODALITY:-pca}"
SEED="${WIPE_DISH_DP_SEED:-42}"
WANDB_MODE="${WIPE_DISH_DP_WANDB_MODE:-offline}"
DATA_ROOT="${WIPE_DISH_DP_DATA_ROOT:-/data/wangzihao}"
OUTPUT_ROOT="${WIPE_DISH_DP_OUTPUT_ROOT:-/data/wangzihao/outputs/real_world/wipe_dish/dp}"
NUM_PROCESSES="${WIPE_DISH_DP_NUM_PROCESSES:-1}"

case "${MODALITY}" in
    pca)
        DATASET_PATH="${WIPE_DISH_DP_DATASET:-${DATA_ROOT}/datasets/real_world/wipe_dish_pca}"
        TASK_CONFIG="wipe_dish_dp_vt_pca_30hz"
        ;;
    rgb)
        DATASET_PATH="${WIPE_DISH_DP_DATASET:-${DATA_ROOT}/wipe-dish-rdp-both}"
        TASK_CONFIG="wipe_dish_dp_vt_rgb_30hz"
        ;;
    *)
        echo "WIPE_DISH_DP_MODALITY must be pca or rgb, got ${MODALITY}" >&2
        exit 2
        ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${MODALITY}/seed_${SEED}"
OVERRIDES=(
    horizon=30
    training.num_epochs="${WIPE_DISH_DP_NUM_EPOCHS:-100}"
    training.checkpoint_every="${WIPE_DISH_DP_CHECKPOINT_EVERY:-10}"
    # Keep every periodic checkpoint unless the caller explicitly opts into
    # top-k pruning.  This leaves room for all checkpoints produced by the
    # upstream zero-based loop.
    checkpoint.topk.k="${WIPE_DISH_DP_TOPK_K:-1000}"
)
if [[ -n "${WIPE_DISH_DP_BATCH_SIZE:-}" ]]; then
    OVERRIDES+=(dataloader.batch_size="${WIPE_DISH_DP_BATCH_SIZE}")
fi
if [[ -n "${WIPE_DISH_DP_NUM_WORKERS:-}" ]]; then
    OVERRIDES+=(dataloader.num_workers="${WIPE_DISH_DP_NUM_WORKERS}")
fi
if [[ -n "${WIPE_DISH_DP_MAX_TRAIN_STEPS:-}" ]]; then
    OVERRIDES+=(training.max_train_steps="${WIPE_DISH_DP_MAX_TRAIN_STEPS}")
fi
if [[ -n "${WIPE_DISH_DP_MAX_VAL_STEPS:-}" ]]; then
    OVERRIDES+=(training.max_val_steps="${WIPE_DISH_DP_MAX_VAL_STEPS}")
fi
if [[ -n "${WIPE_DISH_DP_SAMPLE_EVERY:-}" ]]; then
    OVERRIDES+=(training.sample_every="${WIPE_DISH_DP_SAMPLE_EVERY}")
fi

exec "${RDP_PYTHON}" "${SCRIPT_DIR}/train_dp_sanity.py" \
    --dataset "${DATASET_PATH}" \
    --output "${OUTPUT_DIR}" \
    --task-config "${TASK_CONFIG}" \
    --seed "${SEED}" \
    --wandb-mode "${WANDB_MODE}" \
    --rdp-root "${RDP_ROOT}" \
    --python "${RDP_PYTHON}" \
    --num-processes "${NUM_PROCESSES}" \
    "${OVERRIDES[@]}" "$@"
