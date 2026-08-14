#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
RUN_NAME="${MANIFEEL_RUN_NAME:-dp_multitask_front_wrist_smoke_20260808}"
NUM_EPOCHS="${MANIFEEL_NUM_EPOCHS:-2}"

MANIFEEL_RUN_NAME="${RUN_NAME}" \
MANIFEEL_NUM_DEMOS="${MANIFEEL_NUM_DEMOS:-1}" \
MANIFEEL_NUM_EPOCHS="${NUM_EPOCHS}" \
MANIFEEL_ROLLOUT_EVERY=0 \
MANIFEEL_CHECKPOINT_EVERY="${MANIFEEL_CHECKPOINT_EVERY:-1}" \
MANIFEEL_N_TEST="${MANIFEEL_N_TEST:-1}" \
MANIFEEL_N_TEST_VIS=0 \
MANIFEEL_MAX_STEPS="${MANIFEEL_MAX_STEPS:-5}" \
MANIFEEL_TASK_EMBEDDING_BACKEND="${MANIFEEL_TASK_EMBEDDING_BACKEND:-transformer}" \
"${SCRIPT_DIR}/train_multitask_dp.sh" \
    training.max_train_steps="${MANIFEEL_MAX_TRAIN_STEPS:-2}" \
    training.max_val_steps="${MANIFEEL_MAX_VAL_STEPS:-1}" \
    training.val_every=1 \
    training.sample_every=1 \
    dataloader.batch_size=1 \
    dataloader.num_workers=0 \
    val_dataloader.batch_size=1 \
    val_dataloader.num_workers=0

LAST_EPOCH="$((NUM_EPOCHS - 1))"
CHECKPOINT="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}/checkpoints/latest_epoch${LAST_EPOCH}.ckpt"
EVAL_OUTPUT="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}/eval_smoke"
MANIFEEL_CHECKPOINT="${CHECKPOINT}" \
MANIFEEL_EVAL_OUTPUT_DIR="${EVAL_OUTPUT}" \
MANIFEEL_EVAL_N_TEST="${MANIFEEL_N_TEST:-1}" \
MANIFEEL_EVAL_N_TEST_VIS=0 \
MANIFEEL_EVAL_MAX_STEPS="${MANIFEEL_MAX_STEPS:-5}" \
bash "${SCRIPT_DIR}/eval_multitask_dp.sh"
