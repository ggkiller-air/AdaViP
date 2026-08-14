#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel
print_manifeel_context

RUN_NAME="${MANIFEEL_RUN_NAME:-table1_dp_b416_w12_ep300_seed42}"
if (( ${#RUN_NAME} > 64 )); then
    echo "MANIFEEL_RUN_NAME must be at most 64 characters for W&B run IDs: ${RUN_NAME}" >&2
    exit 2
fi
CONFIG_NAME="${MANIFEEL_CONFIG_NAME:-train_multitask_diffusion_workspace.yaml}"
OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}"
NUM_DEMOS="${MANIFEEL_NUM_DEMOS:-50}"
NUM_EPOCHS="${MANIFEEL_NUM_EPOCHS:-300}"
SEED="${MANIFEEL_SEED:-42}"
ROLLOUT_EVERY="${MANIFEEL_ROLLOUT_EVERY:-0}"
CHECKPOINT_EVERY="${MANIFEEL_CHECKPOINT_EVERY:-30}"
VAL_EVERY="${MANIFEEL_VAL_EVERY:-}"
SAMPLE_EVERY="${MANIFEEL_SAMPLE_EVERY:-}"
N_TEST="${MANIFEEL_N_TEST:-50}"
N_TEST_VIS="${MANIFEEL_N_TEST_VIS:-2}"
MAX_STEPS="${MANIFEEL_MAX_STEPS:-500}"
TEXT_ENCODER="${MANIFEEL_TEXT_ENCODER:-openai/clip-vit-base-patch32}"
TASK_EMBEDDING_PATH="${MANIFEEL_TASK_EMBEDDING_PATH:-${MANIFEEL_OUTPUT_ROOT}/task_embeddings/clip_vit_b32_9task.npz}"
TASK_EMBEDDING_BACKEND="${MANIFEEL_TASK_EMBEDDING_BACKEND:-transformer}"
TASK_EMBEDDING_DIM="${MANIFEEL_TASK_EMBEDDING_DIM:-512}"
TEXT_ENCODER_LOCAL_FILES_ONLY="${MANIFEEL_TEXT_ENCODER_LOCAL_FILES_ONLY:-1}"
WANDB_MODE="${MANIFEEL_WANDB_MODE:-online}"
WANDB_PROJECT="${MANIFEEL_WANDB_PROJECT:-manifeel_multitask}"
export WANDB_MODE
if [[ -n "${MANIFEEL_WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY="${MANIFEEL_WANDB_API_KEY}"
fi

if ! "${MANIFEEL_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
    echo "GPU is required for multi-task DP training." >&2
    exit 2
fi

if [[ ! -f "${TASK_EMBEDDING_PATH}" ]]; then
    mkdir -p "$(dirname -- "${TASK_EMBEDDING_PATH}")"
    embedding_args=(
        --output "${TASK_EMBEDDING_PATH}"
        --backend "${TASK_EMBEDDING_BACKEND}"
        --model "${TEXT_ENCODER}"
        --device cpu
        --dim "${TASK_EMBEDDING_DIM}"
    )
    if [[ "${TEXT_ENCODER_LOCAL_FILES_ONLY}" == "1" ]]; then
        embedding_args+=(--local-files-only)
    fi
    "${MANIFEEL_PYTHON}" "${MANIFEEL_REPO_ROOT}/scripts/manifeel/generate_task_embeddings.py" \
        "${embedding_args[@]}"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${MANIFEEL_ROOT}"

EXTRA_OVERRIDES=()
if [[ -n "${VAL_EVERY}" ]]; then
    EXTRA_OVERRIDES+=(training.val_every="${VAL_EVERY}")
fi
if [[ -n "${SAMPLE_EVERY}" ]]; then
    EXTRA_OVERRIDES+=(training.sample_every="${SAMPLE_EVERY}")
fi
if [[ -n "${MANIFEEL_BATCH_SIZE:-}" ]]; then
    EXTRA_OVERRIDES+=(dataloader.batch_size="${MANIFEEL_BATCH_SIZE}")
fi
if [[ -n "${MANIFEEL_NUM_WORKERS:-}" ]]; then
    EXTRA_OVERRIDES+=(dataloader.num_workers="${MANIFEEL_NUM_WORKERS}")
fi
if [[ -n "${MANIFEEL_PREFETCH_FACTOR:-}" ]]; then
    EXTRA_OVERRIDES+=(dataloader.prefetch_factor="${MANIFEEL_PREFETCH_FACTOR}")
fi
if [[ -n "${MANIFEEL_VAL_BATCH_SIZE:-}" ]]; then
    EXTRA_OVERRIDES+=(val_dataloader.batch_size="${MANIFEEL_VAL_BATCH_SIZE}")
fi
if [[ -n "${MANIFEEL_VAL_NUM_WORKERS:-}" ]]; then
    EXTRA_OVERRIDES+=(val_dataloader.num_workers="${MANIFEEL_VAL_NUM_WORKERS}")
fi
if [[ -n "${MANIFEEL_VAL_PREFETCH_FACTOR:-}" ]]; then
    EXTRA_OVERRIDES+=(val_dataloader.prefetch_factor="${MANIFEEL_VAL_PREFETCH_FACTOR}")
fi
if [[ -n "${MANIFEEL_MAX_TRAIN_STEPS:-}" ]]; then
    EXTRA_OVERRIDES+=(training.max_train_steps="${MANIFEEL_MAX_TRAIN_STEPS}")
fi
if [[ -n "${MANIFEEL_MAX_VAL_STEPS:-}" ]]; then
    EXTRA_OVERRIDES+=(training.max_val_steps="${MANIFEEL_MAX_VAL_STEPS}")
fi

exec "${MANIFEEL_PYTHON}" train.py \
    --config-dir="${MANIFEEL_REPO_ROOT}/configs/manifeel" \
    --config-name="${CONFIG_NAME}" \
    exp_name="${RUN_NAME}" \
    dataset_root="${MANIFEEL_DATA_ROOT}" \
    task_embedding_path="${TASK_EMBEDDING_PATH}" \
    task_embedding_dim="${TASK_EMBEDDING_DIM}" \
    training.device="${MANIFEEL_DEVICE}" \
    training.seed="${SEED}" \
    training.num_epochs="${NUM_EPOCHS}" \
    training.resume=true \
    training.rollout_every="${ROLLOUT_EVERY}" \
    training.checkpoint_every="${CHECKPOINT_EVERY}" \
    task.dataset.max_train_episodes="${NUM_DEMOS}" \
    task.env_runner.n_test="${N_TEST}" \
    task.env_runner.n_test_vis="${N_TEST_VIS}" \
    task.env_runner.max_steps="${MAX_STEPS}" \
    logging.mode="${WANDB_MODE}" \
    logging.project="${WANDB_PROJECT}" \
    logging.name="${RUN_NAME}" \
    logging.id="${RUN_NAME}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    "${EXTRA_OVERRIDES[@]}" \
    "${@}"
