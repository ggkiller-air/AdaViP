#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel
print_manifeel_context

TASK_CONFIG="${MANIFEEL_TASK_CONFIG:-vision_wrist}"
ISAACGYM_CONFIG="${MANIFEEL_ISAACGYM_CONFIG:-isaacgym_config_usb.yaml}"
RUN_NAME="${MANIFEEL_RUN_NAME:-dp_usb_${TASK_CONFIG}_single_task}"
OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}"
NUM_DEMOS="${MANIFEEL_NUM_DEMOS:-50}"
NUM_EPOCHS="${MANIFEEL_NUM_EPOCHS:-1000}"
SEED="${MANIFEEL_SEED:-42}"

if [[ ! -d "${MANIFEEL_DATASET_PATH}" ]]; then
    echo "Official dataset missing: ${MANIFEEL_DATASET_PATH}" >&2
    exit 2
fi
if ! "${MANIFEEL_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
    echo "GPU is required for single-task DP training." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${MANIFEEL_ROOT}"
exec "${MANIFEEL_PYTHON}" train.py \
    --config-name=train_diffusion_workspace.yaml \
    task="${TASK_CONFIG}" \
    exp_name="${RUN_NAME}" \
    dataset_path="${MANIFEEL_DATASET_PATH}" \
    isaacgym_cfg_name="${ISAACGYM_CONFIG}" \
    training.device="${MANIFEEL_DEVICE}" \
    training.seed="${SEED}" \
    training.num_epochs="${NUM_EPOCHS}" \
    training.resume=true \
    training.rollout_every="${MANIFEEL_ROLLOUT_EVERY:-10}" \
    training.checkpoint_every="${MANIFEEL_CHECKPOINT_EVERY:-10}" \
    training.val_every="${MANIFEEL_VAL_EVERY:-10000}" \
    training.sample_every="${MANIFEEL_SAMPLE_EVERY:-10000}" \
    task.dataset.max_train_episodes="${NUM_DEMOS}" \
    task.env_runner.n_test="${MANIFEEL_N_TEST:-50}" \
    task.env_runner.n_test_vis="${MANIFEEL_N_TEST_VIS:-2}" \
    task.env_runner.max_steps="${MANIFEEL_MAX_STEPS:-500}" \
    logging.mode="${MANIFEEL_WANDB_MODE:-offline}" \
    logging.project="${MANIFEEL_WANDB_PROJECT:-manifeel_single_task_dp}" \
    logging.name="${RUN_NAME}" \
    logging.id="${RUN_NAME}" \
    hydra.run.dir="${OUTPUT_DIR}" \
    "$@"
