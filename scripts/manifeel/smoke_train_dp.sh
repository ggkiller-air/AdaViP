#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel
print_manifeel_context

NUM_DEMOS="${MANIFEEL_NUM_DEMOS:-1}"
NUM_EPOCHS="${MANIFEEL_NUM_EPOCHS:-2}"
MAX_TRAIN_STEPS="${MANIFEEL_MAX_TRAIN_STEPS:-2}"
MAX_VAL_STEPS="${MANIFEEL_MAX_VAL_STEPS:-1}"
RUN_NAME="${MANIFEEL_RUN_NAME:-dp_usb_vision_wrist_smoke}"
OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}"

if [[ ! -d "${MANIFEEL_DATASET_PATH}" ]]; then
    echo "Official dataset missing: ${MANIFEEL_DATASET_PATH}" >&2
    exit 2
fi
if ! "${MANIFEEL_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
    echo "GPU is required for the official training smoke test." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
cd "${MANIFEEL_ROOT}"
exec "${MANIFEEL_PYTHON}" train.py \
    --config-name=train_diffusion_workspace.yaml \
    task=vision_wrist \
    exp_name="${RUN_NAME}" \
    dataset_path="${MANIFEEL_DATASET_PATH}" \
    isaacgym_cfg_name=isaacgym_config_usb.yaml \
    training.device="${MANIFEEL_DEVICE}" \
    training.seed="${MANIFEEL_SEED:-42}" \
    training.num_epochs="${NUM_EPOCHS}" \
    training.max_train_steps="${MAX_TRAIN_STEPS}" \
    training.max_val_steps="${MAX_VAL_STEPS}" \
    training.rollout_every=1 \
    training.val_every=1 \
    training.sample_every=1 \
    training.checkpoint_every=1 \
    dataloader.batch_size=1 \
    dataloader.num_workers=0 \
    val_dataloader.batch_size=1 \
    val_dataloader.num_workers=0 \
    task.dataset.max_train_episodes="${NUM_DEMOS}" \
    task.env_runner.n_test=1 \
    task.env_runner.n_test_vis=1 \
    task.env_runner.max_steps=5 \
    logging.mode=offline \
    logging.project=manifeel_dp_smoke \
    hydra.run.dir="${OUTPUT_DIR}"
