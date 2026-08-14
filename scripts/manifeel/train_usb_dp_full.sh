#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel
print_manifeel_context

RUN_NAME="${MANIFEEL_RUN_NAME:-dp_usb_vision_wrist_full_20260808}"
OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}"
NUM_DEMOS="${MANIFEEL_NUM_DEMOS:-50}"
NUM_EPOCHS="${MANIFEEL_NUM_EPOCHS:-1000}"
SEED="${MANIFEEL_SEED:-42}"

if [[ ! -d "${MANIFEEL_DATASET_PATH}" ]]; then
    echo "Official dataset missing: ${MANIFEEL_DATASET_PATH}" >&2
    exit 2
fi
if ! "${MANIFEEL_PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'; then
    echo "GPU is required for USB DP training." >&2
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
    training.seed="${SEED}" \
    training.num_epochs="${NUM_EPOCHS}" \
    training.resume=true \
    training.rollout_every=10 \
    training.checkpoint_every=10 \
    training.val_every=10000 \
    training.sample_every=10000 \
    task.dataset.max_train_episodes="${NUM_DEMOS}" \
    task.env_runner.n_test=50 \
    task.env_runner.n_test_vis=2 \
    task.env_runner.max_steps=500 \
    logging.mode=offline \
    logging.project=manifeel_dp_usb_baseline \
    hydra.run.dir="${OUTPUT_DIR}"
