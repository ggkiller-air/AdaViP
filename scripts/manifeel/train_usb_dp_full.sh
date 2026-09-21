#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MANIFEEL_TASK_CONFIG="${MANIFEEL_TASK_CONFIG:-vision_wrist}"
export MANIFEEL_ISAACGYM_CONFIG="${MANIFEEL_ISAACGYM_CONFIG:-isaacgym_config_usb.yaml}"
export MANIFEEL_RUN_NAME="${MANIFEEL_RUN_NAME:-dp_usb_vision_wrist_full_20260808}"
export MANIFEEL_WANDB_PROJECT="${MANIFEEL_WANDB_PROJECT:-manifeel_dp_usb_baseline}"
exec bash "${SCRIPT_DIR}/train_single_task_dp.sh" "$@"
