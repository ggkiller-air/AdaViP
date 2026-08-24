#!/usr/bin/env bash
#SBATCH --job-name=mf-dp-usb-eval
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"
CHECKPOINT="${MANIFEEL_CHECKPOINT:-${OUTPUT_ROOT}/dp_usb_vision_wrist_full_20260808/checkpoints/latest_epoch280.ckpt}"
OUTPUT_DIR="${MANIFEEL_EVAL_OUTPUT:-${OUTPUT_ROOT}/dp_usb_vision_wrist_full_20260808/eval_epoch280_n10_steps500_job${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_ROOT}/slurm"
cd "${REPO_ROOT}"

export MANIFEEL_EVAL_N_TEST="${MANIFEEL_EVAL_N_TEST:-10}"
export MANIFEEL_EVAL_N_TEST_VIS="${MANIFEEL_EVAL_N_TEST_VIS:-0}"
export MANIFEEL_EVAL_MAX_STEPS="${MANIFEEL_EVAL_MAX_STEPS:-500}"

echo "[eval] job_id=${SLURM_JOB_ID} node=$(hostname) job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "[eval] checkpoint=${CHECKPOINT} output_dir=${OUTPUT_DIR}"
nvidia-smi -L || true
scontrol show job "${SLURM_JOB_ID}" -dd | grep -E 'JobId=|NodeList=|GRES=|IDX:' || true

exec bash scripts/manifeel/smoke_eval_dp.sh "${CHECKPOINT}" "${OUTPUT_DIR}"
