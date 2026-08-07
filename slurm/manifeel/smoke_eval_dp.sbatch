#!/usr/bin/env bash
#SBATCH --job-name=manifeel-dp-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
mkdir -p /data/wangzihao/outputs/manifeel/slurm
cd "${REPO_ROOT}"
CHECKPOINT="${MANIFEEL_CHECKPOINT:-/data/wangzihao/outputs/manifeel/dp_usb_vision_wrist_smoke/checkpoints/latest_epoch0.ckpt}"
OUTPUT_DIR="${MANIFEEL_EVAL_OUTPUT:-/data/wangzihao/outputs/manifeel/dp_usb_vision_wrist_smoke/eval_${SLURM_JOB_ID}}"
exec bash scripts/manifeel/smoke_eval_dp.sh "${CHECKPOINT}" "${OUTPUT_DIR}"
