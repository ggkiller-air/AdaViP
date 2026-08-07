#!/usr/bin/env bash
#SBATCH --job-name=manifeel-dp-smoke
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
export MANIFEEL_NUM_DEMOS="${MANIFEEL_NUM_DEMOS:-1}"
export MANIFEEL_NUM_EPOCHS="${MANIFEEL_NUM_EPOCHS:-2}"
export MANIFEEL_MAX_TRAIN_STEPS="${MANIFEEL_MAX_TRAIN_STEPS:-2}"
export MANIFEEL_MAX_VAL_STEPS="${MANIFEEL_MAX_VAL_STEPS:-1}"
exec bash scripts/manifeel/smoke_train_dp.sh
