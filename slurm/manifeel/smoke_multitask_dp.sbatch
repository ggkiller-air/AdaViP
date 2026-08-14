#!/usr/bin/env bash
#SBATCH --job-name=manifeel_multitask_smoke
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

cd /public/home/wangzihao/projects/AdaViP
mkdir -p /data/wangzihao/outputs/manifeel/slurm

export MANIFEEL_RUN_NAME="${MANIFEEL_RUN_NAME:-dp_multitask_front_wrist_smoke_${SLURM_JOB_ID}}"
export MANIFEEL_TASK_EMBEDDING_PATH="${MANIFEEL_TASK_EMBEDDING_PATH:-/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz}"
export MANIFEEL_TEXT_ENCODER_LOCAL_FILES_ONLY="${MANIFEEL_TEXT_ENCODER_LOCAL_FILES_ONLY:-1}"

bash scripts/manifeel/smoke_multitask_dp.sh
