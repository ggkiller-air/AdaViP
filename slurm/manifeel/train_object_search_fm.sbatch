#!/usr/bin/env bash
#SBATCH --job-name=mf-os-fm-e100
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

cd /public/home/wangzihao/projects/AdaViP
mkdir -p /data/wangzihao/outputs/manifeel/slurm

export MANIFEEL_RUN_NAME="${MANIFEEL_RUN_NAME:-object_search_fm_ep100_seed42}"
export MANIFEEL_CONFIG_NAME="${MANIFEEL_CONFIG_NAME:-train_object_search_flow_matching_workspace.yaml}"
export MANIFEEL_NUM_DEMOS="${MANIFEEL_NUM_DEMOS:-50}"
export MANIFEEL_NUM_EPOCHS="${MANIFEEL_NUM_EPOCHS:-100}"
export MANIFEEL_ROLLOUT_EVERY="${MANIFEEL_ROLLOUT_EVERY:-0}"
export MANIFEEL_CHECKPOINT_EVERY="${MANIFEEL_CHECKPOINT_EVERY:-10}"
export MANIFEEL_VAL_EVERY="${MANIFEEL_VAL_EVERY:-10}"
export MANIFEEL_SAMPLE_EVERY="${MANIFEEL_SAMPLE_EVERY:-10}"
export MANIFEEL_WANDB_MODE="${MANIFEEL_WANDB_MODE:-offline}"
export MANIFEEL_TASK_EMBEDDING_PATH="${MANIFEEL_TASK_EMBEDDING_PATH:-/data/wangzihao/outputs/manifeel/task_embeddings/clip_vit_b32_9task.npz}"
export MANIFEEL_TEXT_ENCODER_LOCAL_FILES_ONLY="${MANIFEEL_TEXT_ENCODER_LOCAL_FILES_ONLY:-1}"

bash scripts/manifeel/train_multitask_dp.sh \
    checkpoint.prune=false \
    checkpoint.save_best_val_ckpt=false \
    "$@"
