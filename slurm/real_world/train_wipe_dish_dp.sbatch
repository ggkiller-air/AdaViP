#!/usr/bin/env bash
#SBATCH --job-name=wipe-dish-dp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/real_world/wipe_dish/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/real_world/wipe_dish/slurm/%x_%j.err

set -euo pipefail

cd /public/home/wangzihao/projects/AdaViP
mkdir -p /data/wangzihao/outputs/real_world/wipe_dish/slurm

# Select pca or rgb at submission time. Short preflight jobs may be submitted
# through Slurm; formal runs use 100 epochs and 10-epoch saves.
: "${WIPE_DISH_DP_MODALITY:?Set WIPE_DISH_DP_MODALITY=pca or rgb}"
export WIPE_DISH_DP_WANDB_MODE="${WIPE_DISH_DP_WANDB_MODE:-disabled}"
export WIPE_DISH_DP_OUTPUT_ROOT="${WIPE_DISH_DP_OUTPUT_ROOT:-/data/wangzihao/outputs/real_world/wipe_dish/dp}"
export WIPE_DISH_DP_NUM_PROCESSES="${WIPE_DISH_DP_NUM_PROCESSES:-4}"

source scripts/real_world/common.sh
activate_rdp
print_dp_sanity_context

exec bash scripts/real_world/train_wipe_dish_dp.sh "$@"
