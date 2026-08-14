#!/usr/bin/env bash
#SBATCH --job-name=table2_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/wangzihao/outputs/real_world/table2/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/real_world/table2/slurm/%x_%j.err

set -euo pipefail

cd /public/home/wangzihao/projects/AdaViP
mkdir -p /data/wangzihao/outputs/real_world/table2/slurm

: "${TABLE2_METHOD:?Set TABLE2_METHOD to vtdp or rdp}"
: "${TABLE2_TASK:?Set TABLE2_TASK}"
: "${TABLE2_DATASET_PATH:?Set TABLE2_DATASET_PATH to the processed Zarr store}"

bash scripts/real_world/train_table2.sh "$@"
