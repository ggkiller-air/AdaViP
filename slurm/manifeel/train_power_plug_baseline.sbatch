#!/usr/bin/env bash
#SBATCH --job-name=mf-plug-dp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/public/home/wangzihao/projects/AdaViP"
METHOD="${1:?Choose vision, tacrgb, or tacff}"
cd "${REPO_ROOT}"
mkdir -p /data/wangzihao/outputs/manifeel/slurm
exec /public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python \
    scripts/manifeel/train_power_plug_baseline.py "${METHOD}"
