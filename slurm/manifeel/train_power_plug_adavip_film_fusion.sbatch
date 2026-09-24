#!/usr/bin/env bash
#SBATCH --job-name=mf-plug-adavip-b8-e400
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/public/home/wangzihao/projects/AdaViP"
cd "${REPO_ROOT}"
mkdir -p /data/wangzihao/outputs/manifeel/slurm
exec bash scripts/manifeel/train_power_plug_adavip_film_fusion.sh "$@"
