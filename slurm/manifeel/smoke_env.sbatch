#!/usr/bin/env bash
#SBATCH --job-name=manifeel-env
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
mkdir -p /data/wangzihao/outputs/manifeel/slurm
cd "${REPO_ROOT}"
export MANIFEEL_UNSET_PROXY="${MANIFEEL_UNSET_PROXY:-0}"
exec bash scripts/manifeel/smoke_env.sh
