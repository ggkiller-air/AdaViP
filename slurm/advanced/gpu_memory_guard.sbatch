#!/usr/bin/env bash
#SBATCH --account=yangyc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec /public/home/wangzihao/.local/miniforge3/bin/python -u \
    /public/home/wangzihao/projects/AdaViP-main/scripts/advanced/supervise_gpu_memory.py \
    --registry /data/wangzihao/experiments/advanced_uvip/20261103_1707/registry.json
