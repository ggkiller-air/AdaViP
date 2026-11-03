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
: "${UVIP_RUN_REGISTRY:?registry required}"
export PYTHONPATH=/public/home/wangzihao/projects/AdaViP-main
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
    /public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python \
    -m adavip.advanced.progress --watch --interval 300
