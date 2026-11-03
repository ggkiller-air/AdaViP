#!/usr/bin/env bash
#SBATCH --account=yangyc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/wangzihao/experiments/tabero/slurm/cache_%j.out
#SBATCH --error=/data/wangzihao/experiments/tabero/slurm/cache_%j.err
set -euo pipefail
: "${TABERO_SOURCE:?Set frozen source directory}"
: "${TABERO_CACHE:?Set cache output directory}"
cd "${TABERO_SOURCE}"
export PYTHONPATH="${TABERO_SOURCE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "${TABERO_CACHE}"
exec 9>"${TABERO_CACHE}/build.lock"
flock -n 9 || exit 0
sha256sum --check source.sha256
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    /public/home/wangzihao/.venvs/openvitac/bin/python -m adavip.tabero.cache \
    --source /data/wangzihao/datasets/tabero/tabero_object_25 \
    --output "${TABERO_CACHE}" --workers 8
