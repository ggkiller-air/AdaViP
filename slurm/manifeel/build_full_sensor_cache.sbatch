#!/usr/bin/env bash
#SBATCH --job-name=full_sensor_decode
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Set a frozen source root}"
: "${ADAVIP_CACHE_DIR:?Set the persistent decoded cache directory}"
: "${ADAVIP_CACHE_CONFIG:?Set the exact training dataset configuration}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export TMPDIR="/data/wangzihao/t/decode_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
srun --unbuffered "${MANIFEEL_PYTHON}" scripts/manifeel/build_full_sensor_cache.py \
    --config "${ADAVIP_CACHE_CONFIG}" --cache-dir "${ADAVIP_CACHE_DIR}" \
    --workers 16 --block-frames 48
