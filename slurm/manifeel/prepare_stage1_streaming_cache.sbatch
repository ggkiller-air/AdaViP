#!/usr/bin/env bash
#SBATCH --job-name=stage1_block_hash_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Set a frozen source root}"
: "${ADAVIP_BLOCK_INDEX:?Set a new independent block-index path}"
: "${ADAVIP_REPO_ROOT:?Set the repository environment root}"
source "${ADAVIP_REPO_ROOT}/scripts/manifeel/common.sh"
activate_manifeel
export PYTHONPATH="${ADAVIP_SOURCE_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "${ADAVIP_SOURCE_ROOT}"
srun --unbuffered "${MANIFEEL_PYTHON}" scripts/manifeel/prepare_stage1_streaming_cache.py \
    --cache-dir /data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017 \
    --output "${ADAVIP_BLOCK_INDEX}" --workers 8
