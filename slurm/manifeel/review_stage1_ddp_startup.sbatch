#!/usr/bin/env bash
#SBATCH --job-name=stage1_startup_review_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=768G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=server09
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_REPO_ROOT:?Set the environment repository root}"
: "${ADAVIP_SOURCE_ROOT:?Set the exact frozen pending GPU source}"
: "${ADAVIP_ASYNC_ROOT:?Set the independent experiment root}"
: "${ADAVIP_PRODUCTION_ROOT:?Set the read-only production root}"
: "${ADAVIP_REVIEW_SCRIPT:?Set the frozen CPU review harness}"
source "${ADAVIP_REPO_ROOT}/scripts/manifeel/common.sh"
activate_manifeel
export PYTHONPATH="${ADAVIP_SOURCE_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
srun --unbuffered "${MANIFEEL_PYTHON}" "${ADAVIP_REVIEW_SCRIPT}" \
    --source "${ADAVIP_SOURCE_ROOT}" --production "${ADAVIP_PRODUCTION_ROOT}" \
    --experiment "${ADAVIP_ASYNC_ROOT}" --output "${ADAVIP_ASYNC_ROOT}/cpu_review_${SLURM_JOB_ID}"
