#!/usr/bin/env bash
#SBATCH --job-name=stage1_ckpt_every10_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_ARCHIVE_SOURCE:?Set the persistent archive script snapshot}"
: "${ADAVIP_RUN_DIR:?Set the existing training directory}"
: "${ADAVIP_SOURCE_ROOT:?Set the original training source snapshot}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
set +e
srun --unbuffered "${MANIFEEL_PYTHON}" \
    "${ADAVIP_ARCHIVE_SOURCE}/archive_stage1_v2_checkpoints.py" \
    --run-dir "${ADAVIP_RUN_DIR}" --every 10 --final-epoch 400
status=$?
set -e
if [[ "${status}" -eq 75 ]]; then
    sbatch --account="${SLURM_JOB_ACCOUNT}" --time=4-00:00:00 \
        --dependency="afterok:${SLURM_JOB_ID}" --export=ALL \
        "${ADAVIP_ARCHIVE_SOURCE}/archive_stage1_v2_checkpoints.sbatch" \
        | tee -a "${ADAVIP_RUN_DIR}/checkpoint_archive/continuation_jobs.log"
else
    exit "${status}"
fi
