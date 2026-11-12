#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --account=yangyc
#SBATCH --qos=user_wangzihao
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${RESIDUAL_PUBLISH_SOURCE:?publisher code directory}"
: "${RESIDUAL_RUN_ROOT:?experiment root}"
/public/home/wangzihao/.local/miniforge3/bin/python -u \
    "${RESIDUAL_PUBLISH_SOURCE}/scripts/manifeel/publish_teacher_residual.py" \
    --root "${RESIDUAL_RUN_ROOT}" &
child=$!
trap 'kill -USR1 "${child}" 2>/dev/null || true' USR1 TERM
while true; do
    if wait "${child}"; then exit 0; else result=$?; fi
    if kill -0 "${child}" 2>/dev/null; then continue; fi
    if [[ "${result}" == 75 ]]; then scontrol requeue "${SLURM_JOB_ID}"; exit 0; fi
    exit "${result}"
done
