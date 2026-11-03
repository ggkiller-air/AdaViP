#!/usr/bin/env bash
#SBATCH --account=yangyc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=5-00:00:00
#SBATCH --time-min=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
set -euo pipefail
: "${UVIP_RUN_REGISTRY:?registry required}"
: "${UVIP_RECOVERY_SOURCE:?recovery code snapshot required}"
export PYTHONPATH="${UVIP_RECOVERY_SOURCE}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
trap 'scontrol requeue "${SLURM_JOB_ID}"; exit 0' USR1
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
    /public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python -u \
    -m adavip.advanced.recovery --registry "${UVIP_RUN_REGISTRY}" --interval 120 &
watcher_pid=$!
wait "${watcher_pid}"
