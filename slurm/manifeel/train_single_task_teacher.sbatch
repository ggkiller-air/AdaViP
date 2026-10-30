#!/usr/bin/env bash
#SBATCH --account=yangyc
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err
set -euo pipefail
: "${ST_SOURCE:?}"
: "${ST_OUTPUT:?}"
: "${ST_TASK:?}"
: "${ST_SENSORS:?}"
: "${ST_DROPOUT:?}"
cd "${ST_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/public/home/wangzihao/.cache/st/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${ST_OUTPUT}"
exec 9>"${ST_OUTPUT}/training.lock"
if ! flock -n 9; then exit 0; fi
if [[ -f "${ST_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
export ADAVIP_STOP_FILE="${ST_OUTPUT}/STOP_REQUEST"
export TEACHER_AFFINITY_LOG="${ST_OUTPUT}/worker_affinity_${SLURM_JOB_ID}.jsonl"
rm -f -- "${ADAVIP_STOP_FILE}" "${ST_OUTPUT}/SAFE_STOP.json"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
        sleep 30
    done
) >> "${ST_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
forward_stop() { printf '%s\n' scheduler_signal > "${ADAVIP_STOP_FILE}"; }
trap forward_stop USR1 TERM
arguments=(--task "${ST_TASK}" --sensors "${ST_SENSORS}" --output "${ST_OUTPUT}")
if [[ "${ST_DROPOUT}" == 1 ]]; then arguments+=(--dropout); fi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    bash scripts/manifeel/run_language_step.sh "${MANIFEEL_PYTHON}" -u \
    scripts/manifeel/train_single_task_teacher.py "${arguments[@]}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ -f "${ST_OUTPUT}/SAFE_STOP.json" && ! -f "${ST_OUTPUT}/COMPLETE.json" ]]; then
    python3 scripts/manifeel/submit_partition_job.py --prefix ST --source "${ST_SOURCE}" \
        --output "${ST_OUTPUT}" --script slurm/manifeel/train_single_task_teacher.sbatch \
        --name "${SLURM_JOB_NAME}" --dependency "afterok:${SLURM_JOB_ID}" \
        --set "TASK=${ST_TASK}" --set "SENSORS=${ST_SENSORS}" --set "DROPOUT=${ST_DROPOUT}"
fi
