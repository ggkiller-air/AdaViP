#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --account=yangyc
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=4-04:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err
set -euo pipefail
: "${LC_SOURCE:?}"
: "${LC_OUTPUT:?}"
: "${LC_VARIANT:?}"
: "${LC_WAVE:?}"
cd "${LC_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/dev/shm/lc_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${LC_OUTPUT}"
exec 9>"${LC_OUTPUT}/training.lock"
if ! flock -n 9; then exit 0; fi
queue_file="${LC_OUTPUT}/queue/${LC_WAVE}.jobs"
for ((attempt=0; attempt<60; attempt++)); do
    if [[ -f "${queue_file}" ]]; then break; fi
    sleep 1
done
# Fail closed if the sibling list is unavailable; never cancel unrelated jobs.
[[ -f "${queue_file}" ]]
while read -r sibling; do
    if [[ "${sibling}" != "${SLURM_JOB_ID}" ]]; then scancel "${sibling}" 2>/dev/null || true; fi
done < "${queue_file}"
if [[ -f "${LC_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
export ADAVIP_STOP_FILE="${LC_OUTPUT}/STOP_REQUEST"
rm -f -- "${ADAVIP_STOP_FILE}" "${LC_OUTPUT}/SAFE_STOP.json"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
taskset -pc $$
df -h /dev/shm
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
        sleep 30
    done
) >> "${LC_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() {
    kill "${monitor_pid}" 2>/dev/null || true
    rm -rf -- "${TMPDIR}"
}
trap cleanup EXIT
arguments=("variant=${LC_VARIANT}" "output_dir=${LC_OUTPUT}")
if [[ -f "${LC_OUTPUT}/checkpoints/latest.ckpt" ]]; then
    arguments+=("resume=${LC_OUTPUT}/checkpoints/latest.ckpt")
fi
trainer_pid=""
forward_stop() {
    # A stop file reaches Python reliably through srun without signalling srun itself.
    printf '%s\n' scheduler_signal > "${ADAVIP_STOP_FILE}"
}
trap forward_stop USR1 TERM
# Core binding and pinned host batches complement the shared read-only RAM cache.
srun --cpu-bind=cores --unbuffered "${MANIFEEL_PYTHON}" -u \
    scripts/manifeel/train_language_conditioning.py "${arguments[@]}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ -f "${LC_OUTPUT}/SAFE_STOP.json" && ! -f "${LC_OUTPUT}/COMPLETE.json" ]]; then
    rm -f -- "${LC_OUTPUT}/SAFE_STOP.json"
    LC_DEPENDENCY="afterok:${SLURM_JOB_ID}" bash "${LC_SOURCE}/scripts/manifeel/submit_language_conditioning.sh"
fi
