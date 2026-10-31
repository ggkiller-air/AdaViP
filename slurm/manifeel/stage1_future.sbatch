#!/usr/bin/env bash
#SBATCH --account=yangyc
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err
set -euo pipefail
: "${S1F_SOURCE:?}"
: "${S1F_OUTPUT:?}"
: "${S1F_PHASE:?}"
cd "${S1F_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 PYTHONDONTWRITEBYTECODE=1
export LC_RUNTIME_PROFILE=tf32_deterministic
export TMPDIR="/public/home/wangzihao/.cache/s1f/i/${SLURM_JOB_ID}"
export STAGE1_FUTURE_AFFINITY_DIR="${S1F_OUTPUT}/affinity_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${S1F_OUTPUT}"
exec 9>"${S1F_OUTPUT}/training.lock"
if ! flock -n 9; then exit 0; fi
if [[ -f "${S1F_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
export ADAVIP_STOP_FILE="${S1F_OUTPUT}/STOP_REQUEST"
rm -f -- "${ADAVIP_STOP_FILE}" "${S1F_OUTPUT}/SAFE_STOP.json"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
taskset -pc $$
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
        sleep 60
    done
) >> "${S1F_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() {
    kill "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT
forward_stop() {
    printf '%s\n' scheduler_signal > "${ADAVIP_STOP_FILE}"
}
trap forward_stop USR1 TERM
if [[ "${S1F_PHASE}" == sanity ]]; then
    arguments=(scripts/manifeel/sanity_stage1_future.py --output "${S1F_OUTPUT}")
else
    : "${S1F_EXPERIMENT:?}"
    : "${S1F_CONFIG:?}"
    arguments=(scripts/manifeel/train_stage1_future.py --config "${S1F_CONFIG}")
    if [[ -f "${S1F_OUTPUT}/checkpoints/latest.ckpt" ]]; then
        arguments+=("resume=${S1F_OUTPUT}/checkpoints/latest.ckpt")
    fi
fi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    bash scripts/manifeel/run_language_step.sh "${MANIFEEL_PYTHON}" -u "${arguments[@]}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ "${S1F_PHASE}" != sanity && -f "${S1F_OUTPUT}/SAFE_STOP.json" && ! -f "${S1F_OUTPUT}/COMPLETE.json" ]]; then
    "${MANIFEEL_PYTHON}" scripts/manifeel/submit_partition_job.py \
        --prefix S1F --source "${S1F_SOURCE}" --output "${S1F_OUTPUT}" \
        --script slurm/manifeel/stage1_future.sbatch --name "s1f_${S1F_EXPERIMENT}_k4_batch32" \
        --dependency "afterok:${SLURM_JOB_ID}" --set PHASE=train \
        --set "EXPERIMENT=${S1F_EXPERIMENT}" --set "CONFIG=${S1F_CONFIG}"
fi
