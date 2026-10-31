#!/usr/bin/env bash
#SBATCH --account=yangyc
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --nice=10000
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err
set -euo pipefail
: "${S1P_SOURCE:?}"
: "${S1P_OUTPUT:?}"
: "${S1P_PRODUCTION_ROOT:?}"
cd "${S1P_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/public/home/wangzihao/.cache/s1p/i/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/public/home/wangzihao/.cache/s1p/inductor"
export TRITON_CACHE_DIR="/public/home/wangzihao/.cache/s1p/triton"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${S1P_OUTPUT}"
exec 9>"${S1P_OUTPUT}/benchmark.lock"
flock -n 9 || exit 0
if [[ -f "${S1P_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
rm -f -- "${S1P_OUTPUT}/STOP_REQUEST" "${S1P_OUTPUT}/SAFE_STOP.json"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
# An inert tensor in a separate step satisfies the requested >=15% VRAM rule.
srun --overlap --ntasks=1 --cpus-per-task=1 --cpu-bind=cores --unbuffered \
    "${MANIFEEL_PYTHON}" -u scripts/manifeel/reserve_teacher_gpu_memory.py \
    --hold --output "${S1P_OUTPUT}" --fraction 0.16 \
    >> "${S1P_OUTPUT}/memory_reservation_${SLURM_JOB_ID}.log" 2>&1 &
holder_pid=$!
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader
        sleep 60
    done
) >> "${S1P_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() {
    kill "${holder_pid}" "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT
request_stop() {
    printf '%s\n' scheduler_signal > "${S1P_OUTPUT}/STOP_REQUEST"
}
trap request_stop USR1 TERM
srun --overlap --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    bash scripts/manifeel/run_language_step.sh "${MANIFEEL_PYTHON}" -u \
    scripts/manifeel/benchmark_stage1_future_runtime.py \
    --production-root "${S1P_PRODUCTION_ROOT}" --output "${S1P_OUTPUT}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ -f "${S1P_OUTPUT}/SAFE_STOP.json" && ! -f "${S1P_OUTPUT}/COMPLETE.json" ]]; then
    "${MANIFEEL_PYTHON}" scripts/manifeel/submit_partition_job.py \
        --prefix S1P --source "${S1P_SOURCE}" --output "${S1P_OUTPUT}" \
        --script slurm/manifeel/benchmark_stage1_future.sbatch --name s1f_runtime_benchmark \
        --dependency "afterok:${SLURM_JOB_ID}" --set "PRODUCTION_ROOT=${S1P_PRODUCTION_ROOT}"
fi
