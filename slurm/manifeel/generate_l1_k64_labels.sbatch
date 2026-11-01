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
: "${L1K_SOURCE:?}"
: "${L1K_OUTPUT:?}"
: "${L1K_CONFIG:?}"
cd "${L1K_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="/public/home/wangzihao/.cache/l1k/i/${SLURM_JOB_ID}"
export STAGE1_FUTURE_AFFINITY_DIR="${L1K_OUTPUT}/affinity_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${L1K_OUTPUT}"
exec 9>"${L1K_OUTPUT}/label_writer.lock"
flock -n 9 || exit 0
if [[ -f "${L1K_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
export ADAVIP_STOP_FILE="${L1K_OUTPUT}/STOP_REQUEST"
rm -f -- "${ADAVIP_STOP_FILE}" "${L1K_OUTPUT}/SAFE_STOP.json"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
taskset -pc $$
srun --overlap --ntasks=1 --cpus-per-task=1 --cpu-bind=cores --unbuffered \
    "${MANIFEEL_PYTHON}" -u scripts/manifeel/reserve_teacher_gpu_memory.py \
    --hold --output "${L1K_OUTPUT}" --fraction 0.16 \
    >> "${L1K_OUTPUT}/memory_reservation_${SLURM_JOB_ID}.log" 2>&1 &
holder_pid=$!
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader
        sleep 60
    done
) >> "${L1K_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() { kill "${holder_pid}" "${monitor_pid}" 2>/dev/null || true; }
trap cleanup EXIT
request_stop() { printf '%s\n' scheduler_signal > "${ADAVIP_STOP_FILE}"; }
trap request_stop USR1 TERM
srun --overlap --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    bash scripts/manifeel/run_language_step.sh "${MANIFEEL_PYTHON}" -u \
    scripts/manifeel/generate_l1_k64_labels.py --config "${L1K_CONFIG}" &
generator_pid=$!
while true; do
    if wait "${generator_pid}"; then break; else result=$?; fi
    if kill -0 "${generator_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ -f "${L1K_OUTPUT}/SAFE_STOP.json" && ! -f "${L1K_OUTPUT}/COMPLETE.json" ]]; then
    "${MANIFEEL_PYTHON}" scripts/manifeel/submit_partition_job.py \
        --prefix L1K --source "${L1K_SOURCE}" --output "${L1K_OUTPUT}" \
        --script slurm/manifeel/generate_l1_k64_labels.sbatch --name l1_E500_K64_labels \
        --dependency "afterok:${SLURM_JOB_ID}" --set "CONFIG=${L1K_CONFIG}"
fi
