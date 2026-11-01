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
: "${L1X_SOURCE:?}"
: "${L1X_OUTPUT:?}"
: "${L1X_MODE:?}"
: "${L1X_CONFIG:?}"
cd "${L1X_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 PYTHONDONTWRITEBYTECODE=1
export LC_RUNTIME_PROFILE=stage0
export TMPDIR="/public/home/wangzihao/.cache/l1d/i/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${L1X_OUTPUT}"
exec 9>"${L1X_OUTPUT}/training.lock"
if ! flock -n 9; then exit 0; fi
if [[ -f "${L1X_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
export ADAVIP_STOP_FILE="${L1X_OUTPUT}/STOP_REQUEST"
rm -f -- "${ADAVIP_STOP_FILE}" "${L1X_OUTPUT}/SAFE_STOP.json"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
taskset -pc $$
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
        sleep 60
    done
) >> "${L1X_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() { kill "${monitor_pid}" 2>/dev/null || true; }
trap cleanup EXIT
forward_stop() { printf '%s\n' scheduler_signal > "${ADAVIP_STOP_FILE}"; }
trap forward_stop USR1 TERM
if [[ "${L1X_MODE}" == train ]]; then
    arguments=(scripts/manifeel/train_l1_dropout.py --config "${L1X_CONFIG}")
    if [[ -f "${L1X_OUTPUT}/checkpoints/latest.ckpt" ]]; then
        arguments+=("resume=${L1X_OUTPUT}/checkpoints/latest.ckpt")
    fi
elif [[ "${L1X_MODE}" == mask ]]; then
    arguments=(scripts/manifeel/eval_input_mask_sensitivity.py --config "${L1X_CONFIG}")
else
    echo "Unknown L1X_MODE: ${L1X_MODE}" >&2
    exit 2
fi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    bash scripts/manifeel/run_language_step.sh "${MANIFEEL_PYTHON}" -u "${arguments[@]}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ -f "${L1X_OUTPUT}/SAFE_STOP.json" && ! -f "${L1X_OUTPUT}/COMPLETE.json" ]]; then
    "${MANIFEEL_PYTHON}" scripts/manifeel/submit_partition_job.py \
        --prefix L1X --source "${L1X_SOURCE}" --output "${L1X_OUTPUT}" \
        --script slurm/manifeel/l1_experiment.sbatch --name "l1_${L1X_MODE}_batch64_ep500" \
        --dependency "afterok:${SLURM_JOB_ID}" --set "MODE=${L1X_MODE}" --set "CONFIG=${L1X_CONFIG}"
fi
