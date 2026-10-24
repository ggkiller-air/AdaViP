#!/usr/bin/env bash
#SBATCH --job-name=manifeel_stage1_v2_teacher_batch8_ep400
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=4-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Submit with an immutable source snapshot}"
: "${ADAVIP_RUN_DIR:?Set a persistent unique run directory}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export MANIFEEL_STAGE1_V2_OUTPUT_ROOT="${ADAVIP_RUN_DIR}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export TMPDIR="/public/home/wangzihao/.cache/avs1v2/${SLURM_JOB_ID}"
export MANIFEEL_DECODED_CACHE_DIR="/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017"
export MANIFEEL_DECODED_CACHE_MODE=ram
mkdir -p "${TMPDIR}"
mkdir -p "${ADAVIP_RUN_DIR}"
echo "node=$(hostname); source=${ADAVIP_SOURCE_ROOT}; CPU affinity: $(taskset -pc $$)"
echo "batch candidates: 8 16 32; formal batch selected after automatic H200 preflight"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
        sleep 120
    done
) >> "${ADAVIP_RUN_DIR}/gpu_monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
srun --unbuffered "${MANIFEEL_PYTHON}" -m adavip.manifeel.stage1_v2_preflight \
    --config "${ADAVIP_SOURCE_ROOT}/configs/manifeel/stage1_v2_teacher.yaml" \
    --output-dir "${ADAVIP_RUN_DIR}"
if [[ -f "${ADAVIP_RUN_DIR}/progress.jsonl" ]] && \
   tail -n 1 "${ADAVIP_RUN_DIR}/progress.jsonl" | \
   "${MANIFEEL_PYTHON}" -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get("event")=="stopped" and d.get("reason")=="SIGUSR1" else 1)'; then
    sbatch --account="${SLURM_JOB_ACCOUNT}" --time=4-00:00:00 \
        --dependency="afterok:${SLURM_JOB_ID}" \
        --export=ALL "${ADAVIP_SOURCE_ROOT}/slurm/manifeel/train_stage1_v2_teacher.sbatch" \
        | tee -a "${ADAVIP_RUN_DIR}/continuation_jobs.log"
fi
