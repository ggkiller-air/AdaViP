#!/usr/bin/env bash
#SBATCH --job-name=full_sensor_decoded_batch64_ep300
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Submit with an immutable source snapshot}"
: "${ADAVIP_RUN_DIR:?Set a persistent unique run directory}"
: "${ADAVIP_CACHE_DIR:?Prepare the decoded cache in a separate CPU job first}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export TMPDIR="/dev/shm/avip_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${ADAVIP_RUN_DIR}"
chmod 700 "${TMPDIR}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
echo "node=$(hostname); CPU affinity: $(taskset -pc $$); batch64; workers=8/4; RAM cache; 384GiB host memory"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
df -h /dev/shm
timeout 90s "${MANIFEEL_PYTHON}" scripts/manifeel/check_training_runtime.py
test -f "${ADAVIP_CACHE_DIR}/manifest.json"
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
        sleep 30
    done
) >> "${ADAVIP_RUN_DIR}/gpu_monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; rmdir "${TMPDIR}" 2>/dev/null || true' EXIT
resume_args=()
if [[ -n "${ADAVIP_RESUME:-}" ]]; then
    resume_args+=("resume=${ADAVIP_RESUME}")
fi
srun --unbuffered bash scripts/manifeel/train_full_sensor_reference.sh \
    "output_dir=${ADAVIP_RUN_DIR}" name=full_sensor_reference_batch64 \
    "dataset.decoded_cache_dir=${ADAVIP_CACHE_DIR}" dataset.decoded_cache_mode=ram \
    dataloader.batch_size=64 val_dataloader.batch_size=64 \
    dataloader.num_workers=8 val_dataloader.num_workers=4 \
    training.num_epochs=300 training.max_train_steps=null training.max_val_steps=null \
    runtime.preflight_windows=5 runtime.min_samples_per_second=150 \
    runtime.max_data_wait_fraction=0.35 "${resume_args[@]}"
if "${MANIFEEL_PYTHON}" - "${ADAVIP_RUN_DIR}/progress.jsonl" <<'PY'
import json
import sys
from pathlib import Path

records = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
last = records[-1] if records else {}
sys.exit(0 if last.get("event") == "stopped" and last.get("reason") == "SIGUSR1" else 1)
PY
then
    export ADAVIP_RESUME="${ADAVIP_RUN_DIR}/checkpoints/latest.ckpt"
    sbatch --account="${SLURM_JOB_ACCOUNT}" --dependency="afterok:${SLURM_JOB_ID}" \
        --export=ALL "${ADAVIP_SOURCE_ROOT}/slurm/manifeel/train_full_sensor_decoded_batch64.sbatch" \
        | tee -a "${ADAVIP_RUN_DIR}/continuation_jobs.log"
fi
