#!/usr/bin/env bash
#SBATCH --job-name=full_sensor_dp_batch64_ep300
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Submit with an immutable source snapshot}"
: "${ADAVIP_RUN_DIR:?Set a persistent unique run directory}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export TMPDIR="/data/wangzihao/t/mf_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
chmod 700 "${TMPDIR}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
echo "node=$(hostname); CPU affinity: $(taskset -pc $$); train_batch=64; val_batch=64; workers=4/2"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
timeout 90s "${MANIFEEL_PYTHON}" scripts/manifeel/check_training_runtime.py
mkdir -p "${ADAVIP_RUN_DIR}"
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
        sleep 60
    done
) > "${ADAVIP_RUN_DIR}/gpu_monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
resume_args=()
if [[ -n "${ADAVIP_RESUME:-}" ]]; then
    resume_args+=("resume=${ADAVIP_RESUME}")
fi
srun --unbuffered bash scripts/manifeel/train_full_sensor_reference.sh \
    "output_dir=${ADAVIP_RUN_DIR}" name=full_sensor_reference_batch64 \
    dataloader.batch_size=64 val_dataloader.batch_size=64 \
    dataloader.num_workers=4 val_dataloader.num_workers=2 \
    training.num_epochs=300 training.max_train_steps=null training.max_val_steps=null \
    "${resume_args[@]}"
