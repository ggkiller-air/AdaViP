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
: "${S2A_SOURCE:?}"
: "${S2A_OUTPUT:?}"
: "${S2A_STRATEGY:?}"
: "${S2A_WAVE:?}"
cd "${S2A_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/dev/shm/s2a_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${S2A_OUTPUT}"
exec 9>"${S2A_OUTPUT}/training.lock"
if ! flock -n 9; then exit 0; fi
queue_file="${S2A_OUTPUT}/queue/${S2A_WAVE}.jobs"
for ((attempt=0; attempt<60; attempt++)); do
    if [[ -f "${queue_file}" ]]; then break; fi
    sleep 1
done
[[ -f "${queue_file}" ]]
while read -r sibling; do
    if [[ "${sibling}" != "${SLURM_JOB_ID}" ]]; then scancel "${sibling}" 2>/dev/null || true; fi
done < "${queue_file}"
if [[ -f "${S2A_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
export ADAVIP_STOP_FILE="${S2A_OUTPUT}/STOP_REQUEST"
rm -f -- "${ADAVIP_STOP_FILE}" "${S2A_OUTPUT}/SAFE_STOP.json"
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
) >> "${S2A_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() {
    kill "${monitor_pid}" 2>/dev/null || true
    rm -rf -- "${TMPDIR}"
}
trap cleanup EXIT
arguments=(
    --strategy "${S2A_STRATEGY}" --output "${S2A_OUTPUT}"
    --teacher /data/wangzihao/outputs/manifeel/checkpoint_archive/full_sensor_20261019_epochs300_400_500/package/checkpoints/epoch500.ckpt
    --labels /public/home/wangzihao/projects/AdaViP/outputs/stage1_utility_labels_ep500_k64
    --cache /data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017
    --cache-config /data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml
    --workers 8 --cache-workers 8
)
if [[ -f "${S2A_OUTPUT}/checkpoints/latest.ckpt" ]]; then
    arguments+=(--resume "${S2A_OUTPUT}/checkpoints/latest.ckpt")
fi
forward_stop() {
    printf '%s\n' scheduler_signal > "${ADAVIP_STOP_FILE}"
}
trap forward_stop USR1 TERM
srun --cpu-bind=cores --unbuffered "${MANIFEEL_PYTHON}" -u \
    scripts/manifeel/train_stage2_encoder_ablation.py "${arguments[@]}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ -f "${S2A_OUTPUT}/SAFE_STOP.json" && ! -f "${S2A_OUTPUT}/COMPLETE.json" ]]; then
    rm -f -- "${S2A_OUTPUT}/SAFE_STOP.json"
    S2A_DEPENDENCY="afterok:${SLURM_JOB_ID}" bash "${S2A_SOURCE}/scripts/manifeel/submit_stage2_encoder_ablation.sh"
fi
