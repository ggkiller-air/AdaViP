#!/usr/bin/env bash
#SBATCH --job-name=adavip_stage2_t2
#SBATCH --partition=gpux
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/stage2_t2_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/stage2_t2_%j.err
set -euo pipefail

: "${STAGE2_SOURCE:?Set immutable Stage 2 source directory}"
: "${STAGE2_OUTPUT:?Set persistent Stage 2 output directory}"
: "${STAGE2_MODE:?Set preflight or full}"

if [[ "${STAGE2_MODE}" == combined && -f "${STAGE2_OUTPUT}/checkpoints/latest.ckpt" ]]; then
    STAGE2_MODE=full
    STAGE2_RESUME="${STAGE2_OUTPUT}/checkpoints/latest.ckpt"
fi

cd "${STAGE2_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TMPDIR="/dev/shm/adavip_stage2_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${STAGE2_OUTPUT}"
trap 'rm -rf -- "${TMPDIR}"' EXIT

print_manifeel_context
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader

arguments=(
    --teacher /data/wangzihao/outputs/manifeel/checkpoint_archive/full_sensor_20261019_epochs300_400_500/package/checkpoints/epoch500.ckpt
    --labels /public/home/wangzihao/projects/AdaViP/outputs/stage1_utility_labels_ep500_k64
    --cache /data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017
    --cache-config /data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml
    --output "${STAGE2_OUTPUT}"
    --batch-size 32 --workers 6 --epochs 600
)
if [[ "${STAGE2_MODE}" == preflight ]]; then
    arguments+=(--preflight-steps 100)
elif [[ "${STAGE2_MODE}" == combined ]]; then
    arguments+=(--preflight-steps 100 --continue-after-preflight)
elif [[ "${STAGE2_MODE}" == full ]]; then
    if [[ -n "${STAGE2_RESUME:-}" ]]; then
        arguments+=(--resume "${STAGE2_RESUME}")
    fi
else
    echo "Unknown STAGE2_MODE=${STAGE2_MODE}" >&2
    exit 2
fi
trainer_pid=""
forward_stop() {
    if [[ -n "${trainer_pid}" ]] && kill -0 "${trainer_pid}" 2>/dev/null; then
        kill -USR1 "${trainer_pid}"
    fi
}
trap forward_stop USR1 TERM
"${MANIFEEL_PYTHON}" -u scripts/manifeel/train_stage2_t2.py "${arguments[@]}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then
        break
    else
        status=$?
    fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then
        continue
    fi
    exit "${status}"
done
if [[ "${STAGE2_MODE}" == full || "${STAGE2_MODE}" == combined ]] &&
   [[ -f "${STAGE2_OUTPUT}/SAFE_STOP.json" && ! -f "${STAGE2_OUTPUT}/COMPLETE.json" ]]; then
    rm -f -- "${STAGE2_OUTPUT}/SAFE_STOP.json"
    sbatch --dependency="afterok:${SLURM_JOB_ID}" --export="ALL,STAGE2_SOURCE=${STAGE2_SOURCE},STAGE2_OUTPUT=${STAGE2_OUTPUT},STAGE2_MODE=full,STAGE2_RESUME=${STAGE2_OUTPUT}/checkpoints/latest.ckpt" "$0"
fi
