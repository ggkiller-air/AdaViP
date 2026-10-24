#!/usr/bin/env bash
#SBATCH --job-name=adavip_st2_mlp
#SBATCH --partition=gpux
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/stage2_mlp_t2_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/stage2_mlp_t2_%j.err
set -euo pipefail

: "${STAGE2_MLP_SOURCE:?Set immutable MLP source directory}"
: "${STAGE2_MLP_OUTPUT:?Set isolated MLP output directory}"
: "${STAGE2_MLP_MODE:?Set combined or full}"
: "${STAGE2_MLP_WAVE:?Set submission wave identifier}"

if [[ "${STAGE2_MLP_MODE}" == combined && -f "${STAGE2_MLP_OUTPUT}/checkpoints/latest.ckpt" ]]; then
    STAGE2_MLP_MODE=full
    STAGE2_MLP_RESUME="${STAGE2_MLP_OUTPUT}/checkpoints/latest.ckpt"
fi

cd "${STAGE2_MLP_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TMPDIR="/dev/shm/adavip_stage2_mlp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${STAGE2_MLP_OUTPUT}"
trap 'rm -rf -- "${TMPDIR}"' EXIT

# Only one of the gpu/gpux requests may enter this training run.
exec 9>"${STAGE2_MLP_OUTPUT}/training.lock"
if ! flock -n 9; then
    echo "Another MLP-fusion allocation owns the training lock; exiting."
    exit 0
fi
wave_file="${STAGE2_MLP_OUTPUT}/queue/${STAGE2_MLP_WAVE}.jobs"
for ((attempt=0; attempt<30; attempt++)); do
    if [[ -f "${wave_file}" ]]; then
        break
    fi
    sleep 1
done
if [[ -f "${wave_file}" ]]; then
    while read -r queued_id; do
        if [[ "${queued_id}" != "${SLURM_JOB_ID}" ]]; then
            scancel "${queued_id}" 2>/dev/null || true
        fi
    done < "${wave_file}"
fi

print_manifeel_context
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader

arguments=(
    --teacher /data/wangzihao/outputs/manifeel/checkpoint_archive/full_sensor_20261019_epochs300_400_500/package/checkpoints/epoch500.ckpt
    --labels /public/home/wangzihao/projects/AdaViP/outputs/stage1_utility_labels_ep500_k64
    --cache /data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017
    --cache-config /data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml
    --output "${STAGE2_MLP_OUTPUT}"
    --batch-size 32 --workers 6 --epochs 600
)
if [[ "${STAGE2_MLP_MODE}" == preflight ]]; then
    arguments+=(--preflight-steps 100)
elif [[ "${STAGE2_MLP_MODE}" == combined ]]; then
    arguments+=(--preflight-steps 100 --continue-after-preflight)
elif [[ "${STAGE2_MLP_MODE}" == full ]]; then
    if [[ -n "${STAGE2_MLP_RESUME:-}" ]]; then
        arguments+=(--resume "${STAGE2_MLP_RESUME}")
    fi
else
    echo "Unknown STAGE2_MLP_MODE=${STAGE2_MLP_MODE}" >&2
    exit 2
fi
trainer_pid=""
forward_stop() {
    if [[ -n "${trainer_pid}" ]] && kill -0 "${trainer_pid}" 2>/dev/null; then
        kill -USR1 "${trainer_pid}"
    fi
}
trap forward_stop USR1 TERM
"${MANIFEEL_PYTHON}" -u scripts/manifeel/train_stage2_mlp_t2.py "${arguments[@]}" &
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
if [[ "${STAGE2_MLP_MODE}" == full || "${STAGE2_MLP_MODE}" == combined ]] &&
   [[ -f "${STAGE2_MLP_OUTPUT}/SAFE_STOP.json" && ! -f "${STAGE2_MLP_OUTPUT}/COMPLETE.json" ]]; then
    rm -f -- "${STAGE2_MLP_OUTPUT}/SAFE_STOP.json"
    STAGE2_MLP_MODE=full STAGE2_MLP_DEPENDENCY="afterok:${SLURM_JOB_ID}" \
        bash "${STAGE2_MLP_SOURCE}/scripts/manifeel/submit_stage2_mlp_t2.sh"
fi
