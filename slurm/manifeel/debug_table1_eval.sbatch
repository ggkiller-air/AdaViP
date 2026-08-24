#!/usr/bin/env bash
#SBATCH --job-name=mf-eval-debug
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
RUN_NAME="${MANIFEEL_RUN_NAME:-table1_vistac_dp_b416_w12_ep300_seed42}"
TASK_ID="${MANIFEEL_EVAL_TASK_ID:-peg_insertion}"
CHECKPOINT="${MANIFEEL_CHECKPOINT:-/data/wangzihao/outputs/manifeel/${RUN_NAME}/checkpoints/best_val.ckpt}"
OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"
OUTPUT_DIR="${MANIFEEL_EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_debug_${TASK_ID}_job${SLURM_JOB_ID}}"
GPU_LOG="${OUTPUT_ROOT}/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv"

mkdir -p "${OUTPUT_ROOT}/slurm"
cd "${REPO_ROOT}"

# shellcheck source=../../scripts/manifeel/common.sh
source scripts/manifeel/common.sh
activate_manifeel

echo "[debug] started_at=$(date --iso-8601=seconds)"
echo "[debug] job_id=${SLURM_JOB_ID} job_name=${SLURM_JOB_NAME}"
echo "[debug] node=$(hostname) job_gpus=${SLURM_JOB_GPUS:-unset} step_gpus=${SLURM_STEP_GPUS:-unset}"
echo "[debug] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[debug] checkpoint=${CHECKPOINT}"
echo "[debug] output_dir=${OUTPUT_DIR}"
print_manifeel_context
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
scontrol show job "${SLURM_JOB_ID}" -dd | grep -E 'JobId=|NodeList=|TresPerNode=|GRES=|IDX:' || true

monitor_gpu() {
    echo "timestamp,index,uuid,memory_used_mib,utilization_gpu_percent" > "${GPU_LOG}"
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu \
            --format=csv,noheader,nounits >> "${GPU_LOG}" 2>&1 || true
        sleep 15
    done
}

monitor_gpu &
MONITOR_PID=$!
cleanup() {
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

set +e
MANIFEEL_EVAL_RETAIN_OUTCOME_VIDEOS="${MANIFEEL_EVAL_RETAIN_OUTCOME_VIDEOS:-1}" \
PYTHONFAULTHANDLER=1 "${MANIFEEL_PYTHON}" scripts/manifeel/eval_multitask_checkpoint.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --task-id "${TASK_ID}" \
    --n-test "${MANIFEEL_EVAL_N_TEST:-10}" \
    --n-test-vis "${MANIFEEL_EVAL_N_TEST_VIS:-0}" \
    --max-steps "${MANIFEEL_EVAL_MAX_STEPS:-120}"
STATUS=$?
set -e

echo "[debug] finished_at=$(date --iso-8601=seconds) status=${STATUS}"
if [[ -f "${OUTPUT_DIR}/eval_log.json" ]]; then
    echo "[debug] aggregate_result=${OUTPUT_DIR}/eval_log.json"
else
    echo "[debug] aggregate_result=missing"
fi
exit "${STATUS}"
