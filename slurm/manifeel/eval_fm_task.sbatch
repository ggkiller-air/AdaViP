#!/usr/bin/env bash
#SBATCH --job-name=mf-fm-task
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
CHECKPOINT="${MANIFEEL_CHECKPOINT:?Set MANIFEEL_CHECKPOINT to an FM checkpoint}"
TASK_ID="${MANIFEEL_EVAL_TASK_ID:?Set MANIFEEL_EVAL_TASK_ID}"
OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"
CHECKPOINT_NAME="${CHECKPOINT##*/}"
if [[ "${CHECKPOINT_NAME}" =~ _epoch([0-9]+)\.ckpt$ ]]; then
    CHECKPOINT_EPOCH="${BASH_REMATCH[1]}"
else
    CHECKPOINT_EPOCH="unknown"
fi
OUTPUT_DIR="${MANIFEEL_EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/table1_fm_epoch${CHECKPOINT_EPOCH}_${TASK_ID}_n10_steps500_job${SLURM_JOB_ID}}"
N_TEST="${MANIFEEL_EVAL_N_TEST:-10}"
MAX_STEPS="${MANIFEEL_EVAL_MAX_STEPS:-500}"
DEVICE="${MANIFEEL_DEVICE:-cuda:0}"
CAMERA_PREFLIGHT_FRAMES="${MANIFEEL_CAMERA_PREFLIGHT_FRAMES:-100}"
CAMERA_PREFLIGHT_TIMEOUT="${MANIFEEL_CAMERA_PREFLIGHT_TIMEOUT:-180}"
GPU_LOG="${OUTPUT_ROOT}/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv"

case "${TASK_ID}" in
    peg_insertion|usb_insertion|power_plug_insertion|gear_assembly|nut_bolt_assembly|bulb_installation|peg_reorientation|object_search|ball_sorting) ;;
    *)
        echo "Unsupported task: ${TASK_ID}" >&2
        exit 2
        ;;
esac
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output path already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}/slurm"
cd "${REPO_ROOT}"
# shellcheck source=../../scripts/manifeel/common.sh
source scripts/manifeel/common.sh
activate_manifeel
export CUDA_DEVICE_ORDER=PCI_BUS_ID

print_manifeel_context
echo "[task-eval] job_id=${SLURM_JOB_ID} node=$(hostname)"
echo "[task-eval] checkpoint=${CHECKPOINT} task=${TASK_ID} output_dir=${OUTPUT_DIR}"
echo "[task-eval] n_test=${N_TEST} max_steps=${MAX_STEPS} embedding_mode=correct"
nvidia-smi -L || true
scontrol show job "${SLURM_JOB_ID}" -dd | grep -E 'JobId=|NodeList=|GRES=|IDX:' || true

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

echo "[task-eval] camera_preflight_frames=${CAMERA_PREFLIGHT_FRAMES} timeout_seconds=${CAMERA_PREFLIGHT_TIMEOUT}"
timeout --signal=TERM --kill-after=10s "${CAMERA_PREFLIGHT_TIMEOUT}s" \
    "${MANIFEEL_PYTHON}" scripts/manifeel/diagnose_isaacgym_camera.py \
        --access gpu \
        --image-type color \
        --num-envs 10 \
        --frames "${CAMERA_PREFLIGHT_FRAMES}" \
        --compute-device-id 0 \
        --graphics-device-id 0
echo "[task-eval] camera_preflight=passed"

MANIFEEL_TASK_EMBEDDING_MODE=correct \
MANIFEEL_EVAL_RETAIN_OUTCOME_VIDEOS=0 \
MANIFEEL_EVAL_SAVE_ACTION_TRAJECTORIES=1 \
PYTHONFAULTHANDLER=1 \
"${MANIFEEL_PYTHON}" scripts/manifeel/eval_multitask_checkpoint.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --task-id "${TASK_ID}" \
    --n-test "${N_TEST}" \
    --n-test-vis 0 \
    --max-steps "${MAX_STEPS}"

echo "[task-eval] result=${OUTPUT_DIR}/eval_log.json"
