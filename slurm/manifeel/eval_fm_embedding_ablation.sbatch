#!/usr/bin/env bash
# Compare task-conditioning interventions for the trained multi-task FM.
# Each mode uses the same task, seeds, checkpoint, and rollout settings.
#SBATCH --job-name=mf-fm-embed
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
CHECKPOINT="${MANIFEEL_CHECKPOINT:-/data/wangzihao/outputs/manifeel/table1_fm_b416_w12_ep300_seed42/checkpoints/latest_epoch270.ckpt}"
OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"
RUN_ROOT="${MANIFEEL_EVAL_OUTPUT_ROOT:-${OUTPUT_ROOT}/table1_fm_epoch270_power_plug_randn_embedding_trajectory_job${SLURM_JOB_ID}}"
MODE="${MANIFEEL_TASK_EMBEDDING_MODE:?Set MANIFEEL_TASK_EMBEDDING_MODE to correct, other_task, or zero}"
OTHER_TASK_ID="${MANIFEEL_OTHER_TASK_ID:-usb_insertion}"
N_TEST="${MANIFEEL_EVAL_N_TEST:-10}"
MAX_STEPS="${MANIFEEL_EVAL_MAX_STEPS:-500}"
DEVICE="${MANIFEEL_DEVICE:-cuda:0}"
CAMERA_PREFLIGHT_FRAMES="${MANIFEEL_CAMERA_PREFLIGHT_FRAMES:-100}"
CAMERA_PREFLIGHT_TIMEOUT="${MANIFEEL_CAMERA_PREFLIGHT_TIMEOUT:-180}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi
case "${MODE}" in
    correct|other_task|zero) ;;
    *)
        echo "Unsupported embedding mode: ${MODE}" >&2
        exit 2
        ;;
esac
OUTPUT_DIR="${RUN_ROOT}/${MODE}"
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Output path already exists: ${OUTPUT_DIR}" >&2
    exit 2
fi
mkdir -p "${RUN_ROOT}"

cd "${REPO_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export CUDA_DEVICE_ORDER=PCI_BUS_ID
print_manifeel_context
echo "[ablation] job_id=${SLURM_JOB_ID} node=$(hostname)"
echo "[ablation] checkpoint=${CHECKPOINT} task=power_plug_insertion n_test=${N_TEST} max_steps=${MAX_STEPS} fm_prior=normal save_action_trajectories=1"
echo "[ablation] embedding_mode=${MODE} other_task_id=${OTHER_TASK_ID}"
nvidia-smi -L || true
scontrol show job "${SLURM_JOB_ID}" -dd | grep -E 'JobId=|NodeList=|GRES=|IDX:' || true

echo "[ablation] camera_preflight_frames=${CAMERA_PREFLIGHT_FRAMES} timeout_seconds=${CAMERA_PREFLIGHT_TIMEOUT}"
timeout --signal=TERM --kill-after=10s "${CAMERA_PREFLIGHT_TIMEOUT}s" \
    "${MANIFEEL_PYTHON}" scripts/manifeel/diagnose_isaacgym_camera.py \
        --access gpu \
        --image-type color \
        --num-envs 10 \
        --frames "${CAMERA_PREFLIGHT_FRAMES}" \
        --compute-device-id 0 \
        --graphics-device-id 0
echo "[ablation] camera_preflight=passed"

MANIFEEL_TASK_EMBEDDING_MODE="${MODE}" \
MANIFEEL_TASK_EMBEDDING_SOURCE_TASK_ID="${OTHER_TASK_ID}" \
MANIFEEL_EVAL_RETAIN_OUTCOME_VIDEOS=0 \
MANIFEEL_EVAL_SAVE_ACTION_TRAJECTORIES=1 \
PYTHONFAULTHANDLER=1 \
"${MANIFEEL_PYTHON}" scripts/manifeel/eval_multitask_checkpoint.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --task-id power_plug_insertion \
    --n-test "${N_TEST}" \
    --n-test-vis 0 \
    --max-steps "${MAX_STEPS}"

"${MANIFEEL_PYTHON}" scripts/manifeel/compare_action_trajectories.py \
    --run-root "${RUN_ROOT}" \
    --modes "${MODE}" \
    --output "${OUTPUT_DIR}/action_trajectory_summary.json"
echo "[ablation] mode=${MODE} results_root=${RUN_ROOT}"
