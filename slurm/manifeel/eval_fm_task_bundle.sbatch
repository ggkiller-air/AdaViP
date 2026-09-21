#!/usr/bin/env bash
#SBATCH --job-name=mf-fm-task-bundle
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"
CHECKPOINT="${MANIFEEL_CHECKPOINT:?Set MANIFEEL_CHECKPOINT}"
EVAL_LABEL="${MANIFEEL_EVAL_LABEL:?Set MANIFEEL_EVAL_LABEL}"
TASK_SEEDS="${MANIFEEL_EVAL_TASK_SEEDS:?Set MANIFEEL_EVAL_TASK_SEEDS as task_id:seed pairs}"
N_TEST="${MANIFEEL_EVAL_N_TEST:-10}"
MAX_STEPS="${MANIFEEL_EVAL_MAX_STEPS:-500}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}/slurm"
IFS=',' read -r -a TASK_SEED_PAIRS <<< "${TASK_SEEDS}"
for pair in "${TASK_SEED_PAIRS[@]}"; do
    IFS=':' read -r task_id test_start_seed <<< "${pair}"
    if [[ -z "${task_id}" || -z "${test_start_seed}" ]]; then
        echo "Invalid task/seed pair: ${pair}" >&2
        exit 2
    fi
    output_dir="${OUTPUT_ROOT}/${EVAL_LABEL}_${task_id}_seed${test_start_seed}_n${N_TEST}_steps${MAX_STEPS}_job${SLURM_JOB_ID}"
    gpu_log="${OUTPUT_ROOT}/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${task_id}_gpu.csv"
    echo "[bundle] task=${task_id} test_start_seed=${test_start_seed} output_dir=${output_dir}"
    MANIFEEL_EVAL_TASK_ID="${task_id}" \
    MANIFEEL_EVAL_OUTPUT_DIR="${output_dir}" \
    MANIFEEL_EVAL_TEST_START_SEED="${test_start_seed}" \
    MANIFEEL_EVAL_N_TEST="${N_TEST}" \
    MANIFEEL_EVAL_MAX_STEPS="${MAX_STEPS}" \
    MANIFEEL_GPU_LOG="${gpu_log}" \
    bash "${REPO_ROOT}/slurm/manifeel/eval_fm_task.sbatch"
done
