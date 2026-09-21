#!/usr/bin/env bash
#SBATCH --job-name=mf-t1-eval
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
RUN_NAME="${MANIFEEL_RUN_NAME:?Set MANIFEEL_RUN_NAME to the Table 1 run name}"
CHECKPOINT="${MANIFEEL_CHECKPOINT:-/data/wangzihao/outputs/manifeel/${RUN_NAME}/checkpoints/best_val.ckpt}"
CHECKPOINT_TAG="${MANIFEEL_CHECKPOINT_TAG:-$(basename "${CHECKPOINT}" .ckpt)}"
OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"

mkdir -p "${OUTPUT_ROOT}/slurm"
cd "${REPO_ROOT}"

export MANIFEEL_CHECKPOINT="${CHECKPOINT}"
export MANIFEEL_EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}_${CHECKPOINT_TAG}_eval_job${SLURM_JOB_ID}"
export MANIFEEL_EVAL_N_TEST="${MANIFEEL_EVAL_N_TEST:-10}"
export MANIFEEL_EVAL_N_TEST_VIS="${MANIFEEL_EVAL_N_TEST_VIS:-0}"
export MANIFEEL_EVAL_MAX_STEPS="${MANIFEEL_EVAL_MAX_STEPS:-120}"
export MANIFEEL_DEVICE="${MANIFEEL_DEVICE:-cuda:0}"

echo "[eval] job_id=${SLURM_JOB_ID} node=$(hostname) job_gpus=${SLURM_JOB_GPUS:-unset}"
echo "[eval] checkpoint=${CHECKPOINT} checkpoint_tag=${CHECKPOINT_TAG}"
nvidia-smi -L || true
scontrol show job "${SLURM_JOB_ID}" -dd | grep -E 'JobId=|NodeList=|GRES=|IDX:' || true

exec bash scripts/manifeel/eval_multitask_dp.sh
