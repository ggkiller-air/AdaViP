#!/usr/bin/env bash
#SBATCH --job-name=mf-plug-vis-paper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="/public/home/wangzihao/projects/AdaViP"
RUN_NAME="dp_power_plug_vision_paper_ep400_seed42"
OUTPUT_ROOT="/data/wangzihao/outputs/manifeel"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to reuse an existing run directory: ${OUTPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}/slurm"
cd "${REPO_ROOT}"

export MANIFEEL_TASK_CONFIG="vision_wrist"
export MANIFEEL_DATASET_PATH="/data/wangzihao/datasets/manifeel/plug_quan_Aug02"
export MANIFEEL_ISAACGYM_CONFIG="isaacgym_config_power_plug.yaml"
export MANIFEEL_RUN_NAME="${RUN_NAME}"
export MANIFEEL_NUM_DEMOS=50
export MANIFEEL_NUM_EPOCHS=400
export MANIFEEL_SEED=42
export MANIFEEL_ROLLOUT_EVERY=0
export MANIFEEL_CHECKPOINT_EVERY=50
export MANIFEEL_VAL_EVERY=10000
export MANIFEEL_SAMPLE_EVERY=10000
export MANIFEEL_WANDB_MODE=offline
export MANIFEEL_WANDB_PROJECT="manifeel_power_plug_paper_baseline"
export MANIFEEL_DEVICE="cuda:0"

exec bash scripts/manifeel/train_single_task_dp.sh \
    _target_=adavip.manifeel.retained_dp_workspace.RetainedDiffusionUnetImageWorkspace \
    checkpoint.topk.k=1000000
