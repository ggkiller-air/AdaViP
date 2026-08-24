#!/usr/bin/env bash
#SBATCH --job-name=mf-camera-check
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
FRAMES="${MANIFEEL_CAMERA_PREFLIGHT_FRAMES:-100}"
TIMEOUT_SECONDS="${MANIFEEL_CAMERA_PREFLIGHT_TIMEOUT:-180}"

cd "${REPO_ROOT}"
# shellcheck source=../../scripts/manifeel/common.sh
source scripts/manifeel/common.sh
activate_manifeel

export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "[preflight] started_at=$(date --iso-8601=seconds)"
echo "[preflight] job_id=${SLURM_JOB_ID} node=$(hostname)"
echo "[preflight] job_gpus=${SLURM_JOB_GPUS:-unset} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[preflight] frames=${FRAMES} timeout_seconds=${TIMEOUT_SECONDS}"
print_manifeel_context
nvidia-smi -L
nvidia-smi \
    --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
scontrol show job "${SLURM_JOB_ID}" -dd | grep -E 'JobId=|NodeList=|TresPerNode=|GRES=|IDX:' || true

set +e
timeout --signal=TERM --kill-after=10s "${TIMEOUT_SECONDS}s" \
    "${MANIFEEL_PYTHON}" scripts/manifeel/diagnose_isaacgym_camera.py \
        --access gpu \
        --image-type color \
        --num-envs 10 \
        --frames "${FRAMES}" \
        --compute-device-id 0 \
        --graphics-device-id 0
STATUS=$?
set -e

echo "[preflight] finished_at=$(date --iso-8601=seconds) status=${STATUS}"
exit "${STATUS}"
