#!/usr/bin/env bash
#SBATCH --job-name=piper-rdp-at
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${ADAVIP_RDP_DATASET:?Export ADAVIP_RDP_DATASET before sbatch}"
: "${ADAVIP_RDP_OUTPUT:?Export ADAVIP_RDP_OUTPUT before sbatch}"
REPO_ROOT="${ADAVIP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATASET="${ADAVIP_RDP_DATASET}"
OUTPUT="${ADAVIP_RDP_OUTPUT}"
export ADAVIP_RDP_DATASET ADAVIP_RDP_OUTPUT

cd "${REPO_ROOT}"
# shellcheck source=../../scripts/real_world/common.sh
source scripts/real_world/common.sh
activate_rdp

export WANDB_MODE=disabled
export OMP_NUM_THREADS=2

echo "[piper-rdp-at] job_id=${SLURM_JOB_ID} host=$(hostname)"
echo "[piper-rdp-at] dataset=${DATASET}"
echo "[piper-rdp-at] output=${OUTPUT}/at"
nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
"${RDP_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

dataset = Path(os.environ["ADAVIP_RDP_DATASET"])
audit = json.loads((dataset / "audit/gelsight_audit.json").read_text())
assert audit["detection"]["success_rate"] == 1.0
assert audit["pca"]["components"] == 15
assert (dataset / "replay_buffer.zarr").is_dir()
print(
    "[piper-rdp-at] data_audit="
    f"{audit['frames']} frames, "
    f"PCA variance={audit['pca']['total_explained_variance_ratio']:.6f}"
)
PY

"${RDP_PYTHON}" scripts/real_world/train_piper_rdp.py \
    --stage at \
    --dataset "${DATASET}" \
    --output "${OUTPUT}" \
    --python "${RDP_PYTHON}" \
    --wandb-mode disabled \
    dataloader.batch_size=256 \
    dataloader.num_workers=8 \
    val_dataloader.batch_size=256 \
    val_dataloader.num_workers=0 \
    val_dataloader.persistent_workers=false \
    training.num_epochs=601 \
    training.checkpoint_every=10 \
    training.val_every=10

CHECKPOINT="${OUTPUT}/at/checkpoints/latest.ckpt"
test -s "${CHECKPOINT}"
echo "[piper-rdp-at] completed checkpoint=${CHECKPOINT}"
tail -n 1 "${OUTPUT}/at/logs.json.txt"
