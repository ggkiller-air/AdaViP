#!/usr/bin/env bash
#SBATCH --job-name=piper-rdp-ldp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${ADAVIP_RDP_DATASET:?Export ADAVIP_RDP_DATASET before sbatch}"
: "${ADAVIP_RDP_OUTPUT:?Export ADAVIP_RDP_OUTPUT before sbatch}"
REPO_ROOT="${ADAVIP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATASET="${ADAVIP_RDP_DATASET}"
OUTPUT="${ADAVIP_RDP_OUTPUT}"
AT_CHECKPOINT="${ADAVIP_RDP_AT_CHECKPOINT:-${OUTPUT}/at/checkpoints/latest.ckpt}"
export ADAVIP_RDP_DATASET ADAVIP_RDP_OUTPUT AT_CHECKPOINT

cd "${REPO_ROOT}"
# shellcheck source=../../scripts/real_world/common.sh
source scripts/real_world/common.sh
activate_rdp

export WANDB_MODE=disabled
export OMP_NUM_THREADS=2

echo "[piper-rdp-ldp] job_id=${SLURM_JOB_ID} host=$(hostname)"
echo "[piper-rdp-ldp] dataset=${DATASET}"
echo "[piper-rdp-ldp] at_checkpoint=${AT_CHECKPOINT}"
echo "[piper-rdp-ldp] output=${OUTPUT}/ldp"
nvidia-smi \
    --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader

"${RDP_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
import torch

dataset = Path(os.environ["ADAVIP_RDP_DATASET"])
checkpoint = Path(os.environ["AT_CHECKPOINT"])
audit = json.loads((dataset / "audit/gelsight_audit.json").read_text())
assert audit["detection"]["success_rate"] == 1.0
assert audit["pca"]["components"] == 15
assert checkpoint.stat().st_size > 0
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert "model" in payload["state_dicts"]
assert set(payload["state_dicts"]["model"]) >= {"encoder", "decoder"}
print(
    "[piper-rdp-ldp] data_audit="
    f"{audit['frames']} frames, "
    f"AT checkpoint={checkpoint.stat().st_size} bytes"
)
PY

"${RDP_PYTHON}" scripts/real_world/train_piper_rdp.py \
    --stage ldp \
    --dataset "${DATASET}" \
    --output "${OUTPUT}" \
    --at-checkpoint "${AT_CHECKPOINT}" \
    --python "${RDP_PYTHON}" \
    --wandb-mode disabled \
    dataloader.batch_size=32 \
    dataloader.num_workers=8 \
    val_dataloader.batch_size=32 \
    val_dataloader.num_workers=0 \
    val_dataloader.persistent_workers=false \
    training.num_epochs=401 \
    training.checkpoint_every=10 \
    training.val_every=10 \
    training.sample_every=10000

CHECKPOINT="${OUTPUT}/ldp/checkpoints/latest.ckpt"
test -s "${CHECKPOINT}"
echo "[piper-rdp-ldp] completed checkpoint=${CHECKPOINT}"
tail -n 1 "${OUTPUT}/ldp/logs.json.txt"
