#!/usr/bin/env bash
#SBATCH --job-name=dp_export
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/slurm/%x_%j.err

set -euo pipefail

cd /public/home/wangzihao/projects/AdaViP

# shellcheck source=../../scripts/real_world/common.sh
source scripts/real_world/common.sh
activate_rdp

echo "[export] hostname=$(hostname)"
"${RDP_PYTHON}" scripts/real_world/export_dp_sanity_inference.py "$@"

ARTIFACT="/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/inference/dp_sanity_no_tactile_ema.pt"
ls -lh "${ARTIFACT}"
sha256sum "${ARTIFACT}"
