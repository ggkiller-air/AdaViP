#!/usr/bin/env bash
#SBATCH --job-name=dp_infer_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/slurm/%x_%j.err

set -euo pipefail

cd /public/home/wangzihao/projects/AdaViP

# shellcheck source=../../scripts/real_world/common.sh
source scripts/real_world/common.sh
activate_rdp
print_dp_sanity_context
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

REPORT_DIR="/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/inference"
mkdir -p "${REPORT_DIR}"

exec "${RDP_PYTHON}" scripts/real_world/infer_dp_sanity.py \
    --output-json "${REPORT_DIR}/inference_smoke_${SLURM_JOB_ID}.json" \
    "$@"
