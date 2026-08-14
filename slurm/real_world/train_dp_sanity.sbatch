#!/usr/bin/env bash
#SBATCH --job-name=dp_sanity
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/real_world/dp_sanity_no_tactile/slurm/%x_%j.err

set -euo pipefail

cd /public/home/wangzihao/projects/AdaViP
mkdir -p /data/wangzihao/outputs/real_world/dp_sanity_no_tactile/slurm

# Install the upstream DP's optional vision import only after Slurm assigns a
# compute node. The no-tactile training path itself does not use a timm encoder.
# shellcheck source=../../scripts/real_world/common.sh
source scripts/real_world/common.sh
activate_rdp
if ! "${RDP_PYTHON}" -c 'import timm, peft' >/dev/null 2>&1; then
    "${RDP_PYTHON}" -m pip install \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        timm peft
fi
"${RDP_PYTHON}" -c \
    'from reactive_diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace'

bash scripts/real_world/train_dp_sanity.sh "$@"
