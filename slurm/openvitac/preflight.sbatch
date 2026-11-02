#!/usr/bin/env bash
#SBATCH --job-name=openvitac_bounded_smoke
#SBATCH --partition=gpu,gpux
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/data/wangzihao/outputs/openvitac/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/openvitac/slurm/%x_%j.err
set -euo pipefail
cd "${OPENVITAC_SOURCE_ROOT:-/public/home/wangzihao/projects/AdaViP-main}"
source scripts/openvitac/common.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
output="/data/wangzihao/outputs/openvitac/gpu_smoke_${SLURM_JOB_ID}"
exec srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
    "$OPENVITAC_PYTHON" scripts/openvitac/smoke.py \
    --root /data/wangzihao/datasets/openvitac/converted/uvip_v1_5fb2757b \
    --output "$output" --device cuda:0 --full-model --workers 2
