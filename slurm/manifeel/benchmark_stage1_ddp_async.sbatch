#!/usr/bin/env bash
#SBATCH --job-name=stage1_ddp4_async_gb128
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=server09
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_ASYNC_ROOT:?Set an independent experiment root}"
: "${ADAVIP_SOURCE_ROOT:?Set the frozen GPU benchmark source}"
: "${ADAVIP_PRODUCTION_ROOT:?Set the read-only production root}"
: "${ADAVIP_REPO_ROOT:?Set the repository environment root}"
source "${ADAVIP_REPO_ROOT}/scripts/manifeel/common.sh"
activate_manifeel
export PYTHONPATH="${ADAVIP_SOURCE_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export TMPDIR="/public/home/wangzihao/.cache/avs1v2/ddp_async_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
cd "${ADAVIP_SOURCE_ROOT}"
echo "node=$(hostname) job=${SLURM_JOB_ID} source=${ADAVIP_SOURCE_ROOT}"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi topo -m
srun --unbuffered "${MANIFEEL_PYTHON}" scripts/manifeel/benchmark_stage1_ddp_async.py \
    --production-dir "${ADAVIP_PRODUCTION_ROOT}" \
    --output-dir "${ADAVIP_ASYNC_ROOT}/preflight_${SLURM_JOB_ID}" \
    --block-index "${ADAVIP_ASYNC_ROOT}/block_index.json" \
    --steps 24 --warmup 8 --cases copy_stream:4 pinned:4
srun --unbuffered "${MANIFEEL_PYTHON}" scripts/manifeel/benchmark_stage1_ddp_async.py \
    --production-dir "${ADAVIP_PRODUCTION_ROOT}" \
    --output-dir "${ADAVIP_ASYNC_ROOT}/stable_${SLURM_JOB_ID}" \
    --block-index "${ADAVIP_ASYNC_ROOT}/block_index.json" \
    --steps 2000 --warmup 100 --cases copy_stream:4 pinned:4 copy_stream:1
