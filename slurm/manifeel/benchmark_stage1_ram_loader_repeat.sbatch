#!/usr/bin/env bash
#SBATCH --job-name=stage1_ram_repeat_batch32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_RAM_ROOT:?Set the independent repeat benchmark directory}"
: "${ADAVIP_PRODUCTION_ROOT:?Set the read-only production directory}"
: "${ADAVIP_REPO_ROOT:?Set the repository environment root}"
source "${ADAVIP_REPO_ROOT}/scripts/manifeel/common.sh"
activate_manifeel
export PYTHONPATH="${ADAVIP_RAM_ROOT}/source:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export TMPDIR="/public/home/wangzihao/.cache/avs1v2/ram_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
cd "${ADAVIP_RAM_ROOT}/source"
echo "node=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > "${ADAVIP_RAM_ROOT}/gpu_${SLURM_JOB_ID}.csv" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
for label in original_a cached_a cached_b original_b; do
    srun --unbuffered "${MANIFEEL_PYTHON}" scripts/manifeel/benchmark_stage1_ram_loader.py \
        --fixture-dir "${ADAVIP_RAM_ROOT}" --production-dir "${ADAVIP_PRODUCTION_ROOT}" \
        --case "${label%%_*}" --label "${label}" --steps 160 --warmup 24
done
"${MANIFEEL_PYTHON}" scripts/manifeel/benchmark_stage1_ram_loader.py --summarize \
    --fixture-dir "${ADAVIP_RAM_ROOT}" --production-dir "${ADAVIP_PRODUCTION_ROOT}" \
    --labels original_a cached_a cached_b original_b
