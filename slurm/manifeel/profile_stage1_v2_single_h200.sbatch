#!/usr/bin/env bash
#SBATCH --job-name=stage1_profile_single_h200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Set the original frozen training source}"
: "${ADAVIP_PROFILE_ROOT:?Set the independently prepared profiling directory}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export TMPDIR="/public/home/wangzihao/.cache/avs1v2/profile_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
echo "node=$(hostname) job=${SLURM_JOB_ID} source=${ADAVIP_SOURCE_ROOT}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
    --format=csv -l 1 > "${ADAVIP_PROFILE_ROOT}/gpu_${SLURM_JOB_ID}.csv" &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
srun --unbuffered "${MANIFEEL_PYTHON}" "${ADAVIP_PROFILE_ROOT}/source/profile_stage1_v2.py" \
    --fixture-dir "${ADAVIP_PROFILE_ROOT}" \
    --output-dir "${ADAVIP_PROFILE_ROOT}/job_${SLURM_JOB_ID}" --steps 80 --warmup 10 --workers 8
