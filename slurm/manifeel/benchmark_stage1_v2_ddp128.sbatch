#!/usr/bin/env bash
#SBATCH --job-name=stage1_ddp4_gb128_steps2000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Set an independent frozen source snapshot}"
: "${ADAVIP_BENCHMARK_ROOT:?Set a unique benchmark root outside production}"
: "${ADAVIP_REFERENCE_PREFLIGHT:?Set the read-only formal preflight report}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export MANIFEEL_DECODED_CACHE_DIR=/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017
export MANIFEEL_DECODED_CACHE_MODE=ram
export TMPDIR="/public/home/wangzihao/.cache/avs1v2/${SLURM_JOB_ID}"
output="${ADAVIP_BENCHMARK_ROOT}/job_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${output}"
echo "node=$(hostname); source=${ADAVIP_SOURCE_ROOT}; output=${output}"
date --iso-8601=seconds
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi topo -m
"${MANIFEEL_PYTHON}" -m adavip.manifeel.stage1_ddp_benchmark \
    --config configs/manifeel/stage1_v2_teacher.yaml --output-dir "${output}" \
    --reference-preflight "${ADAVIP_REFERENCE_PREFLIGHT}" \
    --world-sizes 4 1 --steps 2000 --warmup 100 --workers 8
