#!/usr/bin/env bash
#SBATCH --job-name=stage1_v2_parallel_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:NVIDIAH200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=4-00:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Set the frozen benchmark source}"
: "${ADAVIP_BENCHMARK_DIR:?Set an independent benchmark output}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export MANIFEEL_DECODED_CACHE_DIR=/data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017
export MANIFEEL_DECODED_CACHE_MODE=mmap
export TMPDIR="/public/home/wangzihao/.cache/avs1v2/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${ADAVIP_BENCHMARK_DIR}"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi topo -m
for world_size in ${ADAVIP_BENCHMARK_SIZES:-1 2}; do
    "${MANIFEEL_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${world_size}" \
        scripts/manifeel/benchmark_stage1_v2_parallel.py \
        --config configs/manifeel/stage1_v2_teacher.yaml \
        --output "${ADAVIP_BENCHMARK_DIR}/world${world_size}.json"
done
