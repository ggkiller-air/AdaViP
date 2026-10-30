#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=yangyc
#SBATCH --qos=user_wangzihao
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err
set -euo pipefail
: "${LC_BENCH_SOURCE:?}"
: "${LC_BENCH_OUTPUT:?}"
: "${LC_BENCH_CONFIG:?}"
: "${LC_BENCH_CHECKPOINT:?}"
cd "${LC_BENCH_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/public/home/wangzihao/.cache/lc/i/lb_${SLURM_JOB_ID}"
mkdir -p "${LC_BENCH_OUTPUT}" "${TMPDIR}"
exec 9>"${LC_BENCH_OUTPUT}/benchmark.lock"
flock -n 9 || exit 0
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used --format=csv,noheader
        sleep 30
    done
) >> "${LC_BENCH_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor=$!
trap 'kill "${monitor}" 2>/dev/null || true' EXIT
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    bash scripts/manifeel/run_language_step.sh "${MANIFEEL_PYTHON}" -u \
    scripts/manifeel/benchmark_language_runtime.py --config "${LC_BENCH_CONFIG}" \
    --checkpoint "${LC_BENCH_CHECKPOINT}" --output "${LC_BENCH_OUTPUT}"
