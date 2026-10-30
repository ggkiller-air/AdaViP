#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=yangyc
#SBATCH --qos=user_wangzihao
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err
set -euo pipefail
: "${INFRA_SOURCE:?}"
: "${INFRA_OUTPUT:?}"
: "${INFRA_STRATEGY:?}"
cd "${INFRA_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TMPDIR="/public/home/wangzihao/.cache/lc/i/b_${SLURM_JOB_ID}"
export INFRA_STOP_FILE="${INFRA_OUTPUT}/STOP_REQUEST"
export TORCHINDUCTOR_CACHE_DIR="${INFRA_OUTPUT}/inductor_cache"
export TRITON_CACHE_DIR="${INFRA_OUTPUT}/triton_cache"
mkdir -p "${INFRA_OUTPUT}"
exec 9>"${INFRA_OUTPUT}/benchmark.lock"
flock -n 9 || exit 0
rm -f -- "${INFRA_STOP_FILE}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi topo -m || true
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used,power.draw --format=csv,noheader
        sleep 5
    done
) >> "${INFRA_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor=$!
trap 'kill "${monitor}" 2>/dev/null || true' EXIT
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    bash scripts/manifeel/run_language_step.sh "${MANIFEEL_PYTHON}" -u \
    scripts/manifeel/benchmark_stage2_infra.py --output "${INFRA_OUTPUT}" --strategy "${INFRA_STRATEGY}" &
trainer=$!
trap 'touch "${INFRA_STOP_FILE}"' USR1 TERM
while true; do
    if wait "${trainer}"; then break; else result=$?; fi
    if kill -0 "${trainer}" 2>/dev/null; then continue; fi
    exit "${result}"
done
