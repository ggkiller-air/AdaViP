#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --account=overload
#SBATCH --qos=overload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
: "${CACHE_AUDIT_SOURCE:?source directory}"
: "${CACHE_AUDIT_OUTPUT:?isolated audit directory}"
: "${CACHE_AUDIT_PRODUCTION:?read-only production reference}"
cd "${CACHE_AUDIT_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${CACHE_AUDIT_OUTPUT}"
"${MANIFEEL_PYTHON}" -m pytest -q tests/test_streaming_cache.py tests/test_stage2_parallel.py > "${CACHE_AUDIT_OUTPUT}/pytest.log" 2>&1
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${MANIFEEL_PYTHON}" -u -m scripts.manifeel.audit_stage2_cache \
    --output "${CACHE_AUDIT_OUTPUT}" --production "${CACHE_AUDIT_PRODUCTION}"
