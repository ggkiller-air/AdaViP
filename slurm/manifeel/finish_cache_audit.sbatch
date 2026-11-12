#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --account=overload
#SBATCH --qos=overload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
: "${CACHE_AUDIT_SOURCE:?source directory}"
: "${CACHE_AUDIT_OUTPUT:?audit directory}"
: "${CACHE_ARRAY_START:?first array}"
: "${CACHE_ARRAY_STOP:?exclusive last array}"
cd "${CACHE_AUDIT_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
export PYTHONDONTWRITEBYTECODE=1
if [[ "${CACHE_AUDIT_TESTS:-0}" == 1 ]]; then
    "${MANIFEEL_PYTHON}" -m pytest -q tests/test_stage2_parallel.py > "${CACHE_AUDIT_OUTPUT}/optimizer_reset_pytest.log" 2>&1
fi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${MANIFEEL_PYTHON}" -u -m scripts.manifeel.finish_cache_audit \
    --output "${CACHE_AUDIT_OUTPUT}" --start "${CACHE_ARRAY_START}" --stop "${CACHE_ARRAY_STOP}"
