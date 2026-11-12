#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=overload
#SBATCH --qos=overload
#SBATCH --gres=gpu:NVIDIAH200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=00:45:00
set -euo pipefail
: "${EQUAL_SOURCE:?frozen source}"
: "${EQUAL_OUTPUT:?smoke output}"
: "${EQUAL_BATCH:?global batch}"
: "${EQUAL_GPUS:?world size}"
cd "${EQUAL_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR="/dev/shm/equal_smoke_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${EQUAL_OUTPUT}"
trap 'rmdir "${TMPDIR}" 2>/dev/null || true' EXIT
exec 9>"${EQUAL_OUTPUT}/smoke.lock"
flock -n 9 || exit 0
sha256sum --check source.sha256 > "${EQUAL_OUTPUT}/source_check.log"
scontrol show job "${SLURM_JOB_ID}" -dd
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
taskset -pc $$
"${MANIFEEL_PYTHON}" -m pytest -q tests/test_stage2_equal_updates.py > "${EQUAL_OUTPUT}/pytest.log" 2>&1
for phase in first restore; do
    srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
        "${MANIFEEL_PYTHON}" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${EQUAL_GPUS}" \
        -m scripts.manifeel.train_stage2_equal_updates \
        --output "${EQUAL_OUTPUT}" --batch "${EQUAL_BATCH}" --workers 2 --smoke-phase "${phase}"
done
