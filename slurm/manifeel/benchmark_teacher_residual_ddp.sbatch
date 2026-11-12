#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=overload
#SBATCH --qos=overload
#SBATCH --gres=gpu:NVIDIAH200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=00:30:00
set -euo pipefail
: "${PARALLEL_SOURCE:?frozen benchmark source}"
: "${PARALLEL_OUTPUT:?separate benchmark directory}"
: "${PARALLEL_CHECKPOINT:?same frozen checkpoint for both screens}"
: "${PARALLEL_GPUS:?two or four GPUs}"
cd "${PARALLEL_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${PARALLEL_OUTPUT}"
exec 9>"${PARALLEL_OUTPUT}/benchmark.lock"
flock -n 9 || exit 0
sha256sum --check source.sha256 > "${PARALLEL_OUTPUT}/source_check.log"
scontrol show job "${SLURM_JOB_ID}" -dd
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi topo -m
taskset -pc $$
profile_arguments=()
if [[ "${PARALLEL_PROFILE_PARTS:-0}" == 1 ]]; then
    profile_arguments+=(--profile-parts)
fi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${MANIFEEL_PYTHON}" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${PARALLEL_GPUS}" \
    -m scripts.manifeel.benchmark_teacher_residual_ddp \
    --checkpoint "${PARALLEL_CHECKPOINT}" --output "${PARALLEL_OUTPUT}" "${profile_arguments[@]}"
