#!/usr/bin/env bash
#SBATCH --account=yangyc
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${UVIP_SOURCE:?immutable source directory}"
: "${UVIP_OUTPUT:?output directory}"
: "${UVIP_CONFIG:?run JSON}"
: "${UVIP_MODEL:?ftp1 or tabero}"
cd "${UVIP_SOURCE}"
python_bin=/public/home/wangzihao/.venvs/uvip-ftp1/bin/python
native_source="${UVIP_SOURCE}/native/${UVIP_MODEL}"
export PYTHONPATH="${UVIP_SOURCE}:${native_source}/src:${native_source}/packages/openpi-client/src:/public/home/wangzihao/.venvs/uvip-advanced-extra"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TOKENIZERS_PARALLELISM=false
if [[ "${UVIP_MODEL}" == ftp1 ]]; then export JAX_PLATFORMS=cpu; fi
export UVIP_STOP_FILE="${UVIP_OUTPUT}/stop_${SLURM_JOB_ID}.txt"
export TMPDIR="/dev/shm/uvip_${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="${UVIP_OUTPUT}/jax_cache"
mkdir -p "${TMPDIR}" "${UVIP_OUTPUT}"
exec 9>"${UVIP_OUTPUT}/run.lock"
flock -n 9 || exit 0
if [[ -f "${UVIP_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
rm -f -- "${UVIP_STOP_FILE}"
trap 'printf "SIGUSR1\n" > "${UVIP_STOP_FILE}"' USR1
trap 'printf "SIGTERM\n" > "${UVIP_STOP_FILE}"' TERM
sha256sum --check source.sha256 > "${UVIP_OUTPUT}/source_check_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
"${python_bin}" - <<'PY'
import os,torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert 'H200' in torch.cuda.get_device_name(0)
free,total=torch.cuda.mem_get_info()
assert free > 90*2**30, (free,total)
print({'device':torch.cuda.get_device_name(0),'free':free,'total':total,
       'affinity':sorted(os.sched_getaffinity(0)),'slurm_job':os.environ['SLURM_JOB_ID']},flush=True)
PY
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
        sleep 60
    done
) >> "${UVIP_OUTPUT}/gpu_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; rmdir "${TMPDIR}" 2>/dev/null || true' EXIT
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${python_bin}" -m adavip.advanced.entrypoint \
    --model "${UVIP_MODEL}" --config "${UVIP_CONFIG}" --output "${UVIP_OUTPUT}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ ! -f "${UVIP_OUTPUT}/COMPLETE.json" ]]; then
    "${python_bin}" -m adavip.advanced.scheduler \
        --source "${UVIP_SOURCE}" --output "${UVIP_OUTPUT}" --config "${UVIP_CONFIG}" \
        --model "${UVIP_MODEL}" --dependency "afterok:${SLURM_JOB_ID}"
fi
