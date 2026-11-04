#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=yangyc
#SBATCH --qos=user_wangzihao
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${UVIP_SOURCE:?immutable source directory}"
: "${UVIP_OUTPUT:?output directory}"
: "${UVIP_CONFIG:?run JSON}"
: "${UVIP_MODEL:?ftp1 or tabero}"
allocation="$(scontrol show job "${SLURM_JOB_ID}" -o)"
if [[ " ${allocation} " != *' Account=yangyc '* || " ${allocation} " != *' QOS=user_wangzihao '* ]]; then
    echo 'Expected the normal yangyc / user_wangzihao allocation' >&2
    exit 1
fi
cd "${UVIP_SOURCE}"
python_bin=/public/home/wangzihao/.venvs/uvip-ftp1/bin/python
native_source="${UVIP_SOURCE}/native/${UVIP_MODEL}"
export PYTHONPATH="${UVIP_SOURCE}:${native_source}/src:${native_source}/packages/openpi-client/src:/public/home/wangzihao/.venvs/uvip-advanced-extra"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TOKENIZERS_PARALLELISM=false UVIP_SLURM_ACCOUNT=yangyc
if [[ "${UVIP_MODEL}" == ftp1 ]]; then export JAX_PLATFORMS=cpu; fi
export UVIP_STOP_FILE="${UVIP_OUTPUT}/stop_${SLURM_JOB_ID}.txt"
export TMPDIR="/dev/shm/uvip_${SLURM_JOB_ID}"
export JAX_COMPILATION_CACHE_DIR="${UVIP_OUTPUT}/jax_cache"
mkdir -p "${TMPDIR}" "${UVIP_OUTPUT}"
exec 9>"${UVIP_OUTPUT}/run.lock"
flock -n 9 || exit 0
[[ ! -f "${UVIP_OUTPUT}/COMPLETE.json" ]] || exit 0
rm -f -- "${UVIP_STOP_FILE}"
trap 'printf "scheduler_signal\n" > "${UVIP_STOP_FILE}"' USR1 TERM
sha256sum --check source.sha256 > "${UVIP_OUTPUT}/source_check_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
"${python_bin}" - <<'PY'
import os, torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert 'H200' in torch.cuda.get_device_name(0)
free,total = torch.cuda.mem_get_info()
assert free > 90 * 2**30, (free,total)
print({'device':torch.cuda.get_device_name(0),'free':free,'total':total,
       'affinity':sorted(os.sched_getaffinity(0)), 'account':'yangyc',
       'qos':'user_wangzihao','job_id':os.environ['SLURM_JOB_ID']},flush=True)
PY
# Independent zero-filled memory, outside model, optimizer and random streams.
guard="${UVIP_OUTPUT}/gpu_memory_guard/${SLURM_JOB_ID}"
mkdir -p "${guard}"
ln -sfn "${UVIP_OUTPUT}/COMPLETE.json" "${guard}/COMPLETE.json"
ln -sfn "${UVIP_STOP_FILE}" "${guard}/SAFE_STOP.json"
rm -f -- "${guard}/gpu_memory_reservation.json"
"${python_bin}" -u "${UVIP_MEMORY_HOLDER_SCRIPT:-/public/home/wangzihao/projects/AdaViP-main/scripts/manifeel/reserve_teacher_gpu_memory.py}" \
    --hold --fraction .16 --output "${guard}" > "${guard}/holder.log" 2>&1 &
holder_pid=$!
monitor_pid=''
cleanup() {
    kill "${holder_pid}" ${monitor_pid:+"${monitor_pid}"} 2>/dev/null || true
    rmdir "${TMPDIR}" 2>/dev/null || true
}
trap cleanup EXIT
# Do not begin model loading until the independent reservation is confirmed.
ready=0
for attempt in $(seq 1 60); do
    if ! kill -0 "${holder_pid}" 2>/dev/null; then
        echo 'VRAM reservation process exited before becoming ready' >&2
        exit 1
    fi
    if [[ -s "${guard}/gpu_memory_reservation.json" ]]; then ready=1; break; fi
    sleep 1
done
[[ "${ready}" == 1 ]] || { echo 'VRAM reservation readiness timed out' >&2; exit 1; }
"${python_bin}" - "${guard}/gpu_memory_reservation.json" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1]))
assert str(r['job_id']) == os.environ['SLURM_JOB_ID']
assert r['reserved_bytes'] / r['total_bytes'] > .15, r
print(json.dumps({'event':'vram_floor_verified','job_id':r['job_id'],
                  'reserved_fraction':r['reserved_bytes']/r['total_bytes'],
                  'reserved_bytes':r['reserved_bytes'],'total_bytes':r['total_bytes']}),flush=True)
PY
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
        if ! kill -0 "${holder_pid}" 2>/dev/null; then
            printf 'VRAM_holder_exited\n' > "${UVIP_STOP_FILE}"
            echo 'VRAM holder exited; requesting checkpoint and requeue' >&2
            exit 1
        fi
        sleep 60
    done
) >> "${UVIP_OUTPUT}/gpu_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${python_bin}" -m adavip.advanced.entrypoint \
    --model "${UVIP_MODEL}" --config "${UVIP_CONFIG}" --output "${UVIP_OUTPUT}" &
trainer_pid=$!
result=0
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    break
done
if [[ -f "${UVIP_OUTPUT}/COMPLETE.json" ]]; then
    exit 0
elif [[ -f "${UVIP_STOP_FILE}" ]]; then
    scontrol requeue "${SLURM_JOB_ID}"
else
    if [[ "${result}" == 0 ]]; then result=1; fi
    exit "${result}"
fi
