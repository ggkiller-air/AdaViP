#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=yangyc
#SBATCH --qos=user_wangzihao
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${TBL_UVIP_SOURCE:?immutable source directory}"
: "${TBL_UVIP_OUTPUT:?run directory}"
cd "${TBL_UVIP_SOURCE}"
source /public/home/wangzihao/projects/AdaViP-main/scripts/manifeel/common.sh
activate_manifeel
export PYTHONPATH="${TBL_UVIP_SOURCE}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TABERO_UVIP_STOP_FILE="${TBL_UVIP_OUTPUT}/STOP_REQUEST"
export TMPDIR="/dev/shm/tabero_uvip_${SLURM_JOB_ID}"
mkdir -p "${TBL_UVIP_OUTPUT}" "${TMPDIR}"
exec 9>"${TBL_UVIP_OUTPUT}/pipeline.lock"
flock -n 9 || exit 0
[[ ! -f "${TBL_UVIP_OUTPUT}/train/COMPLETE.json" ]] || exit 0
rm -f -- "${TABERO_UVIP_STOP_FILE}"
trap 'touch "${TABERO_UVIP_STOP_FILE}"' USR1 TERM
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
taskset -pc $$
"${MANIFEEL_PYTHON}" - <<'PY'
import torch
assert torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0)
free,total=torch.cuda.mem_get_info()
assert free > 90*2**30, (free,total)
PY
guard="${TBL_UVIP_OUTPUT}/gpu_memory_guard/${SLURM_JOB_ID}"
mkdir -p "${guard}"
ln -sfn "${TBL_UVIP_OUTPUT}/train/COMPLETE.json" "${guard}/COMPLETE.json"
ln -sfn "${TABERO_UVIP_STOP_FILE}" "${guard}/SAFE_STOP.json"
"${MANIFEEL_PYTHON}" -u /public/home/wangzihao/projects/AdaViP-main/scripts/manifeel/reserve_teacher_gpu_memory.py \
    --hold --fraction .16 --output "${guard}" > "${guard}/holder.log" 2>&1 &
holder_pid=$!
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
        sleep 60
    done
) >> "${TBL_UVIP_OUTPUT}/gpu_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() {
    kill "${holder_pid}" "${monitor_pid}" 2>/dev/null || true
    rmdir "${TMPDIR}" 2>/dev/null || true
}
trap cleanup EXIT
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${MANIFEEL_PYTHON}" -m adavip.tabero.uvip_train \
    --config "${TBL_UVIP_OUTPUT}/launch_config.yaml" --output "${TBL_UVIP_OUTPUT}" &
trainer_pid=$!
result=0
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    break
done
if [[ -f "${TBL_UVIP_OUTPUT}/train/COMPLETE.json" ]]; then
    exit 0
elif [[ -f "${TABERO_UVIP_STOP_FILE}" || "${result}" == 143 || "${result}" == 137 || "${result}" == 15 ]]; then
    scontrol requeue "${SLURM_JOB_ID}"
else
    if [[ "${result}" == 0 ]]; then result=1; fi
    exit "${result}"
fi
