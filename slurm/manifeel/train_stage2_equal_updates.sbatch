#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=yangyc
#SBATCH --qos=user_wangzihao
#SBATCH --gres=gpu:NVIDIAH200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${EQUAL_SOURCE:?immutable training source}"
: "${EQUAL_OUTPUT:?formal run output}"
: "${EQUAL_SMOKE:?passed same-batch smoke}"
: "${EQUAL_BATCH:?global batch}"
: "${EQUAL_GPUS:?world size}"
cd "${EQUAL_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/dev/shm/equal_train_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${EQUAL_OUTPUT}"
exec 9>"${EQUAL_OUTPUT}/training.lock"
flock -n 9 || exit 0
[[ ! -f "${EQUAL_OUTPUT}/COMPLETE.json" ]] || exit 0
export ADAVIP_STOP_FILE="${EQUAL_OUTPUT}/STOP_REQUEST"
rm -f -- "${ADAVIP_STOP_FILE}" "${EQUAL_OUTPUT}/SAFE_STOP.json"
sha256sum --check source.sha256 > "${EQUAL_OUTPUT}/source_check_${SLURM_JOB_ID}.log"
"${MANIFEEL_PYTHON}" - <<'PY'
import json, os
from pathlib import Path
import torch
report=json.loads((Path(os.environ['EQUAL_SMOKE'])/'SMOKE_PASS.json').read_text())
assert report['status']=='PASS' and report['fresh_process_resume_bitwise']
assert report['portable_raw_ema_full_sampling_bitwise']
assert report['batch']==int(os.environ['EQUAL_BATCH']) and report['world']==int(os.environ['EQUAL_GPUS'])
assert report['total_updates']==1224000 and report['comparison_updates']==[408000,816000]
assert torch.cuda.device_count()==int(os.environ['EQUAL_GPUS'])
for i in range(torch.cuda.device_count()):
    assert 'H200' in torch.cuda.get_device_name(i)
    free,total=torch.cuda.mem_get_info(i)
    assert free>90*2**30,(i,free,total)
PY
scontrol show job "${SLURM_JOB_ID}" -dd
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
taskset -pc $$
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader
        sleep 60
    done
) >> "${EQUAL_OUTPUT}/gpu_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() {
    kill "${monitor_pid}" 2>/dev/null || true
    rmdir "${TMPDIR}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'printf "scheduler_signal\n" > "${ADAVIP_STOP_FILE}"' USR1 TERM
arguments=(--output "${EQUAL_OUTPUT}" --batch "${EQUAL_BATCH}" --workers "${EQUAL_WORKERS:-2}")
if [[ -f "${EQUAL_OUTPUT}/checkpoints/latest.ckpt" ]]; then
    arguments+=(--resume "${EQUAL_OUTPUT}/checkpoints/latest.ckpt")
fi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${MANIFEEL_PYTHON}" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${EQUAL_GPUS}" \
    -m scripts.manifeel.train_stage2_equal_updates "${arguments[@]}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ -f "${EQUAL_OUTPUT}/SAFE_STOP.json" && ! -f "${EQUAL_OUTPUT}/COMPLETE.json" ]]; then
    scontrol requeue "${SLURM_JOB_ID}"
fi
