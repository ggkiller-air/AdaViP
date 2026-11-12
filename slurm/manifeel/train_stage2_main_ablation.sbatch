#!/usr/bin/env bash
#SBATCH --partition=gpu,gpux
#SBATCH --account=yangyc
#SBATCH --qos=user_wangzihao
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
set -euo pipefail
: "${L1S2_SOURCE:?immutable source}"
: "${L1S2_OUTPUT:?run directory}"
: "${L1S2_ABLATION:?B C D E F}"
: "${L1S2_MODE:?smoke or train}"
cd "${L1S2_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if [[ "${L1S2_CACHE_MODE:-eager}" == "streaming" ]]; then
    export NUMPY_MADVISE_HUGEPAGE=0
fi
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="/dev/shm/l1s2_${SLURM_JOB_ID}"
mkdir -p "${L1S2_OUTPUT}" "${TMPDIR}"
exec 9>"${L1S2_OUTPUT}/training.lock"
flock -n 9 || exit 0
[[ ! -f "${L1S2_OUTPUT}/COMPLETE.json" ]] || exit 0
export ADAVIP_STOP_FILE="${L1S2_OUTPUT}/STOP_REQUEST"
rm -f -- "${ADAVIP_STOP_FILE}" "${L1S2_OUTPUT}/SAFE_STOP.json"
sha256sum --check source.sha256 > "${L1S2_OUTPUT}/source_check_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
taskset -pc $$
df -h /dev/shm
"${MANIFEEL_PYTHON}" - <<'PY'
import torch
assert torch.cuda.device_count() == 1 and 'H200' in torch.cuda.get_device_name(0)
free,total=torch.cuda.mem_get_info()
assert free > 90*2**30, (free,total)
PY
# Administrator-authorized buffer, isolated from training state and RNG.
guard="${L1S2_OUTPUT}/gpu_memory_guard/${SLURM_JOB_ID}"
mkdir -p "${guard}"
ln -sfn "${L1S2_OUTPUT}/COMPLETE.json" "${guard}/COMPLETE.json"
ln -sfn "${ADAVIP_STOP_FILE}" "${guard}/SAFE_STOP.json"
"${MANIFEEL_PYTHON}" -u scripts/manifeel/reserve_teacher_gpu_memory.py \
    --hold --fraction .16 --output "${guard}" > "${guard}/holder.log" 2>&1 &
holder_pid=$!
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used --format=csv,noheader
        sleep 60
    done
) >> "${L1S2_OUTPUT}/gpu_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
cleanup() {
    kill "${holder_pid}" "${monitor_pid}" 2>/dev/null || true
    rmdir "${TMPDIR}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'printf "scheduler_signal\n" > "${ADAVIP_STOP_FILE}"' USR1 TERM
arguments=(
    --teacher /data/wangzihao/outputs/manifeel/language_conditioning_t2_batch64_ep500_seed42_20261028_204459/l1/checkpoints/epoch50.ckpt
    --labels /data/wangzihao/outputs/manifeel/l1_epoch50_dropout_comparison_k64_20261102_201244/original
    --cache /data/wangzihao/cache/manifeel/full_sensor_float32_9task_v1_seed42_20261017
    --cache-config /data/wangzihao/outputs/manifeel/full_sensor_cache_build_20261017_134757/dataset_config.yaml
    --output "${L1S2_OUTPUT}" --workers 8 --cache-workers 8 --ablation "${L1S2_ABLATION}"
    --cache-mode "${L1S2_CACHE_MODE:-eager}"
)
if [[ "${L1S2_MODE}" == "smoke" ]]; then
    arguments+=(--smoke)
else
    : "${L1S2_SMOKE_GATE:?directory containing all five successful smokes}"
    "${MANIFEEL_PYTHON}" - <<'GATE'
import json, os
from pathlib import Path
root = Path(os.environ['L1S2_SMOKE_GATE'])
for letter in 'BCDEF':
    report = json.loads((root / letter / 'SMOKE_PASS.json').read_text())
    assert report['status'] == 'PASS' and report['ablation'] == letter
assert os.environ['SLURM_JOB_ACCOUNT'] == 'yangyc'
GATE
fi
if [[ -f "${L1S2_OUTPUT}/checkpoints/latest.ckpt" ]]; then
    arguments+=(--resume "${L1S2_OUTPUT}/checkpoints/latest.ckpt")
    if [[ "${L1S2_CACHE_MODE:-eager}" == "streaming" ]]; then
        arguments+=(--resume-data-transport)
    fi
fi
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    bash scripts/manifeel/run_stage2_step.sh "${MANIFEEL_PYTHON}" -u \
    -m scripts.manifeel.train_stage2_l1_s0 "${arguments[@]}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ -f "${L1S2_OUTPUT}/SAFE_STOP.json" && ! -f "${L1S2_OUTPUT}/COMPLETE.json" ]]; then
    scontrol requeue "${SLURM_JOB_ID}"
fi
