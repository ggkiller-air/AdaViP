#!/usr/bin/env bash
#SBATCH --account=yangyc
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/experiments/tabero/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/experiments/tabero/slurm/%x_%j.err
set -euo pipefail
: "${TBL1_SOURCE:?Set immutable source directory}"
: "${TBL1_OUTPUT:?Set run output directory}"
cd "${TBL1_SOURCE}"
python_bin="${TABERO_DP_PYTHON:-/public/home/wangzihao/.local/miniforge3/envs/manifeel/bin/python}"
export PYTHONPATH="${TBL1_SOURCE}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export NUMPY_MADVISE_HUGEPAGE=0
export TABERO_DP_STOP_FILE="${TBL1_OUTPUT}/stop_${SLURM_JOB_ID}.txt"
export TMPDIR="/dev/shm/tbl1_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${TBL1_OUTPUT}"
exec 9>"${TBL1_OUTPUT}/training.lock"
flock -n 9 || exit 0
if [[ -f "${TBL1_OUTPUT}/COMPLETE.json" ]]; then exit 0; fi
rm -f -- "${TABERO_DP_STOP_FILE}"
trap 'printf "SIGUSR1\n" > "${TABERO_DP_STOP_FILE}"' USR1
trap 'printf "SIGTERM\n" > "${TABERO_DP_STOP_FILE}"' TERM
sha256sum --check source.sha256 > "${TBL1_OUTPUT}/source_check_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
"${python_bin}" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Exactly one allocated CUDA device is required")
if "H200" not in torch.cuda.get_device_name(0):
    raise RuntimeError("Expected the allocated H200")
free, total = torch.cuda.mem_get_info(0)
if free < 60 * 2**30:
    raise RuntimeError("Insufficient free VRAM; leave unrelated processes untouched")
print({"gpu": torch.cuda.get_device_name(0), "free_bytes": free, "total_bytes": total}, flush=True)
PY
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
        sleep 60
    done
) >> "${TBL1_OUTPUT}/gpu_monitor_${SLURM_JOB_ID}.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; rmdir "${TMPDIR}" 2>/dev/null || true' EXIT
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
    "${python_bin}" -m adavip.tabero.train \
    --config "${TBL1_OUTPUT}/launch_config.yaml" --output "${TBL1_OUTPUT}" &
trainer_pid=$!
while true; do
    if wait "${trainer_pid}"; then break; else result=$?; fi
    if kill -0 "${trainer_pid}" 2>/dev/null; then continue; fi
    exit "${result}"
done
if [[ ! -f "${TBL1_OUTPUT}/COMPLETE.json" ]]; then
    "${python_bin}" scripts/submit_partition_job.py \
        --prefix TBL1 --source "${TBL1_SOURCE}" --output "${TBL1_OUTPUT}" \
        --script slurm/tabero/l1_field_fs_train.sbatch --name tabero_L1_field_fs_b64 \
        --dependency "afterok:${SLURM_JOB_ID}"
fi
