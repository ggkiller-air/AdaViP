#!/usr/bin/env bash
#SBATCH --job-name=full_sensor_ddp4_batch64_preflight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:NVIDIAH200:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_SOURCE_ROOT:?Submit with an immutable source snapshot}"
: "${ADAVIP_RUN_DIR:?Set a persistent unique run directory}"
: "${ADAVIP_CACHE_DIR:?Prepare the decoded cache first}"
cd "${ADAVIP_SOURCE_ROOT}"
source scripts/manifeel/common.sh
activate_manifeel
export TMPDIR="/dev/shm/avip_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}" "${ADAVIP_RUN_DIR}"
chmod 700 "${TMPDIR}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ADAVIP_STOP_FILE="${TMPDIR}/stop_requested"
trap 'printf "SIGUSR1\n" > "${ADAVIP_STOP_FILE}"' USR1
trap 'printf "SIGTERM\n" > "${ADAVIP_STOP_FILE}"' TERM
echo "node=$(hostname); CPUs=$(taskset -pc $$); DDP4 x batch16 = global64; read-only shared mmap"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
df -h /dev/shm
timeout 90s "${MANIFEEL_PYTHON}" scripts/manifeel/check_training_runtime.py
test -f "${ADAVIP_CACHE_DIR}/manifest.json"
if [[ "${ADAVIP_MODE:-preflight}" == long ]]; then
    : "${ADAVIP_VALIDATION_REPORT:?Long training requires the passed DDP preflight report}"
    "${MANIFEEL_PYTHON}" - "${ADAVIP_VALIDATION_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
required = report.get("required_speedup", 1.15)
if (not report.get("passed") or report.get("speedup", 0) < required
        or report.get("full_epoch_speedup", 0) < required):
    raise SystemExit("Refusing long training without a passed measured DDP preflight")
PY
fi
(
    while true; do
        date --iso-8601=seconds
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
        sleep 60
    done
) >> "${ADAVIP_RUN_DIR}/gpu_monitor.log" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; rmdir "${TMPDIR}" 2>/dev/null || true' EXIT
run_ddp() {
    local stop_epochs="$1" stop_steps="$2" resume="$3"
    local resume_args=() trainer_pid status
    if [[ -n "${resume}" ]]; then
        resume_args+=("resume=${resume}")
    fi
    srun --unbuffered "${MANIFEEL_PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    scripts/manifeel/train_full_sensor_reference.py \
    "output_dir=${ADAVIP_RUN_DIR}" name=full_sensor_reference_batch64 \
    "dataset.decoded_cache_dir=${ADAVIP_CACHE_DIR}" dataset.decoded_cache_mode=mmap \
    dataloader.batch_size=64 val_dataloader.batch_size=64 \
    dataloader.num_workers=3 val_dataloader.num_workers=2 \
    "training.num_epochs=${ADAVIP_EPOCHS:-300}" training.max_train_steps=null training.max_val_steps=null \
    training.checkpoint_every=100 'training.checkpoint_epochs=[1,2,5,10,20,50]' \
    runtime.preflight_windows=5 "runtime.min_samples_per_second=${ADAVIP_MIN_THROUGHPUT:-300}" \
    runtime.max_data_wait_fraction=0.35 "runtime.stop_after_epochs=${stop_epochs}" \
    "runtime.stop_after_steps=${stop_steps}" \
    "runtime.allow_epoch_extension=${ADAVIP_ALLOW_EXTENSION:-false}" "${resume_args[@]}" &
    trainer_pid=$!
    status=0
    wait "${trainer_pid}" || status=$?
    if kill -0 "${trainer_pid}" 2>/dev/null; then
        status=0
        wait "${trainer_pid}" || status=$?
    fi
    return "${status}"
}

run_ddp "${ADAVIP_STOP_EPOCHS:-2}" "${ADAVIP_STOP_STEPS:-0}" "${ADAVIP_RESUME:-}"
if [[ "${ADAVIP_MODE:-preflight}" == preflight && ! -f "${ADAVIP_STOP_FILE}" ]]; then
    run_ddp 0 120 "${ADAVIP_RUN_DIR}/checkpoints/latest.ckpt"
    "${MANIFEEL_PYTHON}" scripts/manifeel/assess_full_sensor_ddp.py \
        --run "${ADAVIP_RUN_DIR}" --baseline-run "${ADAVIP_BASELINE_RUN:?Set the live single-GPU baseline}" \
        --min-speedup 1.15
elif [[ "${ADAVIP_MODE:-preflight}" == long && -f "${ADAVIP_STOP_FILE}" ]] \
        && [[ "$(cat "${ADAVIP_STOP_FILE}")" == SIGUSR1 ]]; then
    if ! "${MANIFEEL_PYTHON}" - "${ADAVIP_RUN_DIR}/progress.jsonl" "${ADAVIP_EPOCHS:-300}" <<'PY'
import json
import sys
from pathlib import Path

last = json.loads(Path(sys.argv[1]).read_text().splitlines()[-1])
sys.exit(0 if last.get("event") == "stopped" and last.get("epoch", 0) < int(sys.argv[2]) else 1)
PY
    then
        exit 0
    fi
    export ADAVIP_RESUME="${ADAVIP_RUN_DIR}/checkpoints/latest.ckpt"
    sbatch --account="${SLURM_JOB_ACCOUNT}" --dependency="afterok:${SLURM_JOB_ID}" \
        --time=2-00:00:00 --time-min=06:00:00 --job-name=full_sensor_ddp4_batch64_ep1000 \
        --export=ALL "${ADAVIP_SOURCE_ROOT}/slurm/manifeel/train_full_sensor_ddp_batch64.sbatch" \
        | tee -a "${ADAVIP_RUN_DIR}/continuation_jobs.log"
fi
