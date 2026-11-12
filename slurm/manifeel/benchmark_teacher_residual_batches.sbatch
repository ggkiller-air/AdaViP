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
: "${PARALLEL_SOURCE:?frozen benchmark source}"
: "${PARALLEL_OUTPUT:?separate benchmark directory}"
: "${PARALLEL_CHECKPOINT:?same frozen checkpoint}"
cd "${PARALLEL_SOURCE}"
source scripts/manifeel/common.sh
activate_manifeel
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMPY_MADVISE_HUGEPAGE=0
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${PARALLEL_OUTPUT}"
exec 9>"${PARALLEL_OUTPUT}/sweep.lock"
flock -n 9 || exit 0
sha256sum --check source.sha256 > "${PARALLEL_OUTPUT}/source_check.log"
scontrol show job "${SLURM_JOB_ID}" -dd
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.free,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi topo -m
taskset -pc $$
# Same allocation/physical primary GPU for all comparisons. Each case is fresh,
# bounded, and restores the identical checkpoint. No benchmark weights survive.
read -r -a batch_cases <<< "${PARALLEL_BATCH_CASES:-1:32 2:64 4:128 1:64 2:128 1:128 4:256}"
for specification in "${batch_cases[@]}"; do
    devices="${specification%%:*}"
    batch="${specification##*:}"
    output="${PARALLEL_OUTPUT}/gpu${devices}_batch${batch}"
    if [[ -f "${output}/REPORT.json" ]]; then continue; fi
    mkdir -p "${output}"
    if [[ -f "${output}/exit_code.txt" ]]; then
        for artifact in stdout.log stderr.log exit_code.txt; do
            if [[ -f "${output}/${artifact}" ]]; then
                mv "${output}/${artifact}" "${output}/${artifact}.before_${SLURM_JOB_ID}"
            fi
        done
    fi
    printf 'START devices=%s global_batch=%s\n' "${devices}" "${batch}"
    if timeout --signal=TERM --kill-after=30s 360s \
        srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores --unbuffered \
        "${MANIFEEL_PYTHON}" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${devices}" \
        -m scripts.manifeel.benchmark_teacher_residual_ddp \
        --checkpoint "${PARALLEL_CHECKPOINT}" --output "${output}" \
        --global-batch "${batch}" --standalone-rate --warmup-steps 10 --timed-steps 30 \
        > "${output}/stdout.log" 2> "${output}/stderr.log"; then
        printf 'COMPLETE devices=%s global_batch=%s\n' "${devices}" "${batch}"
    else
        result=$?
        printf '%s\n' "${result}" > "${output}/exit_code.txt"
        printf 'FAILED devices=%s global_batch=%s exit=%s (inspect stderr)\n' "${devices}" "${batch}" "${result}"
    fi
done
