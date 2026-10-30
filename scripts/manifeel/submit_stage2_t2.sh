#!/usr/bin/env bash
# One job eligible for both H200 partitions; running snapshots are unchanged.
set -euo pipefail
: "${STAGE2_SOURCE:?Set frozen source directory}"
: "${STAGE2_OUTPUT:?Set isolated output directory}"
arguments=()
if [[ -n "${STAGE2_DEPENDENCY:-}" ]]; then arguments+=(--dependency "${STAGE2_DEPENDENCY}"); fi
if [[ "${STAGE2_MODE:-combined}" == full && -f "${STAGE2_OUTPUT}/checkpoints/latest.ckpt" ]]; then
    arguments+=(--set "RESUME=${STAGE2_OUTPUT}/checkpoints/latest.ckpt")
fi
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/submit_partition_job.py" \
    --prefix STAGE2 --source "${STAGE2_SOURCE}" --output "${STAGE2_OUTPUT}" \
    --script "slurm/manifeel/train_stage2_t2_h200.sbatch" --name "univtac_stage2_transformer_t2" \
    --set "MODE=${STAGE2_MODE:-combined}" "${arguments[@]}"
