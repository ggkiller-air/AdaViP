#!/usr/bin/env bash
# One job eligible for both H200 partitions; running snapshots are unchanged.
set -euo pipefail
: "${LC_SOURCE:?Set frozen source directory}"
: "${LC_OUTPUT:?Set isolated output directory}"
: "${LC_VARIANT:?}"
[[ "${LC_VARIANT}" == l1 || "${LC_VARIANT}" == l2 ]]
arguments=()
if [[ -n "${LC_DEPENDENCY:-}" ]]; then arguments+=(--dependency "${LC_DEPENDENCY}"); fi
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/submit_partition_job.py" \
    --prefix LC --source "${LC_SOURCE}" --output "${LC_OUTPUT}" \
    --script "slurm/manifeel/train_language_conditioning.sbatch" --name "lc_${LC_VARIANT}_t2_batch64_ep500" \
    --set "VARIANT=${LC_VARIANT}" "${arguments[@]}"
