#!/usr/bin/env bash
# One job eligible for both H200 partitions; running snapshots are unchanged.
set -euo pipefail
: "${S2A_SOURCE:?Set frozen source directory}"
: "${S2A_OUTPUT:?Set isolated output directory}"
: "${S2A_STRATEGY:?}"
[[ "${S2A_STRATEGY}" =~ ^s[1234]$ ]]
arguments=()
if [[ -n "${S2A_DEPENDENCY:-}" ]]; then arguments+=(--dependency "${S2A_DEPENDENCY}"); fi
exec python3 "$(dirname -- "${BASH_SOURCE[0]}")/submit_partition_job.py" \
    --prefix S2A --source "${S2A_SOURCE}" --output "${S2A_OUTPUT}" \
    --script "slurm/manifeel/train_stage2_encoder_ablation.sbatch" --name "uvip_${S2A_STRATEGY}_t2_batch32_ep600" \
    --set "STRATEGY=${S2A_STRATEGY}" "${arguments[@]}"
