#!/usr/bin/env bash
# Competing partitions are mutually exclusive within one L1 or L2 run.
set -euo pipefail
: "${LC_SOURCE:?Set frozen source directory}"
: "${LC_OUTPUT:?Set isolated run directory}"
: "${LC_VARIANT:?Set l1 or l2}"
[[ "${LC_VARIANT}" == l1 || "${LC_VARIANT}" == l2 ]]
mkdir -p "${LC_OUTPUT}/queue" /data/wangzihao/outputs/manifeel/slurm
wave="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
submitted=()
cleanup_submission() {
    for job_id in "${submitted[@]}"; do scancel "${job_id}" 2>/dev/null || true; done
}
trap cleanup_submission ERR
for partition in gpu gpux; do
    max_time="$(scontrol show partition "${partition}" -o | sed -n 's/.* MaxTime=\([^ ]*\).*/\1/p')"
    [[ -n "${max_time}" && "${max_time}" != UNLIMITED ]]
    options=(--parsable --partition="${partition}" --time="${max_time}" --job-name="lc_${LC_VARIANT}_t2_batch64_ep500")
    if [[ -n "${LC_DEPENDENCY:-}" ]]; then options+=(--dependency="${LC_DEPENDENCY}"); fi
    job_id="$(sbatch "${options[@]}" \
        --export="ALL,LC_SOURCE=${LC_SOURCE},LC_OUTPUT=${LC_OUTPUT},LC_VARIANT=${LC_VARIANT},LC_WAVE=${wave}" \
        "${LC_SOURCE}/slurm/manifeel/train_language_conditioning.sbatch")"
    submitted+=("${job_id%%;*}")
    echo "${LC_VARIANT} ${partition} ${max_time} ${job_id}"
done
printf '%s\n' "${submitted[@]}" > "${LC_OUTPUT}/queue/${wave}.partial"
mv "${LC_OUTPUT}/queue/${wave}.partial" "${LC_OUTPUT}/queue/${wave}.jobs"
trap - ERR
