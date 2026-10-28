#!/usr/bin/env bash
# Race the current maximum walltimes, with one lock and sibling-only cancellation.
set -euo pipefail
: "${S2A_SOURCE:?}"
: "${S2A_OUTPUT:?}"
: "${S2A_STRATEGY:?}"
[[ "${S2A_STRATEGY}" =~ ^s[1234]$ ]]
mkdir -p "${S2A_OUTPUT}/queue" /data/wangzihao/outputs/manifeel/slurm
wave="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
submitted=()
cleanup_submission() {
    for job_id in "${submitted[@]}"; do scancel "${job_id}" 2>/dev/null || true; done
}
trap cleanup_submission ERR
for partition in gpu gpux; do
    max_time="$(scontrol show partition "${partition}" -o | sed -n 's/.* MaxTime=\([^ ]*\).*/\1/p')"
    [[ -n "${max_time}" && "${max_time}" != UNLIMITED ]]
    options=(--parsable --partition="${partition}" --time="${max_time}" --job-name="uvip_${S2A_STRATEGY}_t2_batch32_ep600")
    if [[ -n "${S2A_DEPENDENCY:-}" ]]; then options+=(--dependency="${S2A_DEPENDENCY}"); fi
    job_id="$(sbatch "${options[@]}" \
        --export="ALL,S2A_SOURCE=${S2A_SOURCE},S2A_OUTPUT=${S2A_OUTPUT},S2A_STRATEGY=${S2A_STRATEGY},S2A_WAVE=${wave}" \
        "${S2A_SOURCE}/slurm/manifeel/train_stage2_encoder_ablation.sbatch")"
    submitted+=("${job_id%%;*}")
    echo "${S2A_STRATEGY} ${partition} ${max_time} ${job_id}"
done
printf '%s\n' "${submitted[@]}" > "${S2A_OUTPUT}/queue/${wave}.partial"
mv "${S2A_OUTPUT}/queue/${wave}.partial" "${S2A_OUTPUT}/queue/${wave}.jobs"
trap - ERR
