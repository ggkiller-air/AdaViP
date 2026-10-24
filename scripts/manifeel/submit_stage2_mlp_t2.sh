#!/usr/bin/env bash
# Queue both H200 partitions at their maximum walltime for one isolated run.
set -euo pipefail

: "${STAGE2_MLP_SOURCE:?Set immutable MLP source directory}"
: "${STAGE2_MLP_OUTPUT:?Set isolated MLP output directory}"
STAGE2_MLP_MODE="${STAGE2_MLP_MODE:-combined}"
mkdir -p "${STAGE2_MLP_OUTPUT}/queue" /data/wangzihao/outputs/manifeel/slurm
wave="$(date -u +%Y%m%dT%H%M%S)_${RANDOM}"
script="${STAGE2_MLP_SOURCE}/slurm/manifeel/train_stage2_mlp_t2_h200.sbatch"
job_ids=()
queue_file="${STAGE2_MLP_OUTPUT}/queue/${wave}.jobs"
cleanup_failed_submission() {
    for job_id in "${job_ids[@]}"; do
        scancel "${job_id}" 2>/dev/null || true
    done
}
trap cleanup_failed_submission ERR

for partition in gpu gpux; do
    max_time="$(scontrol show partition "${partition}" -o | sed -n 's/.* MaxTime=\([^ ]*\).*/\1/p')"
    if [[ -z "${max_time}" || "${max_time}" == UNLIMITED ]]; then
        echo "Cannot determine finite maximum walltime for ${partition}" >&2
        exit 2
    fi
    options=(--parsable --partition="${partition}" --time="${max_time}")
    if [[ -n "${STAGE2_MLP_DEPENDENCY:-}" ]]; then
        options+=(--dependency="${STAGE2_MLP_DEPENDENCY}")
    fi
    export_spec="ALL,STAGE2_MLP_SOURCE=${STAGE2_MLP_SOURCE},STAGE2_MLP_OUTPUT=${STAGE2_MLP_OUTPUT},STAGE2_MLP_MODE=${STAGE2_MLP_MODE},STAGE2_MLP_WAVE=${wave}"
    if [[ "${STAGE2_MLP_MODE}" == full ]]; then
        export_spec+=",STAGE2_MLP_RESUME=${STAGE2_MLP_OUTPUT}/checkpoints/latest.ckpt"
    fi
    job_id="$(sbatch "${options[@]}" --export="${export_spec}" "${script}")"
    job_ids+=("${job_id%%;*}")
    echo "${partition} ${max_time} ${job_id}"
done
printf '%s\n' "${job_ids[@]}" > "${queue_file}.partial"
mv "${queue_file}.partial" "${queue_file}"
trap - ERR
