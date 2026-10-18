#!/usr/bin/env bash
#SBATCH --job-name=full_sensor_fix_next_duration
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail
: "${ADAVIP_RUN_DIR:?Set the existing training run}"
: "${ADAVIP_PARENT_JOB_ID:?Set the currently running GPU job}"
parent_state="$(sacct -n -X -j "${ADAVIP_PARENT_JOB_ID}" --format=State | awk 'NF {print $1; exit}')"
if [[ "${parent_state}" != COMPLETED ]]; then
    echo "Parent did not complete safely: ${parent_state}; no jobs changed" >&2
    exit 1
fi
successor="$(awk '/^Submitted batch job [0-9]+$/ {job=$4} END {print job}' "${ADAVIP_RUN_DIR}/continuation_jobs.log")"
if [[ ! "${successor}" =~ ^[0-9]+$ || "${successor}" == "${ADAVIP_PARENT_JOB_ID}" ]]; then
    echo 'No new automatic successor found; no jobs changed' >&2
    exit 1
fi
details="$(scontrol show job "${successor}" -o)"
expected="Command=${ADAVIP_RUN_DIR}/source/slurm/manifeel/train_full_sensor_decoded_batch64.sbatch"
if [[ " ${details} " != *" ${expected} "* || " ${details} " != *" JobState=PENDING "* ]]; then
    echo "Successor is not pending or belongs to another run; no jobs changed: ${details}" >&2
    exit 1
fi
scontrol hold "${successor}"
held_successor="${successor}"
trap 'scontrol release "${held_successor}" || true' EXIT
scontrol update "JobId=${successor}" TimeLimit=2-00:00:00 TimeMin=2-00:00:00
details="$(scontrol show job "${successor}" -o)"
if [[ " ${details} " != *" TimeLimit=2-00:00:00 "* || " ${details} " != *" TimeMin=2-00:00:00 "* ]]; then
    echo "Failed to enforce full 48-hour allocation: ${details}" >&2
    exit 1
fi
scontrol release "${successor}"
trap - EXIT
printf '%s\n' "${details}"
printf '%s parent=%s successor=%s requested=48h minimum=48h\n' \
    "$(date --iso-8601=seconds)" "${ADAVIP_PARENT_JOB_ID}" "${successor}" \
    >> "${ADAVIP_RUN_DIR}/duration_override.log"
