#!/usr/bin/env bash
#SBATCH --job-name=manifeel-table1-preflight
#SBATCH --gres=gpu:NVIDIAH200:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/data/wangzihao/outputs/manifeel/slurm/%x_%j.out
#SBATCH --error=/data/wangzihao/outputs/manifeel/slurm/%x_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/public/home/wangzihao/projects/AdaViP}"
STEPS="${MANIFEEL_PREFLIGHT_STEPS:-6}"
MATRIX_DIR="/data/wangzihao/outputs/manifeel/preflight_table1_matrix_${SLURM_JOB_ID}"
MATRIX_SUMMARY="${MATRIX_DIR}/matrix_summary.txt"
mkdir -p "${MATRIX_DIR}" /data/wangzihao/outputs/manifeel/slurm
cd "${REPO_ROOT}"

run_case() {
    local config="$1"
    local batch_size="$2"
    local workers="$3"
    local label="$4"
    local case_log="${MATRIX_DIR}/${label}.log"
    local status

    printf '\n=== case=%s config=%s batch=%s workers=%s steps=%s ===\n' \
        "${label}" "${config}" "${batch_size}" "${workers}" "${STEPS}" | \
        tee -a "${MATRIX_SUMMARY}"

    set +e
    bash scripts/manifeel/preflight_table1_config.sh \
        "${config}" "${batch_size}" "${workers}" "${label}" "${STEPS}" \
        2>&1 | tee "${case_log}"
    status="${PIPESTATUS[0]}"
    set -e

    awk -v label="${label}" -v status="${status}" '
        /^exit_code=|^wall_seconds=|^peak_memory_|^mean_sm_percent_active=/ ||
        /^gpu_busy_percent_active=|^mean_power_watts_active=/ ||
        /^steady_samples_per_second=|^host_max_rss_kib=|^preflight_output=/ {
            printf "%s.%s\n", label, $0
        }
        END {
            printf "%s.case_exit_code=%d\n", label, status
        }
    ' "${case_log}" | tee -a "${MATRIX_SUMMARY}"
}

run_case train_multitask_diffusion_workspace.yaml 384 4 dp_b384_w4
run_case train_multitask_diffusion_workspace.yaml 384 8 dp_b384_w8
run_case train_multitask_diffusion_workspace.yaml 384 12 dp_b384_w12
run_case train_multitask_diffusion_workspace.yaml 512 8 dp_b512_w8

for spec in \
    "train_multitask_vistac_diffusion_workspace.yaml vistac_dp" \
    "train_multitask_flow_matching_workspace.yaml fm" \
    "train_multitask_adavip_diffusion_workspace.yaml adavip_dp" \
    "train_multitask_adavip_flow_matching_workspace.yaml adavip_fm"
do
    read -r config label <<<"${spec}"
    run_case "${config}" 192 4 "${label}_b192_w4"
    run_case "${config}" 192 8 "${label}_b192_w8"
    run_case "${config}" 256 8 "${label}_b256_w8"
done

echo "matrix_summary=${MATRIX_SUMMARY}"
