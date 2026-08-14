#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
activate_manifeel

CONFIG_NAME="${1:?usage: preflight_table1_config.sh CONFIG BATCH WORKERS LABEL [STEPS]}"
BATCH_SIZE="${2:?batch size is required}"
NUM_WORKERS="${3:?num workers is required}"
LABEL="${4:?label is required}"
TRAIN_STEPS="${5:-10}"

if (( NUM_WORKERS < 1 )); then
    echo "NUM_WORKERS must be at least 1." >&2
    exit 2
fi

SAFE_LABEL="${LABEL:0:24}"
RUN_NAME="pf_t1_${SAFE_LABEL}_b${BATCH_SIZE}_w${NUM_WORKERS}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/${RUN_NAME}"
MONITOR_PATH="${OUTPUT_DIR}/gpu_monitor.csv"
STEP_MONITOR_PATH="${OUTPUT_DIR}/step_monitor.csv"
LOG_PATH="${OUTPUT_DIR}/preflight.log"
SUMMARY_PATH="${OUTPUT_DIR}/preflight_summary.txt"
mkdir -p "${OUTPUT_DIR}"

monitor_pid=""
step_monitor_pid=""
cleanup() {
    if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    if [[ -n "${step_monitor_pid}" ]] && kill -0 "${step_monitor_pid}" 2>/dev/null; then
        kill "${step_monitor_pid}" 2>/dev/null || true
        wait "${step_monitor_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
    --format=csv,noheader,nounits -lms 500 >"${MONITOR_PATH}" 2>&1 &
monitor_pid=$!
(
    while true; do
        line_count=0
        if [[ -f "${OUTPUT_DIR}/logs.json.txt" ]]; then
            line_count="$(wc -l <"${OUTPUT_DIR}/logs.json.txt")"
        fi
        printf '%s,%s\n' "$(date +%s%3N)" "${line_count}"
        sleep 0.5
    done
) >"${STEP_MONITOR_PATH}" 2>&1 &
step_monitor_pid=$!

start_seconds="$(date +%s)"
set +e
/usr/bin/time -v env \
    MANIFEEL_RUN_NAME="${RUN_NAME}" \
    MANIFEEL_CONFIG_NAME="${CONFIG_NAME}" \
    MANIFEEL_NUM_EPOCHS=1 MANIFEEL_ROLLOUT_EVERY=0 \
    MANIFEEL_CHECKPOINT_EVERY=100000 MANIFEEL_VAL_EVERY=100000 \
    MANIFEEL_SAMPLE_EVERY=100000 MANIFEEL_BATCH_SIZE="${BATCH_SIZE}" \
    MANIFEEL_NUM_WORKERS="${NUM_WORKERS}" MANIFEEL_PREFETCH_FACTOR=1 \
    MANIFEEL_VAL_BATCH_SIZE=64 MANIFEEL_VAL_NUM_WORKERS=1 \
    MANIFEEL_VAL_PREFETCH_FACTOR=1 MANIFEEL_MAX_TRAIN_STEPS="${TRAIN_STEPS}" \
    MANIFEEL_MAX_VAL_STEPS=1 MANIFEEL_WANDB_MODE=offline \
    bash "${SCRIPT_DIR}/train_multitask_dp.sh" policy.num_inference_steps=1 \
        checkpoint.save_last_ckpt=false >"${LOG_PATH}" 2>&1
status=$?
set -e
wall_seconds="$(( $(date +%s) - start_seconds ))"
cleanup
monitor_pid=""
step_monitor_pid=""

awk -F, -v status="${status}" -v wall="${wall_seconds}" '
    {
        memory = $2 + 0
        total_memory = $3 + 0
        sm = $4 + 0
        power = $6 + 0
        sample_count += 1
        memory_samples[sample_count] = memory
        sm_samples[sample_count] = sm
        power_samples[sample_count] = power
        if (memory > peak_memory) peak_memory = memory
        if (sm > peak_sm) peak_sm = sm
        if (sm > 0) {
            busy_sm_sum += sm
            busy_samples += 1
        }
    }
    END {
        threshold = peak_memory * 0.75
        for (sample_index = 1; sample_index <= sample_count; sample_index += 1) {
            if (memory_samples[sample_index] >= threshold) {
                active_sm_sum += sm_samples[sample_index]
                active_power_sum += power_samples[sample_index]
                active_samples += 1
                if (sm_samples[sample_index] > 0) active_busy_samples += 1
            }
        }
        mean_sm = active_samples ? active_sm_sum / active_samples : 0
        printf "exit_code=%d\nwall_seconds=%d\npeak_memory_mib=%d\n", status, wall, peak_memory
        printf "peak_memory_percent=%.2f\n", total_memory ? 100 * peak_memory / total_memory : 0
        printf "active_gpu_samples=%d\nmean_sm_percent_active=%.2f\n", active_samples, mean_sm
        printf "gpu_busy_percent_active=%.2f\n", \
            active_samples ? 100 * active_busy_samples / active_samples : 0
        printf "mean_power_watts_active=%.2f\n", \
            active_samples ? active_power_sum / active_samples : 0
        printf "busy_gpu_samples=%d\nmean_sm_percent_busy=%.2f\npeak_sm_percent=%d\n", \
            busy_samples, busy_samples ? busy_sm_sum / busy_samples : 0, peak_sm
    }
' "${MONITOR_PATH}" | tee "${SUMMARY_PATH}"
if [[ -s "${STEP_MONITOR_PATH}" ]]; then
    awk -F, -v batch_size="${BATCH_SIZE}" -v expected_steps="${TRAIN_STEPS}" \
        -v workers="${NUM_WORKERS}" '
        {
            timestamp = $1 + 0
            line_count = $2 + 0
            if (line_count > expected_steps) line_count = expected_steps
            if (line_count > last_count) {
                if (first_count == 0) {
                    first_count = line_count
                    first_timestamp = timestamp
                }
                if (post_prefetch_count == 0 && line_count >= workers + 1) {
                    post_prefetch_count = line_count
                    post_prefetch_timestamp = timestamp
                }
                last_count = line_count
                last_timestamp = timestamp
            }
        }
        END {
            measured_intervals = last_count - first_count
            measured_seconds = (last_timestamp - first_timestamp) / 1000
            samples_per_second = 0
            if (measured_seconds > 0) {
                samples_per_second = batch_size * measured_intervals / measured_seconds
            }
            printf "observed_train_steps=%d\n", last_count
            printf "steady_step_intervals=%d\n", measured_intervals
            printf "steady_train_seconds=%.3f\n", measured_seconds
            printf "steady_samples_per_second=%.2f\n", samples_per_second

            post_prefetch_intervals = last_count - post_prefetch_count
            post_prefetch_seconds = (last_timestamp - post_prefetch_timestamp) / 1000
            post_prefetch_rate = 0
            if (post_prefetch_seconds > 0) {
                post_prefetch_rate = batch_size * post_prefetch_intervals / post_prefetch_seconds
            }
            printf "post_prefetch_step_intervals=%d\n", post_prefetch_intervals
            printf "post_prefetch_train_seconds=%.3f\n", post_prefetch_seconds
            printf "post_prefetch_samples_per_second=%.2f\n", post_prefetch_rate
        }
    ' "${STEP_MONITOR_PATH}" | tee -a "${SUMMARY_PATH}"
fi
awk -F: '/Maximum resident set size/ {
    value = $2
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    printf "host_max_rss_kib=%s\n", value
}' "${LOG_PATH}" | tee -a "${SUMMARY_PATH}"
echo "preflight_output=${OUTPUT_DIR}" | tee -a "${SUMMARY_PATH}"
echo "preflight_log=${LOG_PATH}" | tee -a "${SUMMARY_PATH}"
exit "${status}"
