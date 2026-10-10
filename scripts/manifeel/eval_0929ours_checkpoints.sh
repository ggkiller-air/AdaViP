#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEEL_UNSET_PROXY=1
source "${SCRIPT_DIR}/common.sh"
activate_manifeel

CHECKPOINT_DIR="${MANIFEEL_REPO_ROOT}/checkpoints/0929ours"
OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/eval/0929ours_batch8_n10_steps500_seed100000_20260929"
mkdir -p "${OUTPUT_DIR}/logs"

run_gpu() {
    local phase="$1"
    local gpu="$2"
    shift 2
    local task epoch checkpoint result_dir log_path max_steps

    for task in "$@"; do
        for epoch in 100 200 300; do
            if [[ "${phase}" == "preflight" && "${epoch}" != 100 ]]; then
                continue
            fi
            checkpoint="${CHECKPOINT_DIR}/${task}/latest_epoch${epoch}.ckpt"
            result_dir="${OUTPUT_DIR}/${phase}/${task}/epoch${epoch}"
            log_path="${OUTPUT_DIR}/logs/${phase}_${task}_epoch${epoch}_gpu${gpu}.log"
            max_steps=500
            if [[ "${phase}" == "preflight" ]]; then
                max_steps=8
            fi
            if [[ -s "${result_dir}/eval_log.json" ]]; then
                echo "[skip] ${phase} ${task} epoch${epoch} (already complete)"
                continue
            fi
            if [[ ! -s "${checkpoint}" ]]; then
                echo "Missing checkpoint: ${checkpoint}" >&2
                return 1
            fi
            if [[ -d "${result_dir}" ]]; then
                mv -- "${result_dir}" "${result_dir}.failed.$(date +%s)"
            fi
            echo "[start] ${phase} ${task} epoch${epoch} gpu${gpu}"
            if ! "${MANIFEEL_PYTHON}" "${SCRIPT_DIR}/eval_single_task_checkpoint.py" \
                --checkpoint "${checkpoint}" \
                --output-dir "${result_dir}" \
                --device "cuda:${gpu}" \
                --n-test 10 --n-test-vis 0 \
                --max-steps "${max_steps}" \
                --test-start-seed 100000 > "${log_path}" 2>&1; then
                echo "[failed] ${phase} ${task} epoch${epoch}: ${log_path}" >&2
                return 1
            fi
            if [[ ! -s "${result_dir}/eval_log.json" ]]; then
                echo "Missing evaluation result: ${result_dir}/eval_log.json" >&2
                return 1
            fi
            echo "[done] ${phase} ${task} epoch${epoch} gpu${gpu}"
        done
    done
}

run_phase() {
    local phase="$1"
    local failed=0
    local pid
    local -a pids=()

    run_gpu "${phase}" 1 peg_insertion bulb_installation & pids+=("$!")
    sleep 45
    run_gpu "${phase}" 2 usb_insertion nut_bolt_assembly & pids+=("$!")
    sleep 45
    run_gpu "${phase}" 3 gear_assembly object_search & pids+=("$!")
    sleep 45
    run_gpu "${phase}" 4 peg_reorientation ball_sorting & pids+=("$!")
    sleep 45
    run_gpu "${phase}" 6 fusionhyper & pids+=("$!")
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "${phase} failed; inspect ${OUTPUT_DIR}/logs/" >&2
        return 1
    fi
}

case "${MANIFEEL_0929_EVAL_PHASE:-all}" in
    all)
        run_phase preflight
        run_phase formal
        ;;
    preflight|formal)
        run_phase "${MANIFEEL_0929_EVAL_PHASE}"
        ;;
    *)
        echo "MANIFEEL_0929_EVAL_PHASE must be all, preflight, or formal" >&2
        exit 2
        ;;
esac
