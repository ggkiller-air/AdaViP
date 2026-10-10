#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEEL_UNSET_PROXY=1
source "${SCRIPT_DIR}/common.sh"
activate_manifeel

OUTPUT_DIR="${MANIFEEL_OUTPUT_ROOT}/eval/original_batch8_best_epochs_n50_steps500_seed100000_20260929"
mkdir -p "${OUTPUT_DIR}/logs"

run_gpu() {
    local phase="$1"
    local gpu="$2"
    shift 2
    local spec task relative_checkpoint epoch result_dir log_path max_steps exit_code

    for spec in "$@"; do
        IFS='|' read -r task relative_checkpoint epoch <<< "${spec}"
        result_dir="${OUTPUT_DIR}/${phase}/${task}_epoch${epoch}"
        log_path="${OUTPUT_DIR}/logs/${phase}_${task}_epoch${epoch}_gpu${gpu}.log"
        max_steps=500
        if [[ "${phase}" == "preflight" ]]; then
            max_steps=8
        fi
        if [[ -s "${result_dir}/eval_log.json" ]]; then
            echo "[skip] ${phase} ${task} epoch${epoch}"
            continue
        fi
        if [[ ! -s "${MANIFEEL_REPO_ROOT}/checkpoints/${relative_checkpoint}" ]]; then
            echo "Missing checkpoint: ${relative_checkpoint}" >&2
            return 1
        fi
        if [[ -d "${result_dir}" ]]; then
            mv -- "${result_dir}" "${result_dir}.failed.$(date +%s)"
        fi
        echo "[start] ${phase} ${task} epoch${epoch} gpu${gpu}"
        exit_code=0
        "${MANIFEEL_PYTHON}" "${SCRIPT_DIR}/eval_upstream_checkpoint.py" \
            --checkpoint "${MANIFEEL_REPO_ROOT}/checkpoints/${relative_checkpoint}" \
            --output-dir "${result_dir}" \
            --device "cuda:${gpu}" \
            --n-test 50 --n-test-vis 2 \
            --max-steps "${max_steps}" --test-start-seed 100000 \
            > "${log_path}" 2>&1 || exit_code=$?
        if [[ ! -s "${result_dir}/eval_log.json" ]]; then
            echo "[failed] ${phase} ${task} epoch${epoch} (exit ${exit_code}): ${log_path}" >&2
            return 1
        fi
        echo "[done] ${phase} ${task} epoch${epoch} gpu${gpu} (exit ${exit_code})"
    done
}

run_phase() {
    local phase="$1"
    local failed=0
    local pid
    local -a pids=()

    run_gpu "${phase}" 1 \
        'power_plug_fusion_only|0926/dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42/checkpoints/latest_epoch100.ckpt|100' \
        'bulb_installation|0929ours/bulb_installation/latest_epoch200.ckpt|200' & pids+=("$!")
    sleep 45
    run_gpu "${phase}" 2 \
        'fusionhyper|0929ours/fusionhyper/latest_epoch300.ckpt|300' \
        'nut_bolt_assembly|0929ours/nut_bolt_assembly/latest_epoch200.ckpt|200' & pids+=("$!")
    sleep 45
    run_gpu "${phase}" 3 \
        'peg_insertion|0929ours/peg_insertion/latest_epoch100.ckpt|100' \
        'usb_insertion|0929ours/usb_insertion/latest_epoch300.ckpt|300' & pids+=("$!")
    sleep 45
    run_gpu "${phase}" 4 \
        'gear_assembly|0929ours/gear_assembly/latest_epoch100.ckpt|100' \
        'object_search|0929ours/object_search/latest_epoch300.ckpt|300' & pids+=("$!")
    sleep 45
    run_gpu "${phase}" 6 \
        'peg_reorientation|0929ours/peg_reorientation/latest_epoch300.ckpt|300' \
        'ball_sorting|0929ours/ball_sorting/latest_epoch200.ckpt|200' & pids+=("$!")
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "${phase} failed; inspect ${OUTPUT_DIR}/logs/" >&2
        return 1
    fi
}

case "${MANIFEEL_BEST_EVAL_PHASE:-all}" in
    all)
        run_phase preflight
        run_phase formal
        ;;
    preflight|formal)
        run_phase "${MANIFEEL_BEST_EVAL_PHASE}"
        ;;
    *)
        echo "MANIFEEL_BEST_EVAL_PHASE must be all, preflight, or formal" >&2
        exit 2
        ;;
esac
