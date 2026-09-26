#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy

REPO_ID="${MODELSCOPE_REPO_ID:-ggkiller/multi-fm}"
REVISION="${MODELSCOPE_REVISION:-master}"
REMOTE_ROOT="${MODELSCOPE_REMOTE_ROOT:-0926}"
OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$(command -v modelscope || true)}"
RUNS=(
    dp_power_plug_cs_fusion_batch8_ep400_seed42
    dp_ball_sorting_cs_fusion_batch8_ep400_seed42
    dp_power_plug_hyper_resnet_no_fusion_batch8_ep400_seed42
    dp_power_plug_hyper_resnet_fusion_batch8_ep400_seed42
    dp_power_plug_resnet_fusion_batch8_ep400_seed42
)
EPOCHS=(100 200 300)
MIN_CHECKPOINT_BYTES=4000000000

if [[ -z "${MODELSCOPE_BIN}" || ! -x "${MODELSCOPE_BIN}" ]]; then
    echo "ModelScope CLI not found" >&2
    exit 2
fi

for run in "${RUNS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        checkpoint="${OUTPUT_ROOT}/${run}/checkpoints/latest_epoch${epoch}.ckpt"
        if [[ ! -f "${checkpoint}" ]]; then
            echo "Missing checkpoint: ${checkpoint}" >&2
            exit 1
        fi
        size_bytes="$(stat -c '%s' "${checkpoint}")"
        if (( size_bytes < MIN_CHECKPOINT_BYTES )); then
            echo "Incomplete checkpoint (${size_bytes} bytes): ${checkpoint}" >&2
            exit 1
        fi
        if ! zipinfo -1 "${checkpoint}" archive/data.pkl >/dev/null 2>&1; then
            echo "Invalid workspace archive: ${checkpoint}" >&2
            exit 1
        fi
    done
done

"${MODELSCOPE_BIN}" whoami
"${MODELSCOPE_BIN}" info --repo-type model "${REPO_ID}" >/dev/null

for run in "${RUNS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        checkpoint="${OUTPUT_ROOT}/${run}/checkpoints/latest_epoch${epoch}.ckpt"
        destination="${REMOTE_ROOT}/${run}/checkpoints/latest_epoch${epoch}.ckpt"
        echo "[modelscope] uploading ${checkpoint} -> ${destination}"
        "${MODELSCOPE_BIN}" upload \
            --repo-type model \
            --revision "${REVISION}" \
            --commit-message "Upload ${run} epoch ${epoch}" \
            --use-cache \
            --disable-tqdm \
            "${REPO_ID}" "${checkpoint}" "${destination}"
    done
done

echo "[modelscope] all 15 checkpoints uploaded"
