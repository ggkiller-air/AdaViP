#!/usr/bin/env bash
set -euo pipefail

# Upload complete workspace checkpoints for the three Power Plug baselines.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy

REPO_ID="${MODELSCOPE_REPO_ID:-ggkiller/multi-fm}"
REVISION="${MODELSCOPE_REVISION:-master}"
REMOTE_ROOT="${MODELSCOPE_REMOTE_ROOT:-0923_BaselineCkeck}"
OUTPUT_ROOT="${MANIFEEL_OUTPUT_ROOT:-/data/wangzihao/outputs/manifeel}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$(command -v modelscope || true)}"
EPOCHS=(100 200 300 400 600 800 1000)
MIN_CHECKPOINT_BYTES=4000000000

declare -A RUNS=(
    [vision]="dp_power_plug_vision_seed42"
    [tacrgb]="dp_power_plug_tacrgb_seed42"
    [tacff]="dp_power_plug_tacff_seed42"
)
BASELINES=(vision tacrgb tacff)

if [[ -z "${MODELSCOPE_BIN}" || ! -x "${MODELSCOPE_BIN}" ]]; then
    echo "ModelScope CLI not found" >&2
    exit 2
fi

NO_PROXY_ENV=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy -u NO_PROXY -u no_proxy)

for baseline in "${BASELINES[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        checkpoint="${OUTPUT_ROOT}/${RUNS[${baseline}]}/checkpoints/latest_epoch${epoch}.ckpt"
        if [[ ! -f "${checkpoint}" ]]; then
            echo "Missing checkpoint: ${checkpoint}" >&2
            exit 1
        fi
        size_bytes="$(stat -c '%s' "${checkpoint}")"
        if (( size_bytes < MIN_CHECKPOINT_BYTES )); then
            echo "Refusing incomplete checkpoint (${size_bytes} bytes): ${checkpoint}" >&2
            exit 1
        fi
        if ! zipinfo -1 "${checkpoint}" archive/data.pkl >/dev/null 2>&1; then
            echo "Invalid PyTorch workspace archive: ${checkpoint}" >&2
            exit 1
        fi
    done
done

"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" whoami
"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" info --repo-type model "${REPO_ID}" >/dev/null

for baseline in "${BASELINES[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
        checkpoint="${OUTPUT_ROOT}/${RUNS[${baseline}]}/checkpoints/latest_epoch${epoch}.ckpt"
        destination="${REMOTE_ROOT}/${baseline}/latest_epoch${epoch}.ckpt"
        echo "[modelscope] uploading ${checkpoint} -> ${destination}"
        "${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" upload \
            --repo-type model \
            --revision "${REVISION}" \
            --commit-message "Upload ${baseline} Power Plug baseline epoch ${epoch}" \
            --use-cache \
            --disable-tqdm \
            "${REPO_ID}" "${checkpoint}" "${destination}"
    done
done

echo "[modelscope] all Power Plug baseline checkpoints uploaded"
