#!/usr/bin/env bash
set -euo pipefail

# Upload complete training workspace checkpoints for native ManiFeel evaluation.
# Deployment artifacts are intentionally rejected because ManiFeel eval.py needs
# the saved Hydra cfg, workspace state_dicts, and pickled training state.

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPO_ID="${MODELSCOPE_REPO_ID:-ggkiller/multi-fm}"
REVISION="${MODELSCOPE_REVISION:-master}"
RUN_ROOT="${ADAVIP_FM_RUN_ROOT:-/data/wangzihao/outputs/manifeel/table1_hyper_adavip_fm_ep500_init_b416_w12_seed42}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
ARCHIVE_DIR="${ADAVIP_FM_ARCHIVE_ROOT:-/data/wangzihao/outputs/manifeel/checkpoint_archive/table1_hyper_adavip_fm_ep500_init_b416_w12_seed42}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$(command -v modelscope || true)}"
EPOCHS="${ADAVIP_FM_EPOCHS:-120 150 210}"
MIN_FULL_CHECKPOINT_BYTES="${ADAVIP_FM_MIN_FULL_CHECKPOINT_BYTES:-6000000000}"
ENCODER_SOURCE="${REPO_ROOT}/adavip/manifeel/hyper_adavip_obs_encoder.py"

if [[ -z "${MODELSCOPE_BIN}" || ! -x "${MODELSCOPE_BIN}" ]]; then
    echo "ModelScope CLI not found; install it and run modelscope login first." >&2
    exit 2
fi
if [[ ! -f "${ENCODER_SOURCE}" ]]; then
    echo "Hyper AdaViP encoder source not found: ${ENCODER_SOURCE}" >&2
    exit 2
fi

NO_PROXY_ENV=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy -u NO_PROXY -u no_proxy)

find_checkpoint() {
    local epoch="$1"
    local path="${CHECKPOINT_DIR}/latest_epoch${epoch}.ckpt"
    if [[ ! -f "${path}" ]]; then
        path="${ARCHIVE_DIR}/latest_epoch${epoch}.ckpt"
    fi
    if [[ ! -f "${path}" ]]; then
        echo "Missing FM checkpoint in run and archive: latest_epoch${epoch}.ckpt" >&2
        exit 1
    fi
    local size_bytes
    size_bytes="$(stat -c '%s' "${path}")"
    if (( size_bytes < MIN_FULL_CHECKPOINT_BYTES )); then
        echo "Refusing non-workspace checkpoint (${size_bytes} bytes): ${path}" >&2
        exit 1
    fi
    if ! zipinfo -1 "${path}" archive/data.pkl >/dev/null 2>&1; then
        echo "Checkpoint is not a complete PyTorch workspace archive: ${path}" >&2
        exit 1
    fi
    printf '%s' "${path}"
}

upload_file() {
    local source_path="$1"
    local destination_path="$2"
    echo "[modelscope] uploading ${source_path} -> ${destination_path}"
    "${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" upload \
        --repo-type model \
        --revision "${REVISION}" \
        --commit-message "Upload full AdaViP FM workspace checkpoints" \
        --use-cache \
        --disable-tqdm \
        "${REPO_ID}" "${source_path}" "${destination_path}"
}

"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" whoami
"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" info --repo-type model "${REPO_ID}" >/dev/null

for epoch in ${EPOCHS}; do
    source_path="$(find_checkpoint "${epoch}")"
    upload_file "${source_path}" "adavip-fm/checkpoints/latest_epoch${epoch}.ckpt"
done

upload_file "${RUN_ROOT}/.hydra/config.yaml" "adavip-fm/run/.hydra/config.yaml"
upload_file "${RUN_ROOT}/.hydra/overrides.yaml" "adavip-fm/run/.hydra/overrides.yaml"
upload_file "${ENCODER_SOURCE}" "adavip-fm/source/adavip/manifeel/hyper_adavip_obs_encoder.py"
echo "[modelscope] Full AdaViP FM workspace upload complete: ${REPO_ID}"
