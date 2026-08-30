#!/usr/bin/env bash
set -euo pipefail

# Upload the retained multi-task FM checkpoints and reproducibility metadata.
# ModelScope's upload cache allows an interrupted large-file upload to resume.

REPO_ID="${MODELSCOPE_REPO_ID:-ggkiller/multi-fm}"
REVISION="${MODELSCOPE_REVISION:-master}"
RUN_ROOT="${MULTIFM_RUN_ROOT:-/data/wangzihao/outputs/manifeel/table1_fm_b416_w12_e700_retrain_seed42}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$(command -v modelscope || true)}"

if [[ -z "${MODELSCOPE_BIN}" || ! -x "${MODELSCOPE_BIN}" ]]; then
    echo "ModelScope CLI not found; install it with: pip install modelscope" >&2
    exit 2
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
    exit 2
fi

NO_PROXY_ENV=(
    env
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY
    -u ALL_PROXY -u all_proxy
)

echo "[modelscope] user:"
"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" whoami
echo "[modelscope] repository: ${REPO_ID} revision=${REVISION}"
if ! "${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" info --repo-type model "${REPO_ID}" >/dev/null 2>&1; then
    cat >&2 <<EOF
ModelScope model repository is not accessible: ${REPO_ID}
Create it first at https://www.modelscope.cn/models/${REPO_ID}, then rerun.
EOF
    exit 2
fi

upload_file() {
    local source_path="$1"
    local destination_path="$2"
    if [[ ! -f "${source_path}" ]]; then
        echo "[modelscope] missing, skip: ${source_path}" >&2
        return 0
    fi
    echo "[modelscope] uploading ${source_path} -> ${destination_path}"
    "${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" upload \
        --repo-type model \
        --revision "${REVISION}" \
        --commit-message "Upload multi-task FM checkpoints" \
        --use-cache \
        "${REPO_ID}" "${source_path}" "${destination_path}"
}

for epoch in 100 200 300 400 500 600; do
    upload_file \
        "${CHECKPOINT_DIR}/latest_epoch${epoch}.ckpt" \
        "checkpoints/latest_epoch${epoch}.ckpt"
done

upload_file "${RUN_ROOT}/.hydra/config.yaml" "run/.hydra/config.yaml"
upload_file "${RUN_ROOT}/.hydra/overrides.yaml" "run/.hydra/overrides.yaml"
upload_file "${RUN_ROOT}/logs.json.txt" "run/logs.json.txt"
upload_file "${RUN_ROOT}/loss_curves.png" "run/loss_curves.png"

echo "[modelscope] upload complete: ${REPO_ID}"
