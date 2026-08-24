#!/usr/bin/env bash
# Upload and verify the Piper RDP deployment archive on Tsinghua Cloud.

set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

FILE="${1:-outputs/real_world/piper_rdp/deployment/piper_rdp_18_75hz_ema.tar.gz}"
: "${PIPER_RDP_SHARE_TOKEN:?Set PIPER_RDP_SHARE_TOKEN to the Seafile share token}"
SHARE_TOKEN="${PIPER_RDP_SHARE_TOKEN}"
TARGET_NAME="${PIPER_RDP_TARGET_NAME:-piper_rdp_18_75hz_ema_verified.tar.gz}"
BASE_URL="https://cloud.tsinghua.edu.cn"

if [[ ! -f "${FILE}" ]]; then
    echo "Archive does not exist: ${FILE}" >&2
    exit 1
fi

FILE_SIZE="$(stat -c %s "${FILE}")"
FILE_SHA256="$(sha256sum "${FILE}" | awk '{print $1}')"
DOWNLOAD_URL="${BASE_URL}/d/${SHARE_TOKEN}/files/?p=%2F${TARGET_NAME}&dl=1"

api_get() {
    curl --fail --silent --show-error --location \
        --retry 5 --retry-all-errors --connect-timeout 20 "$1"
}

listing="$(api_get "${BASE_URL}/api/v2.1/share-links/${SHARE_TOKEN}/dirents/?path=%2F")"
if LISTING="${listing}" python3 - "${TARGET_NAME}" "${FILE_SIZE}" <<'PY'
import json
import os
import sys

listing = json.loads(os.environ["LISTING"])["dirent_list"]
name, expected_size = sys.argv[1], int(sys.argv[2])
matched = [item for item in listing if item.get("file_name") == name]
raise SystemExit(0 if matched and int(matched[0]["size"]) == expected_size else 1)
PY
then
    echo "Cloud file with matching size already exists; verifying content."
else
    upload_link="$(
        api_get "${BASE_URL}/api/v2.1/share-links/${SHARE_TOKEN}/upload/?path=%2F" \
            | python3 -c 'import json, sys; print(json.load(sys.stdin)["upload_link"])'
    )"
    echo "Uploading without proxy in one request: ${TARGET_NAME} (${FILE_SIZE} bytes)"
    curl --fail-with-body --silent --show-error --connect-timeout 20 \
        -X POST "${upload_link}?ret-json=1" \
        -F 'parent_dir=/' \
        -F 'replace=1' \
        -F "file=@${FILE};filename=${TARGET_NAME};type=application/gzip"
    echo
    curl --fail --silent --show-error --location \
        --retry 5 --retry-all-errors --connect-timeout 20 \
        -X POST "${BASE_URL}/api/v2.1/share-links/${SHARE_TOKEN}/upload/done/" \
        -F "file_path=/${TARGET_NAME}" >/dev/null
fi

echo "Reading the complete cloud object back for SHA-256 verification."
CLOUD_SHA256="$(api_get "${DOWNLOAD_URL}" | sha256sum | awk '{print $1}')"
if [[ "${CLOUD_SHA256}" != "${FILE_SHA256}" ]]; then
    echo "Cloud SHA-256 mismatch: ${CLOUD_SHA256} != ${FILE_SHA256}" >&2
    exit 1
fi

echo "Cloud verification passed: ${TARGET_NAME} (${FILE_SIZE} bytes)"
echo "SHA-256: ${FILE_SHA256}"
echo "Share: ${BASE_URL}/d/${SHARE_TOKEN}/"
