#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

DATASET_FILE="${1:-usb_quan_Aug05.zip}"
HF_REPO_URL="${MANIFEEL_HF_REPO_URL:-${MANIFEEL_HF_ENDPOINT}/datasets/purdue-mars/manifeel/resolve/main/data}"
DOWNLOAD_URL="${MANIFEEL_DOWNLOAD_URL:-${HF_REPO_URL}/${DATASET_FILE}}"
ARCHIVE_DIR="${MANIFEEL_ARCHIVE_DIR:-${MANIFEEL_DATA_ROOT}/archives}"
CONCURRENCY="${MANIFEEL_DOWNLOAD_CONCURRENCY:-16}"
CHUNK_BYTES="${MANIFEEL_DOWNLOAD_CHUNK_BYTES:-67108864}"
RETRIES="${MANIFEEL_DOWNLOAD_RETRIES:-12}"

mkdir -p "${ARCHIVE_DIR}" "${MANIFEEL_OUTPUT_ROOT}/logs"

ARCHIVE_PATH="${ARCHIVE_DIR}/${DATASET_FILE}"
PARTS_DIR="${ARCHIVE_PATH}.parts"
mkdir -p "${PARTS_DIR}"

echo "[manifeel] dataset_file=${DATASET_FILE}"
echo "[manifeel] url=${DOWNLOAD_URL}"
echo "[manifeel] archive=${ARCHIVE_PATH}"

EXPECTED_SIZE="$(
    curl -L -sI --max-time 60 "${DOWNLOAD_URL}" |
        awk 'tolower($1) == "content-length:" { n=$2 } END { gsub("\r", "", n); print n }'
)"
if [[ -z "${EXPECTED_SIZE}" || ! "${EXPECTED_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "Could not determine remote file size for ${DOWNLOAD_URL}" >&2
    exit 2
fi
echo "[manifeel] expected_size=${EXPECTED_SIZE}"

if [[ -f "${ARCHIVE_PATH}" ]]; then
    CURRENT_SIZE="$(stat -c '%s' "${ARCHIVE_PATH}")"
    if [[ "${CURRENT_SIZE}" == "${EXPECTED_SIZE}" ]]; then
        echo "[manifeel] archive already complete"
        exit 0
    fi
    echo "[manifeel] existing archive has size ${CURRENT_SIZE}; assembling from verified parts"
fi

NUM_PARTS=$(( (EXPECTED_SIZE + CHUNK_BYTES - 1) / CHUNK_BYTES ))
export DOWNLOAD_URL PARTS_DIR EXPECTED_SIZE CHUNK_BYTES RETRIES

download_part() {
    local idx="$1"
    local start=$(( idx * CHUNK_BYTES ))
    local end=$(( start + CHUNK_BYTES - 1 ))
    if (( end >= EXPECTED_SIZE )); then
        end=$(( EXPECTED_SIZE - 1 ))
    fi
    local expected=$(( end - start + 1 ))
    local part
    part="$(printf '%s/part_%05d' "${PARTS_DIR}" "${idx}")"

    if [[ -f "${part}" ]]; then
        local current
        current="$(stat -c '%s' "${part}")"
        if [[ "${current}" == "${expected}" ]]; then
            return 0
        fi
    fi

    curl -L --fail --retry "${RETRIES}" --retry-all-errors --connect-timeout 30 \
        --speed-limit 1024 --speed-time 180 --range "${start}-${end}" \
        --output "${part}" "${DOWNLOAD_URL}"

    local actual
    actual="$(stat -c '%s' "${part}")"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Part ${idx} has size ${actual}, expected ${expected}" >&2
        return 1
    fi
}
export -f download_part

seq 0 "$(( NUM_PARTS - 1 ))" | xargs -n 1 -P "${CONCURRENCY}" bash -c 'download_part "$0"'

ASSEMBLED="${ARCHIVE_PATH}.assembled.$$"
: > "${ASSEMBLED}"
for idx in $(seq 0 "$(( NUM_PARTS - 1 ))"); do
    part="$(printf '%s/part_%05d' "${PARTS_DIR}" "${idx}")"
    cat "${part}" >> "${ASSEMBLED}"
done

ACTUAL_SIZE="$(stat -c '%s' "${ASSEMBLED}")"
if [[ "${ACTUAL_SIZE}" != "${EXPECTED_SIZE}" ]]; then
    echo "Assembled file has size ${ACTUAL_SIZE}, expected ${EXPECTED_SIZE}" >&2
    exit 1
fi

mv -f "${ASSEMBLED}" "${ARCHIVE_PATH}"
echo "[manifeel] download complete: ${ARCHIVE_PATH}"
