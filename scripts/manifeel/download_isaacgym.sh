#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

ISAACGYM_FILE_ID="${MANIFEEL_ISAACGYM_FILE_ID:-13dFRF9EXpzIWaJF2Z6f7BsuPUGQkPE8v}"
ISAACGYM_URL="https://drive.usercontent.google.com/download?id=${ISAACGYM_FILE_ID}&export=download&confirm=t"
EXPECTED_SIZE="${MANIFEEL_ISAACGYM_ARCHIVE_SIZE:-268908854}"
PART_SIZE="${MANIFEEL_DOWNLOAD_PART_SIZE:-4194304}"
PARALLELISM="${MANIFEEL_DOWNLOAD_PARALLELISM:-8}"
DOWNLOAD_PROXY="${MANIFEEL_DOWNLOAD_PROXY:-socks5h://127.0.0.1:17891}"
PART_DIR="${MANIFEEL_DOWNLOAD_PART_DIR:-${MANIFEEL_ISAACGYM_ARCHIVE}.parts_4m}"
ASSEMBLING_PATH="${MANIFEEL_ISAACGYM_ARCHIVE}.assembling"

mkdir -p "${PART_DIR}"
PART_COUNT=$(((EXPECTED_SIZE + PART_SIZE - 1) / PART_SIZE))

download_part() {
    local index="$1"
    local start=$((index * PART_SIZE))
    local end=$((start + PART_SIZE - 1))
    local expected_size="${PART_SIZE}"
    local output

    if ((end >= EXPECTED_SIZE)); then
        end=$((EXPECTED_SIZE - 1))
        expected_size=$((end - start + 1))
    fi
    output="$(printf '%s/part_%03d' "${PART_DIR}" "${index}")"

    if [[ -f "${output}" ]] && [[ "$(stat -c '%s' "${output}")" == "${expected_size}" ]]; then
        echo "[manifeel] part ${index}/${PART_COUNT}: already complete"
        return 0
    fi

    echo "[manifeel] part ${index}/${PART_COUNT}: bytes ${start}-${end}"
    local failures=0
    local actual_size=0
    local chunk_path="${output}.chunk"
    touch "${output}"
    while true; do
        actual_size="$(stat -c '%s' "${output}")"
        if [[ "${actual_size}" == "${expected_size}" ]]; then
            break
        fi
        if ((actual_size > expected_size)); then
            echo "Part ${index} has ${actual_size} bytes; expected ${expected_size}. Restarting part." >&2
            truncate --size 0 "${output}"
            actual_size=0
        fi

        local request_start=$((start + actual_size))
        rm -f "${chunk_path}"
        set +e
        curl --location --fail --silent --show-error \
            --proxy "${DOWNLOAD_PROXY}" \
            --connect-timeout 20 \
            --max-time "${MANIFEEL_DOWNLOAD_MAX_TIME:-120}" \
            --retry 3 \
            --retry-all-errors \
            --retry-delay 1 \
            --range "${request_start}-${end}" \
            --output "${chunk_path}" \
            "${ISAACGYM_URL}"
        local curl_status=$?
        set -e

        local chunk_size=0
        if [[ -f "${chunk_path}" ]]; then
            chunk_size="$(stat -c '%s' "${chunk_path}")"
        fi

        if ((chunk_size > 0)); then
            cat "${chunk_path}" >> "${output}"
            failures=0
        else
            failures=$((failures + 1))
            if ((failures >= ${MANIFEEL_DOWNLOAD_MAX_EMPTY_RETRIES:-80})); then
                echo "Part ${index} stalled after ${failures} empty retries; last curl status ${curl_status}." >&2
                return 1
            fi
            sleep 2
        fi
    done
    rm -f "${chunk_path}"
}

export -f download_part
export ISAACGYM_URL EXPECTED_SIZE PART_SIZE PART_COUNT PART_DIR DOWNLOAD_PROXY
seq 0 $((PART_COUNT - 1)) | xargs -n 1 -P "${PARALLELISM}" bash -c 'download_part "$1"' _

truncate --size 0 "${ASSEMBLING_PATH}"
for ((index = 0; index < PART_COUNT; index++)); do
    part_path="$(printf '%s/part_%03d' "${PART_DIR}" "${index}")"
    dd if="${part_path}" of="${ASSEMBLING_PATH}" \
        bs="${PART_SIZE}" seek="${index}" conv=notrunc status=none
done

actual_size="$(stat -c '%s' "${ASSEMBLING_PATH}")"
if [[ "${actual_size}" != "${EXPECTED_SIZE}" ]]; then
    echo "Archive has ${actual_size} bytes; expected ${EXPECTED_SIZE}." >&2
    exit 1
fi
gzip -t "${ASSEMBLING_PATH}"
mv "${ASSEMBLING_PATH}" "${MANIFEEL_ISAACGYM_ARCHIVE}"
echo "[manifeel] verified archive: ${MANIFEEL_ISAACGYM_ARCHIVE} (${actual_size} bytes)"
