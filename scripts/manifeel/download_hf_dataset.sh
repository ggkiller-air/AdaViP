#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
    echo "Usage: $0 URL OUTPUT_PATH EXPECTED_SIZE_BYTES" >&2
    exit 2
fi

URL="$1"
OUTPUT_PATH="$2"
EXPECTED_SIZE="$3"
PART_SIZE="${MANIFEEL_DOWNLOAD_PART_SIZE:-16777216}"
PARALLELISM="${MANIFEEL_DOWNLOAD_PARALLELISM:-16}"
PART_DIR="${MANIFEEL_DOWNLOAD_PART_DIR:-${OUTPUT_PATH}.parts}"
ASSEMBLING_PATH="${OUTPUT_PATH}.assembling"

if [[ "${MANIFEEL_UNSET_PROXY:-0}" == "1" ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy
fi

mkdir -p "$(dirname -- "${OUTPUT_PATH}")" "${PART_DIR}"
PART_COUNT=$(((EXPECTED_SIZE + PART_SIZE - 1) / PART_SIZE))

download_part() {
    local index="$1"
    local start=$((index * PART_SIZE))
    local end=$((start + PART_SIZE - 1))
    local expected_size="${PART_SIZE}"
    local output="$(printf '%s/part_%04d' "${PART_DIR}" "${index}")"
    local chunk_path="${output}.chunk.$$"
    if ((end >= EXPECTED_SIZE)); then
        end=$((EXPECTED_SIZE - 1))
        expected_size=$((end - start + 1))
    fi
    if [[ -f "${output}" ]] && [[ "$(stat -c '%s' "${output}")" == "${expected_size}" ]]; then
        echo "[hf] part ${index}/${PART_COUNT}: complete"
        return 0
    fi
    touch "${output}"
    local failures=0 actual_size=0 chunk_size=0 curl_status=0 request_start=0
    while true; do
        actual_size="$(stat -c '%s' "${output}")"
        [[ "${actual_size}" == "${expected_size}" ]] && break
        if ((actual_size > expected_size)); then
            : > "${output}"
            actual_size=0
        fi
        request_start=$((start + actual_size))
        : > "${chunk_path}"
        set +e
        curl --location --fail --silent --show-error --connect-timeout 20 \
            --max-time "${MANIFEEL_DOWNLOAD_MAX_TIME:-180}" --retry 3 \
            --retry-all-errors --retry-delay 1 --range "${request_start}-${end}" \
            --output "${chunk_path}" "${URL}"
        curl_status=$?
        set -e
        chunk_size=0
        [[ -f "${chunk_path}" ]] && chunk_size="$(stat -c '%s' "${chunk_path}")"
        if ((chunk_size > 0)); then
            cat "${chunk_path}" >> "${output}"
            failures=0
        else
            failures=$((failures + 1))
            if ((failures >= ${MANIFEEL_DOWNLOAD_MAX_EMPTY_RETRIES:-80})); then
                echo "[hf] part ${index} stalled; curl status ${curl_status}" >&2
                return 1
            fi
            sleep 2
        fi
    done
    rm -f "${chunk_path}"
    echo "[hf] part ${index}/${PART_COUNT}: complete"
}

export -f download_part
export URL EXPECTED_SIZE PART_SIZE PART_COUNT PART_DIR MANIFEEL_DOWNLOAD_MAX_TIME
export MANIFEEL_DOWNLOAD_MAX_EMPTY_RETRIES
seq 0 $((PART_COUNT - 1)) | xargs -n 1 -P "${PARALLELISM}" bash -c 'download_part "$1"' _

: > "${ASSEMBLING_PATH}"
for ((index = 0; index < PART_COUNT; index++)); do
    part_path="$(printf '%s/part_%04d' "${PART_DIR}" "${index}")"
    dd if="${part_path}" of="${ASSEMBLING_PATH}" bs="${PART_SIZE}" seek="${index}" conv=notrunc status=none
done
actual_size="$(stat -c '%s' "${ASSEMBLING_PATH}")"
[[ "${actual_size}" == "${EXPECTED_SIZE}" ]] || { echo "[hf] assembled size mismatch" >&2; exit 1; }
mv "${ASSEMBLING_PATH}" "${OUTPUT_PATH}"
echo "[hf] verified archive: ${OUTPUT_PATH} (${actual_size} bytes)"
