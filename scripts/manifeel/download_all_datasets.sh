#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if (( $# > 0 )); then
    DATASET_FILES=("$@")
else
    DATASET_FILES=(
        usb_quan_Aug05.zip
        sorting_quan_Aug8.zip
        gear_quan_Sep15.zip
        pih_quan_June06.zip
        plug_quan_Aug02.zip
        nutbolt_quan_July1.zip
        bulb_quan_Sep19.zip
        blindinsert_quan_Aug15.zip
        explore_quan_June17.zip
    )
fi

ARCHIVE_DIR="${MANIFEEL_ARCHIVE_DIR:-${MANIFEEL_DATA_ROOT}/archives}"
mkdir -p "${ARCHIVE_DIR}" "${MANIFEEL_DATA_ROOT}"

for dataset_file in "${DATASET_FILES[@]}"; do
    archive_path="${ARCHIVE_DIR}/${dataset_file}"
    dataset_dir="${MANIFEEL_DATA_ROOT}/${dataset_file%.zip}"

    "${SCRIPT_DIR}/download_dataset.sh" "${dataset_file}"
    unzip -tq "${archive_path}"

    if [[ -f "${dataset_dir}/.zgroup" ]]; then
        echo "[manifeel] dataset already extracted: ${dataset_dir}"
        continue
    fi

    # Official archives already contain a top-level directory named after the
    # dataset. Extracting into dataset_dir would create dataset/dataset/.zgroup.
    unzip -oq "${archive_path}" -d "${MANIFEEL_DATA_ROOT}"

    # Some archives contain Zarr at the root, while others wrap it in a
    # directory named after the archive. Keep one canonical dataset path.
    nested_dir="${dataset_dir}/${dataset_file%.zip}"
    if [[ ! -f "${dataset_dir}/.zgroup" && -f "${nested_dir}/.zgroup" ]]; then
        find "${nested_dir}" -mindepth 1 -maxdepth 1 \
            -exec mv -t "${dataset_dir}" -- {} +
        rmdir "${nested_dir}"
    fi
    if [[ ! -f "${dataset_dir}/.zgroup" ]]; then
        echo "Extracted dataset is missing .zgroup: ${dataset_dir}" >&2
        exit 1
    fi
    echo "[manifeel] dataset ready: ${dataset_dir}"
done

echo "[manifeel] all official datasets are downloaded and extracted"
