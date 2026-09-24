#!/usr/bin/env bash
set -euo pipefail

# Download the frozen FiLM Fusion checkpoints and runtime source through a
# direct ModelScope connection.
REPO_ID="${MODELSCOPE_REPO_ID:-ggkiller/multi-fm}"
REVISION="${MODELSCOPE_REVISION:-master}"
ASSET_ROOT="${ADAVIP_FILM_FUSION_ASSET_ROOT:-${MANIFEEL_CHECKPOINT_ROOT:-/data/wangzihao/checkpoints/manifeel}/film_fusion}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$(command -v modelscope || true)}"

if [[ -z "${MODELSCOPE_BIN}" || ! -x "${MODELSCOPE_BIN}" ]]; then
    echo "ModelScope CLI not found; install modelscope or set MODELSCOPE_BIN." >&2
    exit 2
fi

NO_PROXY_ENV=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy -u NO_PROXY -u no_proxy)
CLIP_DIR="${ASSET_ROOT}/clip"
SPARSH_DIR="${ASSET_ROOT}/sparsh"
SOURCE_ARCHIVE_DIR="${ASSET_ROOT}/source_archives"
CLIP_SOURCE_ROOT="${ASSET_ROOT}/source/CLIP"
SPARSH_SOURCE_ROOT="${ASSET_ROOT}/source/sparsh"
mkdir -p "${CLIP_DIR}" "${SPARSH_DIR}" "${SOURCE_ARCHIVE_DIR}"

"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" download \
    --repo-type model \
    --revision "${REVISION}" \
    --local-dir "${CLIP_DIR}" \
    "${REPO_ID}" \
    pretrained/film_fusion/clip/RN50.pt
"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" download \
    --repo-type model \
    --revision "${REVISION}" \
    --local-dir "${SPARSH_DIR}" \
    "${REPO_ID}" \
    pretrained/film_fusion/sparsh/dino_vitbase.ckpt
"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" download \
    --repo-type model \
    --revision "${REVISION}" \
    --local-dir "${SOURCE_ARCHIVE_DIR}" \
    "${REPO_ID}" \
    pretrained/film_fusion/source/CLIP_source.tar.gz
"${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" download \
    --repo-type model \
    --revision "${REVISION}" \
    --local-dir "${SOURCE_ARCHIVE_DIR}" \
    "${REPO_ID}" \
    pretrained/film_fusion/source/sparsh_source.tar.gz

CLIP_CHECKPOINT="${CLIP_DIR}/RN50.pt"
SPARSH_CHECKPOINT="${SPARSH_DIR}/dino_vitbase.ckpt"
CLIP_ARCHIVE="${SOURCE_ARCHIVE_DIR}/CLIP_source.tar.gz"
SPARSH_ARCHIVE="${SOURCE_ARCHIVE_DIR}/sparsh_source.tar.gz"
for path in "${CLIP_CHECKPOINT}" "${SPARSH_CHECKPOINT}" "${CLIP_ARCHIVE}" "${SPARSH_ARCHIVE}"; do
    if [[ ! -s "${path}" ]]; then
        echo "Downloaded asset is missing or empty: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${CLIP_SOURCE_ROOT}" "${SPARSH_SOURCE_ROOT}"
tar -xzf "${CLIP_ARCHIVE}" -C "${CLIP_SOURCE_ROOT}"
tar -xzf "${SPARSH_ARCHIVE}" -C "${SPARSH_SOURCE_ROOT}"
for path in \
    "${CLIP_SOURCE_ROOT}/clip/model.py" \
    "${SPARSH_SOURCE_ROOT}/tactile_ssl/model/__init__.py"; do
    if [[ ! -f "${path}" ]]; then
        echo "Unpacked runtime source is missing: ${path}" >&2
        exit 1
    fi
done

cat <<EOF
Downloaded FiLM Fusion checkpoints:
  CLIP:   ${CLIP_CHECKPOINT}
  Sparsh: ${SPARSH_CHECKPOINT}
  CLIP source:   ${CLIP_SOURCE_ROOT}
  Sparsh source: ${SPARSH_SOURCE_ROOT}

Before eval, export:
  export ADAVIP_FILM_FUSION_CLIP_CHECKPOINT="${CLIP_CHECKPOINT}"
  export ADAVIP_FILM_FUSION_SPARSH_CHECKPOINT="${SPARSH_CHECKPOINT}"
  export ADAVIP_FILM_FUSION_CLIP_SOURCE_ROOT="${CLIP_SOURCE_ROOT}"
  export ADAVIP_FILM_FUSION_SPARSH_SOURCE_ROOT="${SPARSH_SOURCE_ROOT}"
EOF
