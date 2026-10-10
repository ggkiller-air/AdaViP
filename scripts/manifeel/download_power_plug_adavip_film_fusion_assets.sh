#!/usr/bin/env bash
set -euo pipefail

# Download the frozen FiLM Fusion checkpoints and runtime source through a
# direct ModelScope connection.
REPO_ID="${MODELSCOPE_REPO_ID:-ggkiller/multi-fm}"
REVISION="${MODELSCOPE_REVISION:-master}"
ASSET_ROOT="${ADAVIP_FILM_FUSION_ASSET_ROOT:-${MANIFEEL_CHECKPOINT_ROOT:-/data/wangzihao/checkpoints/manifeel}/film_fusion}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-$(command -v modelscope || true)}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${MODELSCOPE_BIN}" || ! -x "${MODELSCOPE_BIN}" ]]; then
    echo "ModelScope CLI not found; install modelscope or set MODELSCOPE_BIN." >&2
    exit 2
fi

NO_PROXY_ENV=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u ALL_PROXY -u all_proxy -u NO_PROXY -u no_proxy)
DOWNLOADED_ROOT="${ASSET_ROOT}/pretrained/film_fusion"
CLIP_SOURCE_ROOT="${REPO_ROOT}/third_party/CLIP"
SPARSH_SOURCE_ROOT="${REPO_ROOT}/third_party/sparsh"
mkdir -p "${ASSET_ROOT}"

for relative_path in \
    pretrained/film_fusion/clip/RN50.pt \
    pretrained/film_fusion/sparsh/dino_vitbase.ckpt \
    pretrained/film_fusion/source/CLIP_source.tar.gz \
    pretrained/film_fusion/source/sparsh_source.tar.gz; do
    if [[ -s "${ASSET_ROOT}/${relative_path}" ]]; then
        continue
    fi
    "${NO_PROXY_ENV[@]}" "${MODELSCOPE_BIN}" download \
        --repo-type model \
        --revision "${REVISION}" \
        --local-dir "${ASSET_ROOT}" \
        "${REPO_ID}" \
        "${relative_path}"
done

CLIP_CHECKPOINT="${DOWNLOADED_ROOT}/clip/RN50.pt"
SPARSH_CHECKPOINT="${DOWNLOADED_ROOT}/sparsh/dino_vitbase.ckpt"
CLIP_ARCHIVE="${DOWNLOADED_ROOT}/source/CLIP_source.tar.gz"
SPARSH_ARCHIVE="${DOWNLOADED_ROOT}/source/sparsh_source.tar.gz"
for path in "${CLIP_CHECKPOINT}" "${SPARSH_CHECKPOINT}" "${CLIP_ARCHIVE}" "${SPARSH_ARCHIVE}"; do
    if [[ ! -s "${path}" ]]; then
        echo "Downloaded asset is missing or empty: ${path}" >&2
        exit 1
    fi
done

mkdir -p "${CLIP_SOURCE_ROOT}" "${SPARSH_SOURCE_ROOT}"
if [[ ! -f "${CLIP_SOURCE_ROOT}/clip/model.py" ]]; then
    tar -xzf "${CLIP_ARCHIVE}" -C "${CLIP_SOURCE_ROOT}"
fi
if [[ ! -f "${SPARSH_SOURCE_ROOT}/tactile_ssl/model/__init__.py" ]]; then
    tar -xzf "${SPARSH_ARCHIVE}" -C "${SPARSH_SOURCE_ROOT}"
fi
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

The downloaded checkpoints embed their original absolute paths. The local
single-task evaluator remaps those saved paths to these assets automatically.
EOF
