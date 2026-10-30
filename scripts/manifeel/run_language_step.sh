#!/usr/bin/env bash
# Use a persistent short socket path and verify actual worker tensor transport.
set -euo pipefail
: "${TMPDIR:?Set a short persistent multiprocessing directory}"
: "${1:?Pass the training Python executable}"
mkdir -p -- "${TMPDIR}"
"${1}" -c 'import json; from adavip.manifeel.training_runtime import check_dataloader_ipc; print(json.dumps(check_dataloader_ipc()), flush=True)'
exec "$@"
