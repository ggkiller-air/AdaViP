#!/usr/bin/env bash
# Slurm steps can have a different /dev/shm mount from the batch shell.
set -euo pipefail
: "${TMPDIR:?Set a short per-job multiprocessing directory}"
mkdir -p -- "${TMPDIR}"
exec "$@"
