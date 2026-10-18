#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
activate_manifeel
cd "${MANIFEEL_REPO_ROOT}"
exec "${MANIFEEL_PYTHON}" scripts/manifeel/train_full_sensor_reference.py "$@"
