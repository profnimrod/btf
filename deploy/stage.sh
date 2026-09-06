#!/usr/bin/env bash
# Delegates to deploy/stage.py (same arguments). Windows without bash: python deploy/stage.py ...
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$(command -v python3 || command -v python)"
exec "$PY" deploy/stage.py "$@"
