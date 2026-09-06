#!/usr/bin/env bash
# Delegates to the cross-platform runner. On Windows without bash, run:
#   python labs/run_all.py [--smoke]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$(command -v python3 || command -v python)"
exec "$PY" labs/run_all.py "$@"
