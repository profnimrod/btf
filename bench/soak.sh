#!/usr/bin/env bash
# Delegates to bench/soak.py (same arguments). Windows without bash: python bench/soak.py ...
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$(command -v python3 || command -v python)"
exec "$PY" bench/soak.py "$@"
