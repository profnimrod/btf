#!/usr/bin/env bash
# Lab E: stage a sealed bundle into the inactive A/B slot and activate it
# under built-in test, reverting automatically on BIT failure (Ch. 19).
set -euo pipefail
TARGET=""; SLOT=""; BUNDLE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2;;
    --slot) SLOT="$2"; shift 2;;
    *) BUNDLE="$1"; shift;;
  esac
done
[ -n "$BUNDLE" ] && [ -n "$TARGET" ] || {
  echo "usage: deploy/stage.sh --target <fleet-dir> [--slot B] <bundle-dir>"; exit 1; }
echo "[stage] verifying bundle at rest before staging"
python3 bundle/bundle.py verify --dir "$BUNDLE"
echo "[stage] activating into ${TARGET} (atomic slot flip + BIT)"
python3 bundle/bundle.py activate --dir "$BUNDLE" --target "$TARGET"
