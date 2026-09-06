"""Lab E: stage a sealed bundle into the inactive A/B slot and activate it
under built-in test, reverting automatically on BIT failure (Ch. 19).
Cross-platform equivalent of deploy/stage.sh."""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bundle.bundle import verify, activate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--target", required=True)
    ap.add_argument("--slot", default=None, help="accepted for parity; slots alternate automatically")
    a = ap.parse_args()
    print("[stage] verifying bundle at rest before staging")
    if not verify(a.bundle):
        sys.exit(1)
    print(f"[stage] activating into {a.target} (atomic slot flip + BIT)")
    sys.exit(0 if activate(a.bundle, a.target) else 1)


if __name__ == "__main__":
    main()
