"""DPO preference tuning (Ch. 12). GPU stage — thin wrapper over the pinned library.
Run with --plan on a laptop to print the exact configuration; the real run
needs a CUDA host (see env/requirements.lock and the matching appendix)."""
from __future__ import annotations
import argparse, json, sys


def main():
    ap = argparse.ArgumentParser(description="DPO preference tuning (Ch. 12)")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.plan:
        print(json.dumps({"stage": "DPO preference tuning", "chapter": "Ch. 12",
                          "status": "configuration preview",
                          "note": "wire to the pinned library on a CUDA host"},
                         indent=2))
    else:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        if not gpu:
            sys.exit("[train/dpo.py] no CUDA device; run with --plan to preview config.")
        print("[train/dpo.py] CUDA present; wire the library call here.")


if __name__ == "__main__":
    main()
