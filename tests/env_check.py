"""Every lab calls this before its first step: verify Python, key
libraries, and (for GPU labs) device availability."""
from __future__ import annotations
import argparse, importlib, sys

CPU_CORE = ["numpy", "yaml", "tokenizers", "datasketch", "cryptography"]
LAB_EXTRA = {
    "B": ["torch"],
    "C": ["torch", "transformers", "peft", "trl"],
    "D": ["torch", "sentence_transformers", "faiss", "sklearn"],
    "E": ["torch", "transformers"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=None, choices=list(LAB_EXTRA))
    a = ap.parse_args()
    print(f"python {sys.version.split()[0]}")
    need = list(CPU_CORE) + (LAB_EXTRA.get(a.lab, []) if a.lab else [])
    missing = []
    for mod in need:
        try:
            m = importlib.import_module(mod)
            print(f"  ok  {mod:22s} {getattr(m, '__version__', '')}")
        except Exception:
            missing.append(mod)
            print(f"  --  {mod:22s} MISSING")
    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        pass
    print(f"GPU available: {gpu}")
    if a.lab in ("B", "C", "E") and not gpu:
        print("note: this lab's full run needs a GPU; --plan modes and tests "
              "run on CPU.")
    if missing:
        print(f"MISSING: {missing} — install env/requirements.lock")
        sys.exit(1)
    print("env check: OK")


if __name__ == "__main__":
    main()
