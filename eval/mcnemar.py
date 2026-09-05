"""McNemar's test on paired per-item correctness (Ch. 15)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.stats import mcnemar, paired_bootstrap  # noqa: E402


def load_items(p):
    d = json.loads(Path(p).read_text(encoding='utf-8'))
    return d["per_item"] if isinstance(d, dict) and "per_item" in d else d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    a = ap.parse_args()
    A, B = load_items(a.a), load_items(a.b)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    b01, b10, chi2, sig = mcnemar([bool(x) for x in A], [bool(x) for x in B])
    d, lo, hi = paired_bootstrap([float(x) for x in A], [float(x) for x in B],
                                 iters=3000)
    print(f"[mcnemar] n={n} a-only={b01} b-only={b10} chi2={chi2:.3f} "
          f"significant@0.05={sig}")
    print(f"[bootstrap] delta={d:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
