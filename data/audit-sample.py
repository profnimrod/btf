"""Draw the acceptance-audit sample for a dataset (Ch. 13).
n >= ln(1-conf)/ln(1-p) — 59 records detects a 5% defect rate at 95%."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.stats import audit_n  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--defect", type=float, default=0.05)
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--out", default="data/audit-sample.jsonl")
    a = ap.parse_args()
    need = a.n or audit_n(a.defect, a.conf)
    rows = [l for l in Path(a.data).read_text().splitlines() if l.strip()]
    sample = random.Random(0).sample(rows, min(need, len(rows)))
    Path(a.out).write_text("\n".join(sample) + "\n")
    print(f"[audit] required n={need} for {a.defect:.0%} defect @ {a.conf:.0%} "
          f"confidence; drew {len(sample)} of {len(rows)} -> {a.out}")
    print("[audit] read every drawn record by hand; a failed batch stops the "
          "pipeline (Ch. 13).")


if __name__ == "__main__":
    main()
