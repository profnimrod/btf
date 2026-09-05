"""Evaluation statistics (Ch. 15, App. A): paired bootstrap CIs for
metric deltas, McNemar's test, binomial audit-sample sizing, nDCG.
"""
from __future__ import annotations
import argparse, json, math, random
from typing import Sequence


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     iters: int = 10000, seed: int = 0):
    """95% CI for mean(b) - mean(a) over paired per-item scores."""
    assert len(a) == len(b)
    rng = random.Random(seed)
    n = len(a)
    diffs = [b[i] - a[i] for i in range(n)]
    boots = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * iters)]
    hi = boots[int(0.975 * iters)]
    return sum(diffs) / n, lo, hi


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]):
    """McNemar on discordant pairs; returns (b01, b10, chi2, significant@.05)."""
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    if b01 + b10 == 0:
        return b01, b10, 0.0, False
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)   # continuity-corrected
    return b01, b10, chi2, chi2 > 3.841


def audit_n(p_defect: float = 0.05, conf: float = 0.95) -> int:
    """Smallest n s.t. detecting >=1 defect at rate p with prob >= conf."""
    return math.ceil(math.log(1 - conf) / math.log(1 - p_defect))


def ndcg_at_k(rels: Sequence[float], k: int) -> float:
    def dcg(xs):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(xs[:k]))
    ideal = sorted(rels, reverse=True)
    z = dcg(ideal)
    return dcg(rels) / z if z > 0 else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    if a.audit or not a.demo:
        print(f"audit n (5% defect, 95% conf) = {audit_n()}")
    if a.demo:
        base = [0.0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
        tuned = [1.0, 1, 1, 1, 1, 0, 1, 1, 1, 1]
        d, lo, hi = paired_bootstrap(base, tuned, iters=2000)
        print(f"delta={d:+.3f} 95%CI=[{lo:+.3f},{hi:+.3f}]")
        print("mcnemar:", mcnemar([bool(x) for x in base], [bool(x) for x in tuned]))
        print(f"nDCG@5 = {ndcg_at_k([2,0,1,0,2,1], 5):.4f}")
