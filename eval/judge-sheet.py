"""Produce judgement sheets and compute inter-judge agreement (Ch. 9, 15).

In the lab, two simulated judges grade the pool against the rubric with a
small disagreement rate, so Cohen's kappa is a real computed number and
the adjudication step has something to adjudicate."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path


def kappa(a, b):
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--judges", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--out", default="eval/JUDGED-v1/judgements.jsonl")
    a = ap.parse_args()
    rng = random.Random(7)
    rows = [json.loads(l) for l in Path(a.pool).read_text(encoding='utf-8').splitlines() if l.strip()]
    j1, j2, judged = [], [], []
    for r in rows:
        for c in r["candidates"]:
            truth = 2 if c["doc_id"] == r["gold"] else 0
            g1 = truth if rng.random() > a.noise else rng.choice([0, 1, 2])
            judged.append({"query": r["query"], "cid": c["cid"],
                           "doc_id": c["doc_id"], "rel": g1})
            if rng.random() < a.overlap:
                g2 = truth if rng.random() > a.noise else rng.choice([0, 1, 2])
                j1.append(g1); j2.append(g2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding='utf-8') as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    k = kappa(j1, j2) if j1 else float("nan")
    print(f"[judge] {len(judged)} judgements; overlap {len(j1)}; "
          f"Cohen's kappa = {k:.3f} "
          f"({'acceptable' if k >= 0.6 else 'REVISE THE RUBRIC'})")


if __name__ == "__main__":
    main()
