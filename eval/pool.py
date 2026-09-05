"""Pooled candidate generation for judged-set construction (Ch. 9, Lab D).
Pools from BM25, dense, and keyword scan so labels are not biased to any
one system's blind spots."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from index.build import Index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--systems", default="bm25,keyword")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = Index.load(a.index)
    enc = None
    if a.model:
        from index.encoder import Encoder
        enc = Encoder.load(a.model, a.tok)
    qs = [json.loads(l) for l in Path(a.queries).read_text(encoding='utf-8').splitlines() if l.strip()]
    out = []
    for q in qs:
        pool = {}
        for sysname in a.systems.split(","):
            mode = "dense" if sysname == "dense" else "bm25"
            for r in idx.search(q["query"], enc=enc, k=a.topk, mode=mode):
                pool[r["cid"]] = r
        out.append({"query": q["query"], "gold": q.get("gold"),
                    "candidates": [{"cid": c, "doc_id": r["doc_id"],
                                    "text": r["text"][:220]}
                                   for c, r in pool.items()]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    avg = sum(len(r["candidates"]) for r in out) / max(len(out), 1)
    print(f"[pool] {len(out)} queries, {avg:.1f} candidates each -> {a.out}")


if __name__ == "__main__":
    main()
