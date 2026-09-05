"""Retrieval metrics against the judged set (Chs. 9, 15; Lab D)."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from index.build import Index  # noqa: E402


def ndcg(hits, k):
    """Normalized DCG with binary relevance. The ideal ranking places all
    retrieved relevant chunks first, so idcg is summed over the number of
    hits actually present (never over 1 item only)."""
    hits = hits[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    n_rel = sum(hits)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(index_path, queries_path, enc=None, k=10, mode="hybrid"):
    idx = Index.load(index_path)
    qs = [json.loads(l) for l in Path(queries_path).read_text().splitlines() if l.strip()]
    rec, nd, per_item = 0, 0.0, []
    for q in qs:
        res = idx.search(q["query"], enc=enc, k=k, mode=mode)
        hits = [1 if r["doc_id"] == q["gold"] else 0 for r in res]
        got = 1 if any(hits) else 0
        rec += got
        nd += ndcg(hits, k)
        per_item.append(got)
    n = len(qs)
    return {"n": n, f"recall@{k}": round(rec / n, 4),
            f"ndcg@{k}": round(nd / n, 4)}, per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", default="eval/JUDGED-v1/queries.jsonl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    enc = None
    if a.model:
        from index.encoder import Encoder
        enc = Encoder.load(a.model, a.tok)
    m, per_item = evaluate(a.index, a.queries, enc, a.k, a.mode)
    print(f"[retrieval] mode={a.mode} {m}")
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps({"metrics": m, "per_item": per_item}))


if __name__ == "__main__":
    main()
