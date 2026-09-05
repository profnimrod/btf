"""Failure-bin instrumentation for the RAG pipeline (Ch. 16, Table 16.1)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from index.build import Index  # noqa: E402

BINS = ["retrieval_miss", "ranking_miss", "assembly_drop", "synthesis_miss",
        "grounding_miss", "corpus_miss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", default="eval/JUDGED-v1/queries.jsonl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--deep", type=int, default=100)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    enc = None
    if a.model:
        from index.encoder import Encoder
        enc = Encoder.load(a.model, a.tok)
    idx = Index.load(a.index)
    docs = {c["doc_id"] for c in idx.chunks}
    qs = [json.loads(l) for l in Path(a.queries).read_text().splitlines() if l.strip()]
    counts = Counter()
    for q in qs:
        if q["gold"] not in docs:
            counts["corpus_miss"] += 1
            continue
        deep = idx.search(q["query"], enc=enc, k=a.deep, mode="hybrid")
        shallow = deep[:a.k]
        if any(r["doc_id"] == q["gold"] for r in shallow):
            counts["answered"] += 1
        elif any(r["doc_id"] == q["gold"] for r in deep):
            counts["ranking_miss"] += 1          # present deep, cut at k
        else:
            counts["retrieval_miss"] += 1        # never surfaced at all
    total = sum(counts.values())
    print(f"[bins] {total} queries")
    for k, v in counts.most_common():
        print(f"   {k:16s} {v:4d}  ({100*v/total:5.1f}%)")
    print("[bins] synthesis/grounding/assembly bins require the generator in "
          "the loop (Ch. 16); the oracle A/B separates them.")
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()
