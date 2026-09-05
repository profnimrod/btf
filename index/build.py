"""Build the hybrid retrieval index: structure-aware chunking, dense
vectors, BM25, reciprocal-rank fusion, optional product quantization
(Chs. 9, 16)."""
from __future__ import annotations
import argparse, pickle, re, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def chunk_document(text: str, doc_id: str, max_words: int = 90):
    """Split on blank lines (document structure), keeping the title attached
    to every chunk so meaning is never severed from its context."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if not blocks:
        return []
    title = blocks[0].split("\n")[0]
    # a title-only first block is a header, not a passage: fold it forward
    if len(blocks) > 1 and len(blocks[0].split()) < 12:
        blocks = [blocks[0] + " " + blocks[1]] + blocks[2:]
    chunks = []
    for b in blocks:
        words = b.split()
        for i in range(0, len(words), max_words):
            body = " ".join(words[i:i + max_words])
            chunks.append({"doc_id": doc_id, "title": title,
                           "text": f"{title} :: {body}"})
    return chunks


def load_corpus(src: str):
    chunks = []
    for p in sorted(Path(src).rglob("*.txt")):
        chunks += chunk_document(p.read_text(), p.stem)
    return chunks


def rrf(rankings, k0: int = 60):
    scores = {}
    for r in rankings:
        for rank, cid in enumerate(r):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k0 + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Index:
    def __init__(self, chunks, vecs=None, bm25=None, pq=None):
        self.chunks, self.vecs, self.bm25, self.pq = chunks, vecs, bm25, pq

    def save(self, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "index.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "vecs": self.vecs,
                         "pq": self.pq}, f)
        print(f"[index] {len(self.chunks)} chunks -> {out}")

    @staticmethod
    def load(path):
        with open(Path(path) / "index.pkl", "rb") as f:
            d = pickle.load(f)
        idx = Index(d["chunks"], d["vecs"], None, d.get("pq"))
        idx._fit_bm25()
        return idx

    def _fit_bm25(self):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in self.chunks])

    def search(self, query, enc=None, k=10, mode="hybrid", reranker=None,
               fuse_depth=50):
        dense_rank, lex_rank = [], []
        if mode in ("hybrid", "dense") and self.vecs is not None and enc is not None:
            qv = enc.encode([query])[0]
            sims = self.vecs @ qv
            dense_rank = list(np.argsort(-sims)[:max(k, fuse_depth)])
        if mode in ("hybrid", "bm25"):
            if self.bm25 is None:
                self._fit_bm25()
            s = self.bm25.get_scores(query.lower().split())
            lex_rank = list(np.argsort(-s)[:max(k, fuse_depth)])
        if mode == "dense":
            order = dense_rank
        elif mode == "bm25":
            order = lex_rank
        else:
            order = rrf([dense_rank, lex_rank])
        if reranker is not None:
            cand = order[:max(k * 3, 30)]
            texts = [self.chunks[i]["text"] for i in cand]
            order = [cand[j] for j in reranker.rank(query, texts)]
        return [dict(self.chunks[i], cid=int(i)) for i in order[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="corpus/v1/text")
    ap.add_argument("--dense", default=None, help="encoder checkpoint")
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--bm25", action="store_true")
    ap.add_argument("--fusion", default="rrf")
    ap.add_argument("--acl-metadata", action="store_true")
    ap.add_argument("--pq", default=None, help="e.g. m=8")
    ap.add_argument("--out", required=True, help="e.g. run/index/v1")
    a = ap.parse_args()
    chunks = load_corpus(a.src)
    vecs = None
    if a.dense:
        from index.encoder import Encoder
        enc = Encoder.load(a.dense, a.tok)
        vecs = enc.encode([c["text"] for c in chunks])
        if a.pq:
            m = int(a.pq.split("=")[1])
            from index.pq import train_pq, encode_pq
            vecs = encode_pq(vecs, train_pq(vecs, m))
            print(f"[index] product quantization m={m} applied")
    if a.acl_metadata:
        for c in chunks:
            c["markings"] = ["UNCLASSIFIED//PROXY"]
    Index(chunks, vecs).save(a.out)


if __name__ == "__main__":
    main()
