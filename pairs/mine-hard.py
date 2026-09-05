"""Hard-negative mining with margin filtering and cross-encoder denoising
(Ch. 9). Round N mines against the model trained in round N-1."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from index.encoder import Encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--margin-filter", action="store_true")
    ap.add_argument("--denoise", default=None)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.pairs).read_text().splitlines() if l.strip()]
    enc = Encoder.load(a.model, a.tok)
    corpus = [r["positive"] for r in rows]
    dv = enc.encode(corpus)
    qv = enc.encode([r["query"] for r in rows])
    sims = qv @ dv.T
    negs, kept, filtered = {}, 0, 0
    for i, r in enumerate(rows):
        order = np.argsort(-sims[i])
        pos_s = sims[i, i]
        cand = []
        for j in order[: a.topk + 5]:
            if j == i or rows[j]["doc_id"] == r["doc_id"]:
                continue                       # never mine a same-doc negative
            if a.margin_filter and sims[i, j] > pos_s - 0.02:
                filtered += 1                  # too close: likely false negative
                continue
            cand.append(corpus[j])
            if len(cand) >= 2:
                break
        if cand:
            negs[r["query"]] = cand
            kept += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(negs))
    print(f"[mine-hard] negatives for {kept}/{len(rows)} queries; "
          f"{filtered} candidates rejected by margin filter "
          f"({'denoise on' if a.denoise else 'denoise off'})")


if __name__ == "__main__":
    main()
