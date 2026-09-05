"""Lab D: cross-encoder reranker over fused candidates (Ch. 9).
Scores (query, passage) jointly with a scalar head trained on the same
judged pairs, using mined negatives as the contrast."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from index.encoder import Encoder  # noqa: E402
from train.embedder import fresh_backbone  # noqa: E402


class CrossEncoder(nn.Module):
    def __init__(self, backbone, tok, max_len=160):
        super().__init__()
        self.enc = Encoder(backbone, tok, max_len)
        self.head = nn.Linear(backbone.cfg.d_model, 1)

    def score_batch(self, queries, passages):
        texts = [f"{q} [SEP] {p}" for q, p in zip(queries, passages)]
        idx = self.enc.ids(texts)
        x = self.enc.model.emb(idx)
        for b in self.enc.model.blocks:
            x = b(x, self.enc.model.cos, self.enc.model.sin)
        x = self.enc.model.norm_f(x)
        mask = (idx != 0).float().unsqueeze(-1)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def rank(self, query, passages):
        s = self.score_batch([query] * len(passages), passages)
        return list(np.argsort(-s.numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--hard-negs", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="ckpt/reranker.pt")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.tok) / "tokenizer.json"))
    backbone = fresh_backbone(tok.get_vocab_size(), d=96, layers=3, heads=4,
                              kv=2, dh=24, ff=384, ctx=192)
    ce = CrossEncoder(backbone, tok)
    pairs = [json.loads(l) for l in Path(a.pairs).read_text(encoding='utf-8').splitlines() if l.strip()]
    negs = json.loads(Path(a.hard_negs).read_text(encoding='utf-8')) if a.hard_negs else {}
    opt = torch.optim.AdamW(ce.parameters(), lr=a.lr)
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        tot, n = 0.0, 0
        order = rng.permutation(len(pairs))
        for s in range(0, len(order) - a.batch + 1, a.batch):
            b = [pairs[i] for i in order[s:s + a.batch]]
            qs = [x["query"] for x in b] * 2
            pos = [x["positive"] for x in b]
            neg = [(negs.get(x["query"]) or [rng.choice(pairs)["positive"]])[0]
                   for x in b]
            y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg))
            logits = ce.score_batch(qs, pos + neg)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        print(f"  [rerank] epoch {ep + 1}: loss {tot / max(n,1):.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": ce.state_dict(), "cfg": backbone.cfg.__dict__,
                "tok": a.tok}, a.out)
    print(f"[rerank] saved -> {a.out}")


if __name__ == "__main__":
    main()
