"""Small domain encoder: mean-pooled transformer embeddings + InfoNCE
fine-tuning (Ch. 9). Reuses the Lab B model as the backbone so the whole
retrieval lab runs on CPU."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.model import Model, ModelConfig  # noqa: E402


class Encoder:
    def __init__(self, model: Model, tok, max_len: int = 192):
        self.model, self.tok, self.max_len = model, tok, max_len

    @staticmethod
    def load(path: str, tok_dir: str, max_len: int = 192) -> "Encoder":
        from tokenizers import Tokenizer
        st = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**st["cfg"])
        m = Model(cfg)
        m.load_state_dict(st["model"])
        m.eval()
        tok = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
        return Encoder(m, tok, max_len)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(),
                    "cfg": self.model.cfg.__dict__}, path)

    def ids(self, texts):
        out = []
        for t in texts:
            i = self.tok.encode(t).ids[: self.max_len]
            out.append(i + [0] * (self.max_len - len(i)))
        return torch.tensor(out, dtype=torch.long)

    def _embed_ids(self, idx):
        x = self.model.emb(idx)
        for b in self.model.blocks:
            x = b(x, self.model.cos, self.model.sin)
        x = self.model.norm_f(x)
        mask = (idx != 0).float().unsqueeze(-1)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)

    def encode(self, texts, batch: int = 32) -> np.ndarray:
        vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                vecs.append(self._embed_ids(self.ids(texts[i:i + batch])).numpy())
        return np.concatenate(vecs, 0) if vecs else np.zeros((0, 1))

    def embed_grad(self, texts):
        return self._embed_ids(self.ids(texts))


def infonce(q: torch.Tensor, d: torch.Tensor, tau: float = 0.05, mrl_dims=None):
    """In-batch negatives: row i of q matches row i of d. With mrl_dims, the
    Matryoshka nested objective sums the loss over leading-dimension prefixes so
    truncated embeddings stay usable (Ch. 9)."""
    def one(qq, dd):
        sim = F.normalize(qq, dim=-1) @ F.normalize(dd, dim=-1).T / tau
        return F.cross_entropy(sim, torch.arange(qq.shape[0]))
    if not mrl_dims:
        return one(q, d)
    return sum(one(q[:, :k], d[:, :k]) for k in mrl_dims) / len(mrl_dims)


def train_infonce(enc: Encoder, pairs, epochs=2, lr=3e-4, batch=16, tau=0.05,
                  hard_negs=None, mrl_dims=None):
    opt = torch.optim.AdamW(enc.model.parameters(), lr=lr)
    enc.model.train()
    n = len(pairs)
    for ep in range(epochs):
        idx = np.random.default_rng(ep).permutation(n)
        tot, steps = 0.0, 0
        for s in range(0, n - batch + 1, batch):
            b = [pairs[i] for i in idx[s:s + batch]]
            q = enc.embed_grad([x["query"] for x in b])
            d = enc.embed_grad([x["positive"] for x in b])
            loss = infonce(q, d, tau, mrl_dims)
            if hard_negs:
                hn = [(hard_negs.get(x["query"]) or [None])[0] for x in b]
                keep = [(i, t) for i, t in enumerate(hn) if t]
                if keep:
                    nv = enc.embed_grad([t for _, t in keep])
                    qs = q[[i for i, _ in keep]]
                    # explicit negative term: push query away from hard negative
                    loss = loss + 0.3 * F.softplus((qs * nv).sum(-1) / tau).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            steps += 1
        print(f"  [infonce] epoch {ep + 1}: loss {tot / max(steps, 1):.4f}")
    enc.model.eval()
    return enc
