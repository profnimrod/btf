"""Bits per byte — the tokenizer-independent comparison (Ch. 7, Lab B).
BPB = (mean_loss_nats / ln 2) * (tokens / bytes) on the same text."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.model import Model, ModelConfig  # noqa: E402


def bpb(ckpt, tok_dir, text, seq=256):
    from tokenizers import Tokenizer
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = Model(ModelConfig(**st["cfg"]))
    m.load_state_dict(st["model"])
    m.eval()
    tok = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
    ids = tok.encode(text).ids
    n_bytes = len(text.encode())
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, max(1, len(ids) - seq - 1), seq):
            x = torch.tensor([ids[i:i + seq]])
            y = torch.tensor([ids[i + 1:i + seq + 1]])
            if x.shape[1] < 2 or x.shape != y.shape:
                break
            _, loss = m(x, y)
            tot += loss.item() * x.shape[1]
            n += x.shape[1]
    mean_nats = tot / max(n, 1)
    return {"mean_loss_nats": round(mean_nats, 4),
            "tokens": len(ids), "bytes": n_bytes,
            "tokens_per_byte": round(len(ids) / n_bytes, 4),
            "bpb": round((mean_nats / math.log(2)) * (len(ids) / n_bytes), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--text", required=True, help="held-out text file or dir")
    ap.add_argument("--label", default="model")
    a = ap.parse_args()
    p = Path(a.text)
    text = "\n".join(f.read_text(encoding='utf-8') for f in sorted(p.rglob("*.txt"))[:40]) \
        if p.is_dir() else p.read_text(encoding='utf-8')
    r = bpb(a.ckpt, a.tok, text)
    print(f"[bpb] {a.label}: {json.dumps(r)}")


if __name__ == "__main__":
    main()
