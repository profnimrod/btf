"""Lab D: fine-tune the domain embedder with InfoNCE and optional hard
negatives (Ch. 9). Runs on CPU at lab scale; the same script scales to a
GPU host with a larger backbone."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from index.encoder import Encoder, train_infonce  # noqa: E402
from train.model import Model, ModelConfig  # noqa: E402


def fresh_backbone(vocab, d=128, layers=4, heads=4, kv=2, dh=32, ff=512, ctx=256):
    return Model(ModelConfig(vocab=vocab, d_model=d, n_layers=layers,
                             n_heads=heads, n_kv_heads=kv, d_head=dh,
                             d_ff=ff, context=ctx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None,
                    help="LM checkpoint to initialise from (Lab B), or omit for fresh")
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--hard-negs", default=None)
    ap.add_argument("--loss", default="infonce")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-cache", action="store_true")
    ap.add_argument("--mrl", default=None, help="nested dims, e.g. 128,64,32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.tok) / "tokenizer.json"))
    if a.base and Path(a.base).exists():
        enc = Encoder.load(a.base, a.tok)
        print(f"[embedder] initialised from {a.base}")
    else:
        enc = Encoder(fresh_backbone(tok.get_vocab_size()), tok)
        print("[embedder] fresh backbone (no LM checkpoint supplied)")
    pairs = [json.loads(l) for l in Path(a.pairs).read_text(encoding='utf-8').splitlines() if l.strip()]
    negs = json.loads(Path(a.hard_negs).read_text(encoding='utf-8')) if a.hard_negs else None
    print(f"[embedder] {len(pairs)} pairs"
          + (f", hard negatives for {len(negs)}" if negs else ""))
    mrl = [int(x) for x in a.mrl.split(',')] if a.mrl else None
    train_infonce(enc, pairs, epochs=a.epochs, lr=a.lr, batch=a.batch,
                  tau=a.tau, hard_negs=negs, mrl_dims=mrl)
    enc.save(a.out)
    print(f"[embedder] saved -> {a.out}")


if __name__ == "__main__":
    main()
