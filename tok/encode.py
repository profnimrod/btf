"""Encode a built corpus (corpus/<v>/text/) into train.bin with a trained
tokenizer. Called automatically by train/pretrain.py when train.bin is
absent; usable standalone."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


def load_tokenizer(tok_dir: str):
    from tokenizers import Tokenizer
    p = Path(tok_dir) / "tokenizer.json"
    if not p.exists():
        sys.exit(f"tokenizer not found: {p} (run tok/train.py first)")
    return Tokenizer.from_file(str(p))


def encode_dir(corpus_dir: str, tok_dir: str) -> Path:
    corpus = Path(corpus_dir)
    text_dir = corpus / "text"
    files = sorted(list(text_dir.glob("*.txt")) + list(text_dir.glob("*.md")))
    if not files:
        sys.exit(f"no text files under {text_dir}")
    tok = load_tokenizer(tok_dir)
    eos = tok.token_to_id("<|eot|>")
    if eos is None:
        eos = 0
    ids: list[int] = []
    for f in files:
        ids.extend(tok.encode(f.read_text(encoding='utf-8', errors='ignore')).ids)
        ids.append(eos)
    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab < 65536 else np.uint32
    arr = np.asarray(ids, dtype=dtype)
    out = corpus / "train.bin"
    arr.tofile(out)
    meta = {"tokenizer": str(tok_dir), "vocab": vocab,
            "n_tokens": int(arr.size), "dtype": str(arr.dtype)}
    (corpus / "meta.json").write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"[encode] {arr.size:,} tokens -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tok", required=True)
    a = ap.parse_args()
    encode_dir(a.corpus, a.tok)
