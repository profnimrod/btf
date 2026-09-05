"""Preflight for speculative decoding (Ch. 17, Lab E): the draft and target
must share identical tokenizer artifacts — vocabulary, token ids, special
tokens, and template — or acceptance rates collapse and outputs diverge."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="tokenizer dir of the draft model")
    ap.add_argument("--target", required=True, help="tokenizer dir of the target model")
    a = ap.parse_args()
    d, t = Path(a.draft) / "tokenizer.json", Path(a.target) / "tokenizer.json"
    ok = digest(d) == digest(t)
    if not ok:
        from tokenizers import Tokenizer
        td, tt = Tokenizer.from_file(str(d)), Tokenizer.from_file(str(t))
        same_vocab = td.get_vocab() == tt.get_vocab()
        print(f"[tokcompat] byte-identical: False | vocab identical: {same_vocab}")
        ok = same_vocab
    print("[tokcompat]", "PASS — shared tokenizer, speculative decoding is sound"
          if ok else "FAIL — different tokenizers; use universal assisted decoding "
                     "and expect lower acceptance, or pick a same-family draft")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
