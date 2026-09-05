"""Measure tokens-per-term for a domain vs general tokenizer (Ch. 6, Lab B)."""
from __future__ import annotations
import argparse
from pathlib import Path


def frag(tok_dir, terms):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
    n = sum(len(tok.encode(t).ids) for t in terms)
    return n / max(1, len(terms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--terms", required=True)
    a = ap.parse_args()
    terms = [x.strip() for x in Path(a.terms).read_text().splitlines() if x.strip()]
    print(f"[frag] {a.tok}: {frag(a.tok, terms):.3f} tokens/term")
    if a.ref:
        print(f"[frag] {a.ref}: {frag(a.ref, terms):.3f} tokens/term")


if __name__ == "__main__":
    main()
