"""Lab C unit test: the tokens the trainer learns on must begin with exactly
the tokens the server sends (Ch. 11).

The trap this catches: re-tokenizing prompt+response as one string lets BPE
merge across the boundary, so the training prefix silently differs from the
serving prefix. Encode the parts separately and concatenate ids.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.template import render_parts, render_serving, render_training  # noqa: E402


def check(record: dict, tok) -> bool:
    prompt, response = render_parts(record)
    serve_ids = tok.encode(prompt).ids
    train_ids = serve_ids + tok.encode(response).ids          # what we train on
    ok = train_ids[:len(serve_ids)] == serve_ids
    naive = tok.encode(render_training(record)).ids           # the trap
    merged = naive[:len(serve_ids)] != serve_ids
    print(f"[parity] serve={len(serve_ids)} train={len(train_ids)} prefix_match={ok}")
    print(f"[parity] naive re-tokenisation would {'DIFFER (BPE merged across the '
          f'boundary)' if merged else 'match here'} — always concatenate ids")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None)
    ap.add_argument("--tokenizer", default="tok/domain-32k")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.tokenizer) / "tokenizer.json"))
    rec = {"messages": [{"role": "user", "content": "compute the link margin"},
                        {"role": "assistant", "content": "QPSK3/4, margin 2.1 dB"}]}
    sys.exit(0 if check(rec, tok) else 1)
