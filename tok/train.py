"""Train a domain BPE tokenizer with byte fallback and digit splitting (Ch. 6)."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--digits", choices=["split", "keep"], default="split")
    ap.add_argument("--out", default="tok/domain-32k")
    a = ap.parse_args()
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    pre = [pre_tokenizers.ByteLevel(add_prefix_space=True)]
    if a.digits == "split":
        pre.insert(0, pre_tokenizers.Digits(individual_digits=True))
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence(pre)
    tok.decoder = decoders.ByteLevel()   # decode back to clean text
    files = [str(p) for p in Path(a.corpus).rglob("*.txt")] + \
            [str(p) for p in Path(a.corpus).rglob("*.md")]
    tok.train(files, trainers.BpeTrainer(
        vocab_size=a.vocab, special_tokens=["<unk>", "<|eot|>", "<|user|>",
                                            "<|assistant|>"], show_progress=False))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tok.save(str(Path(a.out) / "tokenizer.json"))
    print(f"[tok] vocab={tok.get_vocab_size()} -> {a.out}")


if __name__ == "__main__":
    main()
