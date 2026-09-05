"""Merge a LoRA adapter into its base and write a single artifact (Ch. 14)."""
import argparse, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.generate import load
from train.lora import merge_lora

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="single adapter checkpoint")
    ap.add_argument("--adapters", default=None, help="ordered stack, e.g. sft,dpo,grpo (last wins; each is bound to the base)")
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = a.adapter or (f"ckpt/{a.adapters.split(',')[-1]}-adapter.pt" if a.adapters else None)
    if not src:
        raise SystemExit("[merge] provide --adapter or --adapters sft,dpo,grpo")
    m, _ = load(src, a.tok)
    n = merge_lora(m)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": m.state_dict(), "cfg": m.cfg.__dict__}, a.out)
    print(f"[merge] folded {n} adapters -> {a.out}")
