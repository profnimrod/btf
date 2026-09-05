"""Lab E: quantization-aware repair (Ch. 14). Trains the bf16 MASTER under
fake quantization of the exact target scheme — it never optimizes packed
low-bit tensors — then saves the repaired master for export/quantization."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.generate import load  # noqa: E402
from quant.quantize import quantize_tensor  # noqa: E402
from train.sft_local import build_examples, batches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=None, help="bf16 master checkpoint (preferred)")
    ap.add_argument("--init", default=None, help="legacy alias for --master")
    ap.add_argument("--fake-quant", default="int4:g128",
                    help="target scheme to simulate, e.g. awq-int4:g128")
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--data", default="data/sft-v1/train.jsonl")
    ap.add_argument("--focus", default="telemetry",
                    help="substring selecting the failing task family")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = a.master or a.init
    if not src:
        sys.exit("[qat] provide --master (bf16 master checkpoint)")
    scheme = a.fake_quant.split(":")
    bits = int(''.join(c for c in scheme[0] if c.isdigit()) or a.bits)
    group = int(scheme[1].lstrip("g")) if len(scheme) > 1 else a.group
    a.bits, a.group = bits, group
    print(f"[qat] master={src} fake-quant=int{bits} group={group}")
    model, tok = load(src, a.tok)
    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    focus = [r for r in rows if a.focus.lower() in json.dumps(r).lower()] or rows
    ex = build_examples(focus, tok, a.seq)
    print(f"[qat] repairing on {len(ex)} '{a.focus}' examples, quantizer live")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    model.train()
    step, tot = 0, 0.0
    while step < a.steps:
        for X, Y in batches(ex, a.seq, 4, rng):
            _, loss = model(X, Y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # re-apply the quantizer after every update (straight-through)
            with torch.no_grad():
                for name, mod in model.named_modules():
                    if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                        mod.weight.data = quantize_tensor(mod.weight.data,
                                                          a.bits, a.group)
            tot += loss.item(); step += 1
            if step >= a.steps:
                break
    print(f"[qat] {step} steps, final loss {tot/max(step,1):.4f}")
    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__,
                "quant": {"scheme": f"qat-int{a.bits}", "group": a.group}}, a.out)
    print(f"[qat] saved -> {a.out}; re-run the FULL delta report and the "
          f"guardrail row before accepting the repair (Ch. 14)")


if __name__ == "__main__":
    main()
