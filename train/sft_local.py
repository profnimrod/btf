"""Lab C (small-scale): supervised fine-tuning of the lab model on the
instruction set, with template parity, packing, and response-only loss
(Ch. 11). Same invariants as the GPU path, runnable on a laptop."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.model import Model, ModelConfig  # noqa: E402
from train.lora import apply_lora, adapter_state, trainable  # noqa: E402
from serve.template import render_parts  # noqa: E402


def build_examples(rows, tok, max_len):
    """Encode prompt and response separately and concatenate ids, so the
    training prefix is byte-identical to what the server will send."""
    ex = []
    for r in rows:
        prompt, response = render_parts(r)
        p_ids = tok.encode(prompt).ids
        r_ids = tok.encode(response).ids
        if not r_ids or len(p_ids) + len(r_ids) > max_len:
            continue
        ex.append((p_ids, r_ids))
    return ex


def batches(ex, seq, bs, rng):
    idx = rng.permutation(len(ex))
    for s in range(0, len(idx) - bs + 1, bs):
        X, Y = [], []
        for i in idx[s:s + bs]:
            p, r = ex[i]
            ids = (p + r)[:seq]
            ids = ids + [0] * (seq - len(ids))
            tgt = list(ids[1:]) + [0]
            # response-only loss: mask the prompt and the padding
            for j in range(len(tgt)):
                if j < len(p) - 1 or j >= len(p) + len(r) - 1:
                    tgt[j] = -100
            X.append(ids); Y.append(tgt)
        yield torch.tensor(X), torch.tensor(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="pretrained checkpoint (Lab B)")
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--data", default="data/sft-v1/train.jsonl")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--full-finetune", action="store_true")
    ap.add_argument("--oversample", default=None,
                    help="substring:factor, e.g. 'Recommend a MODCOD:6' — "
                         "weight a task family whose format must be solid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.tok) / "tokenizer.json"))
    st = torch.load(a.base, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**st["cfg"])
    cfg.context = max(cfg.context, a.seq)
    model = Model(cfg)
    model.load_state_dict(st["model"], strict=False)
    if not a.full_finetune:
        n = apply_lora(model, r=a.lora_r, alpha=2 * a.lora_r)
        params = trainable(model)
        print(f"[sft] LoRA on {n} projections; "
              f"{sum(p.numel() for p in params):,} trainable params")
    else:
        params = list(model.parameters())
        print(f"[sft] full fine-tune; {sum(p.numel() for p in params):,} params")

    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    if a.oversample:
        sub, factor = a.oversample.rsplit(":", 1)
        extra = [r for r in rows
                 if sub.lower() in r["messages"][0]["content"].lower()]
        rows = rows + extra * (int(factor) - 1)
        print(f"[sft] oversampled {len(extra)} '{sub}' rows x{factor}")
    ex = build_examples(rows, tok, a.seq)
    print(f"[sft] {len(ex)} usable examples of {len(rows)}")
    opt = torch.optim.AdamW(params, lr=a.lr)
    rng = np.random.default_rng(0)
    model.train()
    for ep in range(a.epochs):
        tot, n = 0.0, 0
        for X, Y in batches(ex, a.seq, a.batch, rng):
            _, loss = model(X, Y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f"  [sft] epoch {ep + 1}: loss {tot / max(n, 1):.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.full_finetune:
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, a.out)
    else:
        torch.save({"adapter": adapter_state(model), "base": a.base,
                    "cfg": cfg.__dict__, "lora_r": a.lora_r}, a.out)
        mb = Path(a.out).stat().st_size / 1e6
        print(f"[sft] adapter saved -> {a.out} ({mb:.2f} MB)")


if __name__ == "__main__":
    main()
