"""Lab C (small-scale): direct preference optimization on the lab model
(Ch. 12). Reference policy is a frozen copy of the SFT checkpoint; the
beta sweep is the lab's leash-tension exercise."""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.generate import load  # noqa: E402
from serve.template import render_serving  # noqa: E402
from train.lora import adapter_state, trainable  # noqa: E402


def seq_logprob(model, tok, prompt, response, max_len=192):
    p_ids = tok.encode(prompt).ids
    r_ids = tok.encode(response).ids
    ids = (p_ids + r_ids)[:max_len]
    if len(ids) < 2:
        return None
    x = torch.tensor([ids[:-1]])
    y = torch.tensor([ids[1:]])
    logits, _ = model(x)
    lp = F.log_softmax(logits[0], dim=-1)
    tgt = y[0]
    mask = torch.zeros_like(tgt, dtype=torch.bool)
    mask[max(0, len(p_ids) - 1):] = True          # score the response only
    sel = lp[torch.arange(len(tgt)), tgt]
    return (sel * mask).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SFT checkpoint or adapter")
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--pairs", default="data/pref-v1/train.jsonl")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    policy, tok = load(a.base, a.tok)
    ref, _ = load(a.base, a.tok)
    for p in ref.parameters():
        p.requires_grad = False
    params = trainable(policy) or list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr)

    rows = [json.loads(l) for l in Path(a.pairs).read_text().splitlines() if l.strip()]
    rows = rows[:a.limit]
    print(f"[dpo] beta={a.beta} pairs={len(rows)}")
    for ep in range(a.epochs):
        tot, acc, n = 0.0, 0, 0
        for r in rows:
            prompt = render_serving({"messages": [{"role": "user",
                                                   "content": r["prompt"]}]})
            pw = seq_logprob(policy, tok, prompt, r["chosen"])
            pl = seq_logprob(policy, tok, prompt, r["rejected"])
            if pw is None or pl is None:
                continue
            with torch.no_grad():
                rw = seq_logprob(ref, tok, prompt, r["chosen"])
                rl = seq_logprob(ref, tok, prompt, r["rejected"])
            logits = a.beta * ((pw - pl) - (rw - rl))
            loss = -F.logsigmoid(logits)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); acc += int(logits.item() > 0); n += 1
        print(f"  [dpo] epoch {ep + 1}: loss {tot / max(n,1):.4f} "
              f"margin>0 on {acc}/{n} pairs")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    st = torch.load(a.base, map_location="cpu", weights_only=False)
    if "adapter" in st:
        torch.save({"adapter": adapter_state(policy), "base": st["base"],
                    "cfg": st["cfg"], "lora_r": st.get("lora_r", 8)}, a.out)
    else:
        torch.save({"model": policy.state_dict(), "cfg": st["cfg"]}, a.out)
    print(f"[dpo] saved -> {a.out}")


if __name__ == "__main__":
    main()
