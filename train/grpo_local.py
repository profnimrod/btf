"""Lab C (small-scale): GRPO against the deterministic link-budget checker
(Ch. 12). Group-relative advantage: A_i = r_i - mean(r_group). The point of
the lab is the mechanism and the gaming audit, not the leaderboard."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.generate import load  # noqa: E402
from serve.template import render_serving  # noqa: E402
from train.lora import adapter_state, trainable  # noqa: E402
from eval.link_budget import score  # noqa: E402


def sample(model, tok, prompt, max_new=24, temperature=1.0):
    ids = torch.tensor([tok.encode(prompt).ids])
    out = ids
    logps = []
    for _ in range(max_new):
        logits, _ = model(out[:, -model.cfg.context:])
        step = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(step, dim=-1)
        nxt = torch.multinomial(probs, 1)
        logps.append(F.log_softmax(step, dim=-1)[0, nxt[0, 0]])
        out = torch.cat([out, nxt], dim=1)
    text = tok.decode(out[0].tolist())
    gen = text[len(tok.decode(ids[0].tolist())):]
    return gen, torch.stack(logps).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--prompts", default="data/lb-prompts.jsonl")
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--audit", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    policy, tok = load(a.base, a.tok)
    params = trainable(policy) or list(policy.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr)
    scen = [json.loads(l) for l in Path(a.prompts).read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(0)
    history, transcripts = [], []

    for it in range(a.iters):
        s = scen[rng.integers(len(scen))]
        sc = s["scenario"]
        q = (f"Recommend a MODCOD and report margin. EIRP {sc['eirp_dbw']} dBW, "
             f"range {sc['range_km']} km, freq {sc['freq_ghz']} GHz, atmos "
             f"{sc['atmos_db']} dB, G/T {sc['g_t_dbk']} dB/K, symbol rate "
             f"{sc['symbol_rate_hz']/1e6:.0f} Msps.")
        prompt = render_serving({"messages": [{"role": "user", "content": q}]})
        gens, logps, rewards = [], [], []
        for _ in range(a.group):
            g, lp = sample(policy, tok, prompt, temperature=a.temperature)
            r = score(sc, g)
            gens.append(g); logps.append(lp); rewards.append(r.value)
        R = torch.tensor(rewards, dtype=torch.float32)
        adv = (R - R.mean()) / (R.std() + 1e-6)   # group-relative, std-normalized (App. A)
        if adv.abs().sum() > 0:
            loss = -(torch.stack(logps) * adv).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        history.append(float(R.mean()))
        transcripts += [(rewards[i], gens[i]) for i in range(len(gens))]
        if (it + 1) % 10 == 0:
            print(f"  [grpo] iter {it+1}: mean reward "
                  f"{np.mean(history[-10:]):.4f}")

    print(f"[grpo] first-quarter mean {np.mean(history[:max(1,len(history)//4)]):.4f} "
          f"-> last-quarter mean {np.mean(history[-max(1,len(history)//4):]):.4f}")
    print(f"\n[grpo] GAMING AUDIT — {a.audit} highest-reward transcripts "
          f"(read these by hand; Ch. 12):")
    for r, g in sorted(transcripts, key=lambda x: -x[0])[:a.audit]:
        print(f"   reward {r:.2f} :: {g[:90]!r}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    st = torch.load(a.base, map_location="cpu", weights_only=False)
    if "adapter" in st:
        torch.save({"adapter": adapter_state(policy), "base": st["base"],
                    "cfg": st["cfg"], "lora_r": st.get("lora_r", 8)}, a.out)
    else:
        torch.save({"model": policy.state_dict(), "cfg": st["cfg"]}, a.out)
    print(f"[grpo] saved -> {a.out}")


if __name__ == "__main__":
    main()
