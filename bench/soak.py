"""Lab E: sustained-throughput soak (cross-platform). The first hour is
the truth (Ch. 19): a run that starts at 15 tok/s and sags to 7 is
thermal throttling."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--power", default=None, help="accepted for parity with the field command")
    ap.add_argument("--log", default="soak/run.csv")
    ap.add_argument("--tok", default="tok/domain-32k")
    a = ap.parse_args()
    import torch
    from serve.generate import load
    from serve.template import render_serving
    m, tok = load(a.model, a.tok)
    prompt = render_serving({"messages": [{"role": "user",
              "content": "Summarise the interference response procedure."}]})
    ids = torch.tensor([tok.encode(prompt).ids])
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    t0, total = time.time(), 0
    with open(a.log, "w", encoding="utf-8") as f:
        f.write("elapsed_s,tokens,tok_per_s\n")
        while time.time() - t0 < a.minutes * 60:
            s = time.time()
            m.generate(ids, max_new=16)
            total += 16
            f.write(f"{time.time()-t0:.1f},{total},{16/max(time.time()-s,1e-9):.2f}\n"); f.flush()
    el = max(time.time() - t0, 1e-9)
    print(f"[soak] {total} tokens in {el:.1f}s = {total/el:.2f} tok/s sustained -> {a.log}")


if __name__ == "__main__":
    main()
