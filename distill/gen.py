"""Lab E: teacher generation for distillation, filtered by the checker and
sequenced routine-tier first (Chs. 13, 14)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.generate import load, answer  # noqa: E402
from eval.link_budget import score as lb_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--queries", default="eval/JUDGED-v1/queries.jsonl")
    ap.add_argument("--lb-prompts", default="data/lb-prompts.jsonl")
    ap.add_argument("--grid", default="full")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--filter", default="checker")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m, tok = load(a.teacher, a.tok)
    rows, kept, dropped = [], 0, 0
    src = [json.loads(l) for l in Path(a.queries).read_text().splitlines() if l.strip()]
    for r in src[:a.n]:
        gen = answer(m, tok, r["query"], max_new=32, temperature=a.temperature)
        if len(gen.split()) < 3:
            dropped += 1
            continue
        rows.append({"messages": [{"role": "user", "content": r["query"]},
                                  {"role": "assistant", "content": gen}],
                     "tier": "routine"})
        kept += 1
    lb = [json.loads(l) for l in Path(a.lb_prompts).read_text().splitlines() if l.strip()]
    for s in lb:
        sc = s["scenario"]
        q = (f"Recommend a MODCOD and report margin. EIRP {sc['eirp_dbw']} dBW, "
             f"range {sc['range_km']} km, freq {sc['freq_ghz']} GHz, atmos "
             f"{sc['atmos_db']} dB, G/T {sc['g_t_dbk']} dB/K, symbol rate "
             f"{sc['symbol_rate_hz']/1e6:.0f} Msps.")
        gen = answer(m, tok, q, max_new=28, temperature=a.temperature)
        rew = lb_score(sc, gen)
        if "checker" in a.filter and rew.value <= 0:
            dropped += 1                      # the checker rejects it: drop it
            continue
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": gen}],
                     "tier": "edge", "reward": rew.value})
        kept += 1
    rows.sort(key=lambda r: 0 if r["tier"] == "routine" else 1)  # curriculum
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[distill-gen] kept {kept}, dropped {dropped} on filter "
          f"'{a.filter}' -> {a.out} (routine tier first)")


if __name__ == "__main__":
    main()
