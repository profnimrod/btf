"""Task-suite harness (Ch. 15): scores a checkpoint on the five suites and
writes a report the delta/gate tooling consumes."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.generate import load, answer  # noqa: E402
from eval.link_budget import score as lb_score  # noqa: E402

SUITES = ["standards_qa", "traceability", "telemetry_narration",
          "report_drafting", "link_budget"]


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def suite_of(doc_id):
    return {"STD": "standards_qa", "REQ": "traceability", "TLM": "telemetry_narration",
            "PROC": "report_drafting", "TKT": "report_drafting",
            "LNK": "link_budget"}.get(doc_id.split("-")[0], "standards_qa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--queries", default="eval/JUDGED-v1/queries.jsonl")
    ap.add_argument("--lb-prompts", default="data/lb-prompts.jsonl")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--lb-limit", type=int, default=10)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="eval/reports")
    a = ap.parse_args()
    m, tok = load(a.ckpt, a.tok)
    rows = [json.loads(l) for l in Path(a.queries).read_text(encoding='utf-8').splitlines() if l.strip()]
    hits = {s: [] for s in SUITES}
    for r in rows[:a.limit]:
        gen = answer(m, tok, r["query"], max_new=32)
        ok = 1 if norm(r["answer"])[:8] and norm(r["answer"])[:8] in norm(gen) else 0
        hits[suite_of(r["gold"])].append(ok)
    lb = [json.loads(l) for l in Path(a.lb_prompts).read_text(encoding='utf-8').splitlines() if l.strip()]
    for s in lb[:a.lb_limit]:
        sc = s["scenario"]
        q = (f"Recommend a MODCOD and report margin. EIRP {sc['eirp_dbw']} dBW, "
             f"range {sc['range_km']} km, freq {sc['freq_ghz']} GHz, atmos "
             f"{sc['atmos_db']} dB, G/T {sc['g_t_dbk']} dB/K, symbol rate "
             f"{sc['symbol_rate_hz']/1e6:.0f} Msps.")
        hits["link_budget"].append(1 if lb_score(sc, answer(m, tok, q, 28)).parsed else 0)
    report, per_item = {}, []
    for s in SUITES:
        v = hits[s]
        report[s] = round(100.0 * sum(v) / len(v), 2) if v else 0.0
        per_item += v
    report["overall"] = round(sum(report[s] for s in SUITES) / len(SUITES), 2)
    report["per_item"] = per_item
    name = a.name or Path(a.ckpt).stem
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{name}.json").write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"[harness] {name}: " +
          "  ".join(f"{s}={report[s]:.1f}" for s in SUITES) +
          f"  overall={report['overall']:.1f}")


if __name__ == "__main__":
    main()
