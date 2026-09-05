"""Checkpoint selection protocol (Ch. 11): rank candidates by suite score,
print the transcripts an SME must spot-read before the winner is published."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="eval/reports")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--metric", default="overall")
    a = ap.parse_args()
    SUITES = {"standards_qa", "traceability", "telemetry_narration",
              "report_drafting", "link_budget"}
    rows = []
    for f in sorted(Path(a.reports).glob("*.json")):
        d = json.loads(f.read_text())
        if not SUITES.issubset(d.keys()):
            continue          # not a model scorecard (retrieval/bins reports)
        score = d.get(a.metric)
        if score is None:
            vals = [v for v in d.values() if isinstance(v, (int, float))]
            score = sum(vals) / len(vals) if vals else 0
        rows.append((score, f.stem, d))
    rows.sort(reverse=True)
    print(f"[select] {len(rows)} candidates ranked by {a.metric}")
    for i, (s, name, d) in enumerate(rows[:a.top], 1):
        print(f"  {i}. {name:28s} {s:.3f}")
    if rows:
        print(f"\n[select] finalist: {rows[0][1]}")
        print("[select] SME spot-read required on 5 transcripts before publish "
              "(Ch. 11); record seed, data version, and config hash in the registry.")


if __name__ == "__main__":
    main()
