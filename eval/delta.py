"""Lab C/E: score parent vs child checkpoints per task suite with gates
(Ch. 14, 15). --plan documents the suites and gate thresholds; scoring
reads eval reports (JSON) when present."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

GATES = {"standards_qa": 1.0, "telemetry_narration": 1.5,
         "link_budget": 1.0, "report_drafting": 1.0, "traceability": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--children", default="")
    ap.add_argument("--suites", default="all")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--reports", default="eval/reports")
    a = ap.parse_args()
    rep = Path(a.reports)
    print(f"[delta] parent={a.parent} gates={'on' if a.gates else 'off'}")
    print(f"[delta] suites: {list(GATES)}")
    print(f"[delta] gate thresholds (pp): {GATES}")
    pf = rep / f"{Path(a.parent).name}.json"
    if not pf.exists():
        print(f"[delta] no report for parent at {pf}; run the harness first.")
        print("[delta] (this is the plan view — real deltas need eval reports)")
        return
    parent = json.loads(pf.read_text())
    any_fail = False
    for child in filter(None, a.children.split(",")):
        cf = rep / f"{Path(child).name}.json"
        if not cf.exists():
            print(f"[delta] missing report {cf}"); continue
        c = json.loads(cf.read_text())
        print(f"\n  {child}")
        for s in GATES:
            d = c.get(s, 0) - parent.get(s, 0)
            bad = a.gates and (-d) > GATES[s]
            any_fail = any_fail or bad
            print(f"    {s:20s} {d:+5.1f}pp  {'GATE FAIL' if bad else 'ok'}")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
