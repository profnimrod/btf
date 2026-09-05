"""Template-based synthetic query generation with decontamination against
the judged set (Chs. 9, 13). The gate, not the generator, is the lesson."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

TEMPLATES = [
    "what does {id} specify",
    "summarise the requirement in {id}",
    "steps for {topic}",
    "how is {topic} handled",
    "which document covers {topic}",
]


def norm(t):
    return re.sub(r"\s+", " ", t.lower()).strip()


def shingles(t, k=8):
    w = norm(t).split()
    return {" ".join(w[i:i + k]) for i in range(max(0, len(w) - k + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/v1/text")
    ap.add_argument("--template", default="grounded_query_v2")
    ap.add_argument("--quota-grid", action="store_true")
    ap.add_argument("--decontam", default="eval/JUDGED-v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    banned = set()
    jd = Path(a.decontam)
    if jd.exists():
        for f in jd.rglob("*.jsonl"):
            for line in f.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    banned |= shingles(json.loads(line)["query"], 6)

    rows, dropped = [], 0
    for p in sorted(Path(a.corpus).rglob("*.txt")):
        text = p.read_text(encoding='utf-8')
        first = text.split("\n")[0]
        ident = first.split(" ")[0]
        topic = first.split("—")[-1].strip() if "—" in first else first
        body = re.split(r"\n\s*\n", text)[1] if "\n\n" in text else text
        for t in TEMPLATES:
            q = t.format(id=ident, topic=topic)
            if shingles(q, 6) & banned:
                dropped += 1
                continue
            rows.append({"query": q, "positive": f"{first} :: {body.strip()}",
                         "doc_id": p.stem, "source": "synthetic"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    Path("data/ATTESTATION-synth.json").write_text(json.dumps(
        {"decontaminated_against": a.decontam, "dropped": dropped,
         "kept": len(rows)}, indent=2))
    print(f"[synth] kept {len(rows)}, dropped {dropped} on decontamination")


if __name__ == "__main__":
    main()
