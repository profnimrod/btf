"""Mine (query, positive) training pairs from corpus structure (Ch. 9)."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path


def mine(src: str, qa_history: str | None = None):
    pairs = []
    # operational Q&A / ticket history: the extractor that teaches the model
    # what real user questions look like (Ch. 9)
    if qa_history and Path(qa_history).exists():
        docs = {p.stem: p.read_text() for p in Path(src).rglob("*.txt")}
        for line in Path(qa_history).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            body = docs.get(r["doc_id"])
            if not body:
                continue
            blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
            title = blocks[0].split("\n")[0]
            best = max(blocks[1:], key=len) if len(blocks) > 1 else blocks[0]
            pairs.append({"query": r["query"], "positive": f"{title} :: {best}",
                          "doc_id": r["doc_id"], "source": "qa-history"})
    for p in sorted(Path(src).rglob("*.txt")):
        text, doc = p.read_text(), p.stem
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if not blocks:
            continue
        title = blocks[0].split("\n")[0]
        # heading -> body (skip boilerplate-only blocks)
        for b in blocks[1:]:
            if len(b.split()) < 12:
                continue
            pairs.append({"query": title, "positive": f"{title} :: {b}",
                          "doc_id": doc, "source": "heading-body"})
        # identifier -> full text
        pairs.append({"query": doc, "positive": f"{title} :: {blocks[0]}",
                      "doc_id": doc, "source": "identifier"})
        # ticket symptom -> resolution
        m = re.search(r"Resolution\.(.+?)(?:\n\n|$)", text, re.S)
        if m:
            pairs.append({"query": title, "positive": m.group(1).strip(),
                          "doc_id": doc, "source": "ticket-resolution"})
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus/v1/text")
    ap.add_argument("--extractors", default="all")
    ap.add_argument("--qa-history", default="data/qa-history.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ps = mine(a.corpus, a.qa_history)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for p in ps:
            f.write(json.dumps(p) + "\n")
    from collections import Counter
    print(f"[mine] {len(ps)} pairs  {dict(Counter(p['source'] for p in ps))}")
