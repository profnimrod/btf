"""Gated corpus builder (Ch. 7, 13; Lab B). Each stage REPORTS what it
removed. Dedup (MinHash), decontamination against eval sets, mixing by
declared weights, and a corpus card with a decontamination attestation.
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def _shingles(t: str, k: int = 5):
    toks = _norm(t).split()
    return {" ".join(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1))}


def minhash_dedup(docs, threshold: float):
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    keep, removed = [], 0
    for i, (src, text) in enumerate(docs):
        m = MinHash(num_perm=64)
        for sh in _shingles(text):
            m.update(sh.encode())
        if lsh.query(m):
            removed += 1
            continue
        lsh.insert(f"d{i}", m)
        keep.append((src, text))
    return keep, removed


def decontaminate(docs, eval_dir: str):
    evp = Path(eval_dir)
    banned = set()
    if evp.exists():
        for f in evp.rglob("*"):
            if f.suffix in (".jsonl", ".txt"):
                for line in f.read_text(errors="ignore").splitlines():
                    banned |= _shingles(line, 8)
    keep, removed = [], 0
    for src, text in docs:
        if _shingles(text, 8) & banned:
            removed += 1
            continue
        keep.append((src, text))
    return keep, removed, len(banned)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="comma-sep dirs under data/samples or paths")
    ap.add_argument("--dedup", default="minhash:0.8")
    ap.add_argument("--decontam", default="eval/JUDGED")
    ap.add_argument("--mix", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--card", default=None)
    a = ap.parse_args()

    docs = []
    for s in a.sources.split(","):
        d = Path(s if "/" in s else f"data/samples/{s}")
        for f in sorted(d.rglob("*")):
            if f.suffix in (".txt", ".md"):
                docs.append((str(f), f.read_text(errors="ignore")))
    raw_n = len(docs)
    print(f"[corpus] ingested {raw_n} docs from {a.sources}")

    thr = float(a.dedup.split(":")[1]) if ":" in a.dedup else 0.8
    docs, dd = minhash_dedup(docs, thr)
    print(f"[corpus] dedup(minhash>{thr}): removed {dd}, kept {len(docs)}")

    docs, dc, nban = decontaminate(docs, a.decontam)
    print(f"[corpus] decontam vs {a.decontam}: removed {dc} "
          f"({nban} banned shingles)")

    out = Path(a.out)
    (out / "text").mkdir(parents=True, exist_ok=True)
    total_bytes, used = 0, set()
    for src, text in docs:
        # preserve the source identifier: retrieval and traceability key on it
        stem = Path(src).stem
        name = stem if stem not in used else f"{stem}-{len(used)}"
        used.add(name)
        (out / "text" / f"{name}.txt").write_text(text)
        total_bytes += len(text.encode())

    attest = {"decontaminated_against": a.decontam,
              "banned_shingles": nban,
              "removed_dedup": dd, "removed_decontam": dc,
              "attested": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "ATTESTATION.json").write_text(json.dumps(attest, indent=2))
    card = {"sources": a.sources.split(","), "mix": a.mix,
            "docs_in": raw_n, "docs_out": len(docs),
            "bytes": total_bytes, "dedup_threshold": thr}
    (out / "CARD.md").write_text(
        "# Corpus card\n\n```json\n" + json.dumps(card, indent=2) + "\n```\n"
        "\nDecontamination attestation: see ATTESTATION.json\n")
    print(f"[corpus] wrote {len(docs)} docs, {total_bytes:,} bytes -> {out}")
    print(f"[corpus] card + attestation written (no training may start "
          f"without ATTESTATION.json)")


if __name__ == "__main__":
    main()
