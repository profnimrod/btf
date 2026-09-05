"""Exact and near-duplicate removal over mined/synthetic pairs (Ch. 13)."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path


def key(t):
    return hashlib.sha1(re.sub(r"\s+", " ", t.lower()).strip().encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--semantic", type=float, default=0.92)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen, rows, dropped = set(), [], 0
    from datasketch import MinHash, MinHashLSH
    perm = 256 if a.semantic >= 0.9 else 64
    lsh = MinHashLSH(threshold=a.semantic, num_perm=perm)
    for path in a.inputs:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = key(r["query"] + r["positive"])
            if k in seen:
                dropped += 1
                continue
            mh = MinHash(num_perm=perm)
            for w in set(re.sub(r"\s+", " ", r["positive"].lower()).split()):
                mh.update(w.encode())
            if lsh.query(mh):
                dropped += 1
                continue
            lsh.insert(k, mh)
            seen.add(k)
            rows.append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[dedup] kept {len(rows)}, dropped {dropped}")


if __name__ == "__main__":
    main()
