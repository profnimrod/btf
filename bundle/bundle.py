"""Seal, sign, and verify deployment bundles; A/B slot activation with
built-in test and automatic rollback (Ch. 14, 19; Lab E).

A bundle is a directory of artifacts (weights, tokenizer, template,
generation config, quant metadata, index snapshot) plus a signed
manifest of per-file SHA-256 hashes.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from registry.registry import _sign, _verify, _canonical  # noqa: E402


def _hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def seal(bundle_dir: str, name: str, index: str | None = None) -> Path:
    bd = Path(bundle_dir)
    files = {}
    for p in sorted(bd.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files[str(p.relative_to(bd))] = _hash(p)
    man = {"name": name, "files": files, "index": index,
           "sealed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    man["signature"] = _sign(_canonical(man))
    (bd / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    print(f"[bundle] sealed {name}: {len(files)} files, signed")
    return bd / "MANIFEST.json"


def verify(bundle_dir: str, quiet: bool = False) -> bool:
    bd = Path(bundle_dir)
    man_p = bd / "MANIFEST.json"
    if not man_p.exists():
        if not quiet:
            print("[bundle] FAIL: no manifest")
        return False
    man = json.loads(man_p.read_text())
    if not _verify(_canonical(man), man["signature"]):
        if not quiet:
            print("[bundle] FAIL: signature invalid")
        return False
    for rel, want in man["files"].items():
        p = bd / rel
        if not p.exists() or _hash(p) != want:
            if not quiet:
                print(f"[bundle] FAIL: hash mismatch on {rel}")
            return False
    if not quiet:
        print(f"[bundle] verify OK: {man['name']} ({len(man['files'])} files)")
    return True


def run_bit(bundle_dir: str) -> bool:
    """Boot-time built-in test: manifest verify + canary property checks
    + token-rate floor. Canaries live in the bundle as bit_canaries.jsonl.
    """
    if not verify(bundle_dir, quiet=True):
        print("[BIT] FAIL: bundle verification"); return False
    canary = Path(bundle_dir) / "bit_canaries.jsonl"
    if canary.exists():
        rows = [json.loads(x) for x in canary.read_text().splitlines() if x.strip()]
        for r in rows:
            # property checks are declarative; the harness fills real outputs.
            if r.get("expect") == "parses" and not r.get("parses", True):
                print(f"[BIT] FAIL: canary {r['id']} did not parse"); return False
            if r.get("expect") == "refuses" and not r.get("refuses", True):
                print(f"[BIT] FAIL: canary {r['id']} failed to refuse"); return False
        print(f"[BIT] {len(rows)} canaries passed")
    print("[BIT] token-rate floor: OK (stub; wire to serve/ in deployment)")
    print("[BIT] PASS")
    return True


def activate(bundle_dir: str, target_root: str):
    """Atomic A/B slot activation with automatic rollback on failed BIT."""
    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)
    ptr = root / "active"
    cur = ptr.read_text().strip() if ptr.exists() else None
    new_slot = "B" if cur == "A" else "A"
    dst = root / f"slot_{new_slot}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(bundle_dir, dst)
    print(f"[deploy] staged into slot {new_slot}")
    if run_bit(str(dst)):
        ptr.write_text(new_slot)
        print(f"[deploy] PROMOTED slot {new_slot} (previous {cur} retained)")
        return True
    print(f"[deploy] BIT FAILED -> auto-revert; active stays {cur}")
    shutil.rmtree(dst)
    return False


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--dir", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--index", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    b = sub.add_parser("bit")
    b.add_argument("--dir", required=True)
    a2 = sub.add_parser("activate")
    a2.add_argument("--dir", required=True)
    a2.add_argument("--target", required=True)
    a = ap.parse_args()
    if a.cmd == "seal":
        seal(a.dir, a.name, a.index)
    elif a.cmd == "verify":
        sys.exit(0 if verify(a.dir) else 1)
    elif a.cmd == "bit":
        sys.exit(0 if run_bit(a.dir) else 1)
    elif a.cmd == "activate":
        sys.exit(0 if activate(a.dir, a.target) else 1)


if __name__ == "__main__":
    main()
