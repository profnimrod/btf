"""Content-addressed model/dataset registry with Ed25519 signing (Ch. 3,
19, 21, App. F). SHA-256 CAS + a JSON manifest store; fetch --verify
refuses on hash or signature mismatch; bom walks the parent graph.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path

ROOT = Path("registry/store")
OBJ, MAN, KEYS = ROOT / "objects", ROOT / "manifests", ROOT / "keys"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _keypair():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    KEYS.mkdir(parents=True, exist_ok=True)
    sk_p, pk_p = KEYS / "root_ed25519.sk", KEYS / "root_ed25519.pk"
    if not sk_p.exists():
        sk = Ed25519PrivateKey.generate()
        sk_p.write_bytes(sk.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
            serialization.NoEncryption()))
        pk_p.write_bytes(sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw))
        print(f"[registry] generated offline root key -> {sk_p}")
    return sk_p, pk_p


def _sign(payload: bytes) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    sk_p, _ = _keypair()
    sk = Ed25519PrivateKey.from_private_bytes(sk_p.read_bytes())
    return sk.sign(payload).hex()


def _verify(payload: bytes, sig_hex: str) -> bool:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    _, pk_p = _keypair()
    pk = Ed25519PublicKey.from_public_bytes(pk_p.read_bytes())
    try:
        pk.verify(bytes.fromhex(sig_hex), payload)
        return True
    except InvalidSignature:
        return False


def _canonical(man: dict) -> bytes:
    m = {k: v for k, v in man.items() if k != "signature"}
    return json.dumps(m, sort_keys=True, separators=(",", ":")).encode()


def publish(artifact: str, atype: str, parents, datasets, evals=None,
            card=None, markings=None, license_="open"):
    OBJ.mkdir(parents=True, exist_ok=True)
    MAN.mkdir(parents=True, exist_ok=True)
    src = Path(artifact)
    digest = sha256_file(src)
    blob = OBJ / digest
    if not blob.exists():
        shutil.copyfile(src, blob)
    man = {"name": src.name, "type": atype, "sha256": digest,
           "parents": list(parents or []), "datasets": list(datasets or []),
           "evals": evals or {}, "card": card, "markings": markings or [],
           "license": license_, "size": src.stat().st_size,
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    man["signature"] = _sign(_canonical(man))
    (MAN / f"{digest}.json").write_text(json.dumps(man, indent=2))
    print(f"[registry] published {src.name} type={atype} sha256={digest[:16]}…")
    return digest


def fetch(digest: str, out: str | None = None, verify: bool = True) -> Path:
    man_p = MAN / f"{digest}.json"
    if not man_p.exists():
        sys.exit(f"[registry] no manifest for {digest[:16]}…")
    man = json.loads(man_p.read_text())
    blob = OBJ / digest
    if verify:
        if sha256_file(blob) != digest:
            sys.exit("[registry] FAIL: content hash mismatch (tamper)")
        if not _verify(_canonical(man), man["signature"]):
            sys.exit("[registry] FAIL: signature invalid")
        print(f"[registry] verified {man['name']} ({digest[:16]}…)")
    dst = Path(out) if out else blob
    if out:
        shutil.copyfile(blob, dst)
    return dst


def bom(digest: str, depth=0, seen=None):
    seen = seen or set()
    man = json.loads((MAN / f"{digest}.json").read_text())
    print("  " * depth + f"- {man['name']} [{man['type']}] {digest[:12]}… "
          f"license={man['license']} datasets={man['datasets']}")
    for p in man["parents"]:
        if p not in seen:
            seen.add(p)
            bom(p, depth + 1, seen)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("publish")
    p.add_argument("--artifact", required=True)
    p.add_argument("--type", default="model")
    p.add_argument("--parents", nargs="*", default=[])
    p.add_argument("--datasets", nargs="*", default=[])
    p.add_argument("--card", default=None)
    p.add_argument("--markings", nargs="*", default=[])
    p.add_argument("--license", default="open")
    f = sub.add_parser("fetch")
    f.add_argument("--digest", required=True)
    f.add_argument("--out", default=None)
    f.add_argument("--verify", action="store_true")
    f.add_argument("--no-verify", dest="verify", action="store_false")
    f.set_defaults(verify=True)
    b = sub.add_parser("bom")
    b.add_argument("--digest", required=True)
    a = ap.parse_args()
    if a.cmd == "publish":
        print(publish(a.artifact, a.type, a.parents, a.datasets,
                      card=a.card, markings=a.markings, license_=a.license))
    elif a.cmd == "fetch":
        fetch(a.digest, a.out, a.verify)
    elif a.cmd == "bom":
        bom(a.digest)


if __name__ == "__main__":
    main()
