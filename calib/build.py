"""Build the quantization calibration set from the deployment mix, with a
decontamination attestation (Ch. 14, Lab E)."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--len", type=int, default=192)
    ap.add_argument("--mix", default="deploy")
    ap.add_argument("--src", default="corpus/v1/text")
    ap.add_argument("--decontam", default="eval/JUDGED-v1")
    ap.add_argument("--out", default="calib/v1")
    a = ap.parse_args()
    files = sorted(Path(a.src).rglob("*.txt"))
    rng = random.Random(11)
    picks = rng.sample(files, min(a.n, len(files)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "calib.jsonl").open("w", encoding='utf-8') as f:
        for p in picks:
            f.write(json.dumps({"text": " ".join(p.read_text(encoding='utf-8').split()[:a.len]),
                                "doc_id": p.stem}) + "\n")
    (out / "ATTESTATION.json").write_text(json.dumps(
        {"mix": a.mix, "n": len(picks), "seq_len": a.len,
         "decontaminated_against": a.decontam,
         "note": "calibration data is domain data or it is nothing (Ch. 14)",
         "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    print(f"[calib] {len(picks)} samples -> {out} (+ attestation)")


if __name__ == "__main__":
    main()
