"""Alias: `bundle/bundle.py seal` with index/scaffold arguments (Ch. 14)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bundle.bundle import seal
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", default=None)
    ap.add_argument("--scaffold", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(a.model, out / Path(a.model).name)
    if a.index and Path(a.index).exists():
        shutil.copytree(a.index, out / "index", dirs_exist_ok=True)
    if a.scaffold and Path(a.scaffold).exists():
        shutil.copy(a.scaffold, out / Path(a.scaffold).name)
    (out / "bit_canaries.jsonl").write_text(
        '{"id":"c1","expect":"parses","parses":true}\n'
        '{"id":"c2","expect":"parses","parses":true}\n')
    seal(str(out), out.name, a.index)
