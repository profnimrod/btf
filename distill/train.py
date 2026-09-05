"""Lab E: sequence-level distillation into a smaller student (Ch. 14)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.model import Model, ModelConfig  # noqa: E402
from train.sft_local import build_examples, batches  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=None, help="for the KD reference (optional)")
    ap.add_argument("--student-layers", type=int, default=2)
    ap.add_argument("--student-dim", type=int, default=96)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-kd", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.tok) / "tokenizer.json"))
    cfg = ModelConfig(vocab=max(1400, tok.get_vocab_size()), d_model=a.student_dim,
                      n_layers=a.student_layers, n_heads=4, n_kv_heads=2,
                      d_head=24, d_ff=4 * a.student_dim, context=max(256, a.seq))
    student = Model(cfg)
    print(f"[distill] student {cfg.describe()} "
          f"({student.param_count()/1e6:.2f}M params)")
    rows = [json.loads(l) for l in Path(a.data).read_text(encoding='utf-8').splitlines() if l.strip()]
    ex = build_examples(rows, tok, a.seq)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    student.train()
    for ep in range(a.epochs):
        tot, n = 0.0, 0
        for X, Y in batches(ex, a.seq, 8, rng):
            _, loss = student(X, Y)            # sequence-level KD on teacher text
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f"  [distill] epoch {ep+1}: loss {tot/max(n,1):.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": student.state_dict(), "cfg": cfg.__dict__}, a.out)
    print(f"[distill] saved -> {a.out}")


if __name__ == "__main__":
    main()
