"""Lab B resume drill: prove checkpoint/resume determinism.

Loads a checkpoint, replays the next K steps against the same data, and
compares per-step loss and batch hashes to the original training log —
bitwise data order, matching losses (Chs. 5, 7, 21).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.model import Model, ModelConfig                      # noqa: E402
from train.pretrain import Windows, batch_hash, get_batch, cosine_lr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--log", default="logs/train.jsonl")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import yaml
    tr = yaml.safe_load(open(a.config)).get("train", {})
    state = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg = ModelConfig(**state["cfg"])
    torch.set_num_threads(1)
    model = Model(cfg).to(a.device)
    model.load_state_dict(state["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=tr.get("lr", 3e-4),
                            betas=tuple(tr.get("betas", (0.9, 0.95))),
                            weight_decay=tr.get("weight_decay", 0.1))
    opt.load_state_dict(state["opt"])
    step0 = state["step"]

    meta = json.loads((Path(a.data) / "meta.json").read_text())
    data = np.memmap(Path(a.data) / "train.bin",
                     dtype=np.dtype(meta["dtype"]), mode="r")
    seq = a.seq_len or tr.get("seq_len", cfg.context)
    micro = a.micro_batch or tr.get("micro_batch", 8)
    accum = a.grad_accum or tr.get("grad_accum", 1)
    win = Windows(len(data), seq, micro * accum, tr.get("seed", 1337))

    ref = {}
    for line in Path(a.log).read_text().splitlines():
        r = json.loads(line)
        ref[r["step"]] = r
    total_ref = max(ref)

    ok = True
    step = step0
    for _ in range(a.steps):
        starts = win.batch_starts(step)
        bh = batch_hash(starts)
        lr = cosine_lr(step, tr.get("lr", 3e-4),
                       min(tr.get("warmup_steps", 2000), max(1, total_ref // 10)),
                       total_ref)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for i in range(accum):
            xb, yb = get_batch(data, starts[i * micro:(i + 1) * micro], seq, a.device)
            _, loss = model(xb, yb)
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("grad_clip", 1.0))
        opt.step()
        step += 1
        want = ref.get(step)
        if want is None:
            print(f"step {step}: no reference row"); ok = False; continue
        data_ok = want["batch_sha1"] == bh
        loss_ok = abs(want["loss"] - loss_acc) < 1e-6
        print(f"step {step}: data {'OK ' if data_ok else 'FAIL'} "
              f"loss {'OK' if loss_ok else 'FAIL'} "
              f"(replay {loss_acc:.6f} vs log {want['loss']:.6f})")
        ok = ok and data_ok and loss_ok
    print("RESUME DRILL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
