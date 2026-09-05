"""Lab B pretraining loop: AdamW, cosine schedule, deterministic
stateless batching, checkpoint/resume, canary heartbeat (Chs. 5, 7).

Book command (Lab B):
  torchrun --nproc_per_node=1 train/pretrain.py \
      --config train/cfg-125m.yaml --data corpus/v1 \
      --tokens 5e9 --bf16 --act-ckpt --ckpt-every 30min \
      --eval-heartbeat eval/canary-prompts.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.model import Model, ModelConfig  # noqa: E402


def parse_every(s: str):
    s = s.strip().lower()
    if s.endswith("min"):
        return ("min", float(s[:-3]))
    if s.endswith("steps"):
        return ("steps", int(s[:-5]))
    return ("steps", int(s))


def cosine_lr(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


class Windows:
    """Deterministic step -> token-window mapping (resume-safe by design)."""

    def __init__(self, n_tokens: int, seq: int, batch: int, seed: int):
        self.seq, self.batch, self.seed = seq, batch, seed
        self.starts = np.arange(0, n_tokens - seq - 1, seq, dtype=np.int64)
        self.per_epoch = max(1, len(self.starts) // batch)

    def batch_starts(self, step: int) -> np.ndarray:
        epoch, k = divmod(step, self.per_epoch)
        rng = np.random.default_rng(self.seed + epoch)
        perm = rng.permutation(self.starts)
        return perm[k * self.batch:(k + 1) * self.batch]


def batch_hash(starts: np.ndarray) -> str:
    return hashlib.sha1(starts.tobytes()).hexdigest()[:12]


def get_batch(data: np.memmap, starts: np.ndarray, seq: int, device):
    xs = np.stack([data[s:s + seq] for s in starts]).astype(np.int64)
    ys = np.stack([data[s + 1:s + seq + 1] for s in starts]).astype(np.int64)
    return (torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device))


def save_ckpt(dirp: Path, model, opt, step, cfg):
    dirp.mkdir(parents=True, exist_ok=True)
    path = dirp / f"step{step:08d}.pt"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "cfg": cfg.__dict__,
                "torch_rng": torch.get_rng_state()}, path)
    shutil.copyfile(path, dirp / "latest.pt")
    kept = sorted(dirp.glob("step*.pt"))[:-3]
    for p in kept:
        p.unlink()
    return path


def heartbeat(model, tok_dir: str, prompts_path: str, step: int, log: Path):
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
    except Exception as e:  # pragma: no cover
        log.open("a", encoding='utf-8').write(f"[step {step}] heartbeat skipped: {e}\n")
        return
    lines = []
    for row in Path(prompts_path).read_text(encoding='utf-8').splitlines():
        if not row.strip():
            continue
        prompt = json.loads(row)["prompt"]
        ids = torch.tensor([tok.encode(prompt).ids])
        out = model.generate(ids, max_new=24)[0].tolist()
        lines.append(f"[step {step}] {prompt!r} -> {tok.decode(out)!r}")
    log.parent.mkdir(exist_ok=True)
    with log.open("a", encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokens", type=float, default=None,
                    help="training budget in tokens, e.g. 5e9")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--act-ckpt", action="store_true")
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--quant", choices=["none", "ternary"], default=None)
    ap.add_argument("--ckpt-every", default="500steps")
    ap.add_argument("--ckpt-dir", default="ckpt/run")
    ap.add_argument("--eval-heartbeat", default=None)
    ap.add_argument("--resume", default="auto")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--log", default="logs/train.jsonl")
    a = ap.parse_args(argv)

    raw = yaml.safe_load(open(a.config, encoding='utf-8'))
    tr = raw.get("train", {})
    cfg = ModelConfig.from_yaml(a.config)
    if a.quant:
        cfg.quant = a.quant
    seq = a.seq_len or tr.get("seq_len", cfg.context)
    cfg.context = max(cfg.context, seq)
    micro = a.micro_batch or tr.get("micro_batch", 8)
    accum = a.grad_accum or tr.get("grad_accum", 1)
    seed = tr.get("seed", 1337)

    torch.manual_seed(seed)
    if a.device == "cpu":
        torch.set_num_threads(1)  # bitwise-stable CPU runs (verify-resume)

    data_dir = Path(a.data)
    if not (data_dir / "train.bin").exists():
        from tok.encode import encode_dir
        print(f"[pretrain] train.bin absent; encoding with {tr.get('tokenizer')}")
        encode_dir(str(data_dir), tr["tokenizer"])
    meta = json.loads((data_dir / "meta.json").read_text(encoding='utf-8'))
    data = np.memmap(data_dir / "train.bin", dtype=np.dtype(meta["dtype"]), mode="r")
    cfg.vocab = max(cfg.vocab, meta["vocab"])

    model = Model(cfg).to(a.device)
    model.grad_checkpoint = a.act_ckpt
    if a.fsdp and int(os.environ.get("WORLD_SIZE", "1")) > 1:  # pragma: no cover
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.get("lr", 3e-4),
                            betas=tuple(tr.get("betas", (0.9, 0.95))),
                            weight_decay=tr.get("weight_decay", 0.1))

    tokens_per_step = micro * accum * seq
    if a.max_steps:
        total_steps = a.max_steps
    elif a.tokens:
        total_steps = int(a.tokens / tokens_per_step)
    else:
        sys.exit("provide --tokens or --max-steps")
    warmup = min(tr.get("warmup_steps", 2000), max(1, total_steps // 10))

    win = Windows(len(data), seq, micro * accum, seed)
    step = 0
    ckdir = Path(a.ckpt_dir)
    if a.resume and (a.resume != "auto" or (ckdir / "latest.pt").exists()):
        path = ckdir / "latest.pt" if a.resume == "auto" else Path(a.resume)
        state = torch.load(path, map_location=a.device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"[pretrain] resumed from {path} at step {step}")

    every_kind, every_val = parse_every(a.ckpt_every)
    log_path = Path(a.log)
    log_path.parent.mkdir(exist_ok=True)
    log = log_path.open("a", encoding='utf-8')
    amp = a.bf16 and a.device == "cuda"
    t_last, tok_count = time.time(), 0
    print(f"[pretrain] {cfg.describe()} | {model.param_count()/1e6:.1f}M params | "
          f"{total_steps} steps x {tokens_per_step} tok")

    last_ck = time.time()
    while step < total_steps:
        starts = win.batch_starts(step)
        bh = batch_hash(starts)
        lr = cosine_lr(step, tr.get("lr", 3e-4), warmup, total_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for i in range(accum):
            xb, yb = get_batch(data, starts[i * micro:(i + 1) * micro], seq, a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                _, loss = model(xb, yb)
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               tr.get("grad_clip", 1.0)).item()
        opt.step()
        step += 1
        tok_count += tokens_per_step
        rec = {"step": step, "loss": round(loss_acc, 6), "lr": round(lr, 8),
               "gnorm": round(gnorm, 4), "batch_sha1": bh}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % max(1, total_steps // 50) == 0 or step <= 3:
            dt = time.time() - t_last
            print(f"  step {step:6d}/{total_steps} loss {loss_acc:7.4f} "
                  f"lr {lr:.2e} gnorm {gnorm:5.2f} "
                  f"{tok_count/max(dt,1e-9):,.0f} tok/s")
            t_last, tok_count = time.time(), 0
        due = (every_kind == "steps" and step % int(every_val) == 0) or \
              (every_kind == "min" and time.time() - last_ck > every_val * 60)
        if due or step == total_steps:
            p = save_ckpt(ckdir, model, opt, step, cfg)
            last_ck = time.time()
            print(f"  [ckpt] {p}")
            if a.eval_heartbeat:
                heartbeat(model, tr["tokenizer"], a.eval_heartbeat, step,
                          Path("logs/heartbeat.txt"))
    print("[pretrain] done.")


if __name__ == "__main__":
    main()
