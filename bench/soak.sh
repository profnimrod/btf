#!/usr/bin/env bash
# Lab E: sustained-throughput soak. The first hour is the truth (Ch. 19):
# a run that starts at 15 tok/s and sags to 7 is thermal throttling.
set -euo pipefail
MODEL=""; MINUTES=1; LOG="soak/run.csv"; TOK="tok/domain-32k"
while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --minutes) MINUTES="$2"; shift 2;;
    --power) shift 2;;          # accepted for parity with the field command
    --log) LOG="$2"; shift 2;;
    --tok) TOK="$2"; shift 2;;
    *) shift;;
  esac
done
[ -n "$MODEL" ] || { echo "usage: bench/soak.sh --model <ckpt> [--minutes N]"; exit 1; }
mkdir -p "$(dirname "$LOG")"
echo "elapsed_s,tokens,tok_per_s" > "$LOG"
python3 - "$MODEL" "$MINUTES" "$LOG" "$TOK" <<'PY'
import sys, time, torch
sys.path.insert(0, ".")
from serve.generate import load
from serve.template import render_serving
ckpt, minutes, log, tokdir = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
m, tok = load(ckpt, tokdir)
prompt = render_serving({"messages": [{"role": "user",
         "content": "Summarise the interference response procedure."}]})
ids = torch.tensor([tok.encode(prompt).ids])
t0, total = time.time(), 0
with open(log, "a") as f:
    while time.time() - t0 < minutes * 60:
        s = time.time()
        out = m.generate(ids, max_new=16)
        total += 16
        el = time.time() - t0
        f.write(f"{el:.1f},{total},{16/max(time.time()-s,1e-9):.2f}\n"); f.flush()
print(f"[soak] {total} tokens in {time.time()-t0:.1f}s "
      f"= {total/(time.time()-t0):.2f} tok/s sustained -> {log}")
PY
