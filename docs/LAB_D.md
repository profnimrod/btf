# Lab D — runnable walkthrough

Full method and rationale: **Appendix D** of the book. This file is the
executable companion; every command below runs on a laptop CPU against the
synthetic corpus.

## Prerequisites
```bash
pip install -r env/requirements-cpu.txt
python3 tests/env_check.py --lab D
python3 data/make_synthetic.py --scale small     # once, for all labs
```

## Commands
```bash
python3 pairs/mine.py --corpus corpus/v1/text --out data/pairs-mined.jsonl
python3 pairs/dedup.py data/pairs-*.jsonl --semantic 0.95 --out data/pairs-v1.jsonl
python3 train/embedder.py --pairs data/pairs-v1.jsonl --batch 32 --epochs 8 --out ckpt/emb-r1.pt
python3 pairs/mine-hard.py --model ckpt/emb-r1.pt --pairs data/pairs-v1.jsonl \
    --margin-filter --out data/hard-negs.json
python3 train/embedder.py --base ckpt/emb-r1.pt --pairs data/pairs-v1.jsonl \
    --hard-negs data/hard-negs.json --epochs 3 --out ckpt/emb-r2.pt
python3 eval/retrieval.py --index run/index/v1 --model ckpt/emb-r2.pt --mode hybrid
```

## What to expect
See `docs/EXPECTED.md` for the measured reference numbers for every stage of
this lab, including the failures that are *supposed* to happen.

## Scaling up
The same scripts take GPU-scale arguments. `train/cfg-125m.yaml` is the
125M-class configuration; `train/sft.py`, `quant/awq.py --plan`, and the
other wrappers target the pinned HF/TRL/autoawq stack for 8B-class models on
a CUDA host (`env/requirements.lock`).
