# Lab D — runnable walkthrough

Full method and rationale: **Appendix D** of the book. This file is the
executable companion; every command below runs on a laptop CPU against the
synthetic corpus.

## Prerequisites
`python` means your Python 3.12 interpreter (`python3` on Linux/macOS, `py -3.12` on Windows); see the README's platform notes.

```bash
pip install -r env/requirements-cpu.txt
python tests/env_check.py --lab D
python data/make_synthetic.py --scale small     # once, for all labs
```

## Commands
```bash
python pairs/mine.py --corpus corpus/v1/text --out data/pairs-mined.jsonl
python pairs/dedup.py data/pairs-*.jsonl --semantic 0.95 --out data/pairs-v1.jsonl
python train/embedder.py --pairs data/pairs-v1.jsonl --batch 32 --epochs 8 --out ckpt/emb-r1.pt
python pairs/mine-hard.py --model ckpt/emb-r1.pt --pairs data/pairs-v1.jsonl \
    --margin-filter --out data/hard-negs.json
python train/embedder.py --base ckpt/emb-r1.pt --pairs data/pairs-v1.jsonl \
    --hard-negs data/hard-negs.json --epochs 3 --out ckpt/emb-r2.pt
python eval/retrieval.py --index run/index/v1 --model ckpt/emb-r2.pt --mode hybrid
```

## What to expect
See `docs/EXPECTED.md` for the measured reference numbers for every stage of
this lab, including the failures that are *supposed* to happen.

## Scaling up
The same scripts take GPU-scale arguments. `train/cfg-125m.yaml` is the
125M-class configuration; `train/sft.py`, `quant/awq.py --plan`, and the
other wrappers target the pinned HF/TRL/autoawq stack for 8B-class models on
a CUDA host (`env/requirements.lock`).
