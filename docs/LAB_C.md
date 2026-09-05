# Lab C — runnable walkthrough

Full method and rationale: **Appendix C** of the book. This file is the
executable companion; every command below runs on a laptop CPU against the
synthetic corpus.

## Prerequisites
```bash
pip install -r env/requirements-cpu.txt
python3 tests/env_check.py --lab C
python3 data/make_synthetic.py --scale small     # once, for all labs
```

## Commands
```bash
python3 tests/template_parity.py --tokenizer tok/domain-32k
python3 tests/packing_inspect.py
python3 train/sft_local.py --base ckpt/lab/step00001200.pt --lora-r 16 --epochs 6 \
    --lr 2e-3 --oversample "Recommend a MODCOD:6" --out ckpt/sft-adapter.pt
python3 train/dpo_local.py --base ckpt/sft-adapter.pt --beta 0.1 --out ckpt/dpo-adapter.pt
python3 train/grpo_local.py --base ckpt/dpo-adapter.pt --iters 40 --temperature 0.7 \
    --out ckpt/grpo-adapter.pt
```

## What to expect
See `docs/EXPECTED.md` for the measured reference numbers for every stage of
this lab, including the failures that are *supposed* to happen.

## Scaling up
The same scripts take GPU-scale arguments. `train/cfg-125m.yaml` is the
125M-class configuration; `train/sft.py`, `quant/awq.py --plan`, and the
other wrappers target the pinned HF/TRL/autoawq stack for 8B-class models on
a CUDA host (`env/requirements.lock`).
