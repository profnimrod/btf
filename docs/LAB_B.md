# Lab B — runnable walkthrough

Full method and rationale: **Appendix B** of the book. This file is the
executable companion; every command below runs on a laptop CPU against the
synthetic corpus.

## Prerequisites
`python` means your Python 3.12 interpreter (`python3` on Linux/macOS, `py -3.12` on Windows); see the README's platform notes.

```bash
pip install -r env/requirements-cpu.txt
python tests/env_check.py --lab B
python data/make_synthetic.py --scale small     # once, for all labs
```

## Commands
```bash
python train/pretrain.py --config train/cfg-lab.yaml --data corpus/v1 \
    --max-steps 1200 --ckpt-dir ckpt/lab --device cpu --log logs/lab.jsonl
python train/verify-resume.py --ckpt ckpt/lab/step00000600.pt \
    --config train/cfg-lab.yaml --data corpus/v1 --log logs/lab.jsonl
python eval/bpb.py --ckpt ckpt/lab/step00001200.pt --tok tok/domain-32k --text /tmp/heldout
```

## What to expect
See `docs/EXPECTED.md` for the measured reference numbers for every stage of
this lab, including the failures that are *supposed* to happen.

## Scaling up
The same scripts take GPU-scale arguments. `train/cfg-125m.yaml` is the
125M-class configuration; `train/sft.py`, `quant/awq.py --plan`, and the
other wrappers target the pinned HF/TRL/autoawq stack for 8B-class models on
a CUDA host (`env/requirements.lock`).
