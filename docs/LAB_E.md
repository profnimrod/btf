# Lab E — runnable walkthrough

Full method and rationale: **Appendix E** of the book. This file is the
executable companion; every command below runs on a laptop CPU against the
synthetic corpus.

## Prerequisites
```bash
pip install -r env/requirements-cpu.txt
python3 tests/env_check.py --lab E
python3 data/make_synthetic.py --scale small     # once, for all labs
```

## Commands
```bash
python3 calib/build.py --out calib/v1
python3 quant/merge.py --adapter ckpt/grpo-adapter.pt --out ckpt/merged.pt
python3 quant/awq.py --native --model ckpt/merged.pt --calib calib/v1 --bits 4 --out ckpt/awq-int4.pt
python3 eval/delta.py --parent parent --children awq-int4,gptq-int4 --gates
python3 export/gguf.py --model ckpt/awq-int4.pt --out gguf/asst-q4.gguf
python3 bundle/seal.py --model ckpt/awq-int4.pt --index run/index/v1-pq --out bundle/ops-edge-v1
bash deploy/stage.sh --target run/fleet bundle/ops-edge-v1
```

## What to expect
See `docs/EXPECTED.md` for the measured reference numbers for every stage of
this lab, including the failures that are *supposed* to happen.

## Scaling up
The same scripts take GPU-scale arguments. `train/cfg-125m.yaml` is the
125M-class configuration; `train/sft.py`, `quant/awq.py --plan`, and the
other wrappers target the pinned HF/TRL/autoawq stack for 8B-class models on
a CUDA host (`env/requirements.lock`).

## Deployment branches
- **Branch A** — stable quantized base + unmerged adapters; adapter-delta updates are valid, and each adapter is bound to the exact base it was trained against.
- **Branch B** — merged (SFT→DPO→GRPO), QAT-repaired, re-quantized GGUF bundle; adapters are already folded in, so updates ship as whole bundles.

## Orin build
`cmake -B llama.cpp/build llama.cpp -DGGML_CUDA=ON` from the tag in `env/llama_cpp.pin`, with the matching JetPack toolchain; confirm GPU offload in the soak configuration.
