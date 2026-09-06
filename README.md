# Beyond the Frontier — Companion Repository

Working code for the laboratories in *Beyond the Frontier: Domain-Specific
Generative AI from First Principles to the Tactical Edge* (Emergence
Publications, 2026). This is the repository Appendix F describes and the
labs (Appendices B–E) drive.

> **Proxy data only.** Every example uses unclassified proxy data — public
> CCSDS/DVB-S2X text, synthetic telemetry, public orbital elements — so the
> whole build reproduces openly while the techniques transfer to the
> classified and proprietary corpora where readers actually work.

## What runs where

The repository is split by **what a laptop can execute** versus **what needs
a GPU**, and nothing is faked in between:

- **CPU-executable, fully implemented and tested** — the model definition
  (`train/model.py`, incl. the ternary branch), the training loop with
  bit-exact checkpoint/resume (`train/pretrain.py`, `train/verify-resume.py`),
  the gated corpus builder (`corpus/build.py`), the signing registry
  (`registry/registry.py`), bundle seal/verify/BIT/rollback
  (`bundle/bundle.py`), the link-budget checker (`eval/link_budget.py`), the
  statistics (`eval/stats.py`), packing and template invariants
  (`data/packing.py`, `serve/template.py`, `tests/`).
- **Every lab stage, CPU-executable** — SFT with LoRA, DPO, GRPO against the
  deterministic checker, embedder training with hard-negative mining, hybrid
  indexing, AWQ/GPTQ quantization, QAT repair, distillation, real GGUF export,
  bundle sealing with built-in test and rollback drills. All of it runs at lab
  scale on the bundled corpus.
- **8B-class runs** — the same scripts with GPU-scale arguments, over the
  pinned HF/TRL/autoawq stack (`env/requirements.lock`). `--plan` on those
  wrappers prints the exact configuration without a card.

## Platform notes (read first)

Everything runs on **Windows, Linux, and macOS** with **Python 3.12** (3.11 untested; 3.13 lacks
some pinned wheels). In the commands below, `python` means your 3.12 interpreter — `python3` on
Linux/macOS, `py -3.12` or `python` on Windows. Use a virtual environment:

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r env\requirements-cpu.txt
python tests\env_check.py
python labs\run_all.py --smoke
```

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -r env/requirements-cpu.txt
python tests/env_check.py
python labs/run_all.py --smoke
```

The `.sh` scripts (`labs/run_all.sh`, `bench/soak.sh`, `deploy/stage.sh`) are thin wrappers around
the Python equivalents (`labs/run_all.py`, `bench/soak.py`, `deploy/stage.py`) for readers who prefer
bash; on Windows they need Git Bash or WSL, or just call the `.py` file directly. All text I/O is
explicitly UTF-8, and `.gitattributes` keeps line endings LF on checkout.

GPU stages on Windows: install a CUDA-enabled PyTorch from the official PyTorch index for your CUDA
version before the pinned stack; the default PyPI `torch` wheel on Windows is CPU-only.

## Quickstart (laptop, no GPU)

**Everything below runs end to end on one CPU core with the bundled synthetic
corpus — no downloads, no GPU, no data hunt.**

```bash
pip install -r env/requirements-cpu.txt
python data/make_synthetic.py --scale small   # 217-document proxy corpus
python labs/run_all.py --smoke                 # 5-min wiring check first
python labs/run_all.py                         # all four labs, ~30-60 min
```

`docs/EXPECTED.md` lists the reference numbers for every stage.


```bash
pip install -r env/requirements-cpu.txt
python tests/env_check.py
python -m pytest tests/test_repo.py -q         # 9 tests: model, checker, drills
```

Then run the real micro-pipeline end to end on CPU:

```bash
# tiny corpus + tokenizer are created by the Lab B walkthrough in docs/;
# this trains, checkpoints, and proves resume determinism:
python train/pretrain.py --config train/cfg-tiny.yaml --data corpus/tiny \
    --max-steps 12 --ckpt-every 8steps --ckpt-dir ckpt/tiny \
    --device cpu --log logs/tiny.jsonl
python train/verify-resume.py --ckpt ckpt/tiny/step00000008.pt \
    --config train/cfg-tiny.yaml --data corpus/tiny --log logs/tiny.jsonl
```

See `eval/link_budget.py` and `bundle/bundle.py` — both have `__main__`
demos (the checker's reward table; the seal → BIT → rollback drill).

## Layout

```
corpus/    build.py (gates + attestation), samples
tok/       tokenizer train + fragmentation + encode
data/      packing, mining, dataset cards
train/     model.py, pretrain.py, verify-resume.py, sft/dpo/grpo, configs
distill/   teacher-gen, student training
quant/     awq, gptq, qat, calibration
eval/      link_budget, stats, delta, bins, harness
index/     chunking, hybrid build, PQ variants
serve/     template, vLLM + llama.cpp configs, gateway
bundle/    seal, sign, verify, BIT, A/B activate
deploy/    slots, beacons, fleet ledger
registry/  content-addressed store + Ed25519 signing + BOM
tests/     env_check, template_parity, packing_inspect, test_repo (pytest)
env/       requirements.lock, requirements-cpu.txt
ci/        gates.yaml
docs/      per-lab walkthroughs (LAB_B … LAB_F)
```

## Mapping to the book

| Book | Here |
|---|---|
| Lab B — pretrain from scratch | `train/`, `tok/`, `corpus/`, `docs/LAB_B.md` |
| Lab C — SFT + DPO + GRPO | `train/sft.py` `dpo.py` `grpo.py`, `eval/link_budget.py` |
| Lab D — embedder + RAG | `pairs/`, `index/`, `eval/stats.py` |
| Lab E — compress + field | `quant/`, `bundle/`, `deploy/`, `export/` |
| Appendix F — tooling/registry | `registry/`, `env/`, `ci/`, this README |

## Honesty about scope

The CPU-path modules are complete and exercised by `pytest`. The GPU-stage
wrappers are real integration points, not full training runs — they call the
pinned libraries and expose every hyperparameter, but reproducing the book's
headline numbers needs the hardware the book specifies. Versions in
`env/requirements.lock` were current at the August 2026 snapshot; re-verify
before adopting (Appendix F).

## License

Code: MIT (see LICENSE). Proxy data: assembled from public sources for
instructional use.
