v1.2.3 (6 Sep 2026) - cross-platform release
- labs/run_all.py: cross-platform orchestrator (Windows/Linux/macOS), resumable with --from; the .sh scripts delegate to it
- bench/soak.py and deploy/stage.py: Python equivalents of the bash tools
- .gitattributes: LF line endings enforced so scripts survive a Windows checkout
- gguf pinned in both requirement files (was missing); tests/env_check.py verifies every module the lab suite imports; regression test asserts every third-party import is pinned
- README: platform notes with PowerShell and bash venv recipes; docs use `python`, not `python3`
- quant/merge.py --adapters sft,dpo,grpo; Lab E artifact chain and deployment branches documented

v1.2.2 (6 Sep 2026) - Windows compatibility
- all text file I/O explicitly UTF-8 (fixes tok/train.py "stream did not contain valid UTF-8" on Windows)

v1.2.1 (5 Sep 2026) — book v1.7 release
- quant/merge.py --adapters sft,dpo,grpo: merge the final aligned stack (each adapter bound to its base)
- Lab E artifact chain: QAT writes the bf16 master only (ckpt/qat-master); packing at export; the packed GGUF is what is gated
- docs/LAB_E.md: the two deployment branches (base+adapters vs merged bundle); Orin CUDA build note
v1.2.0 (4 Sep 2026) — book release for Beyond the Frontier v1.6
- llama.cpp pinned by immutable tag (env/llama_cpp.pin)
- llmcompressor replaces deprecated AutoAWQ in env/requirements.lock
- tests/tokenizer_compat.py: speculative-decoding preflight (Ch. 17, Lab E)
- tests/test_repo.py: every book JSON sample must parse
- corpus/build.py sources include procedures (App. B command)
Beyond the Frontier companion repository v1.1
- synthetic SATCOM proxy corpus + all derived datasets (seeded, reproducible)
- all four labs runnable end to end on CPU: labs/run_all.sh
- reference results: docs/EXPECTED.md
v1.1.1 (2 Sep 2026): GRPO advantage normalized by group std (matches Appendix A); --mrl nested Matryoshka objective on train/embedder.py (Lab D truncation exercise now has a trained basis).
