# Expected results (CPU, synthetic corpus, seed 20260826)

**Two ways to run.** `python labs/run_all.py --smoke` exercises every command in
about five minutes as an environment and wiring check — its *numbers are
meaningless* (the model barely trains), and stages legitimately report 0.0.
`python labs/run_all.py` is the real run; the table below is that run.

Measured on a single CPU core. Your numbers will differ slightly with
threading and library versions; the **shape** is the contract, not the
third decimal. Total wall time for `labs/run_all.sh`: roughly 30–60 minutes.

## Setup
| Check | Expected |
|---|---|
| Documents generated | 217 across 7 types |
| After dedup gate (0.9) | 213 kept, 4 removed |
| Judged queries / train facts | 85 / 102 (disjoint by document) |
| Tokenizer fragmentation | domain ≈4.8 vs general ≈11.9 tokens/term |

## Lab B — pretrain
| Stage | Expected |
|---|---|
| Loss at step 1 | ≈7.2 (≈ln V, the uniform-guess ceiling) |
| Loss at step 1200 | ≈0.4 |
| Resume drill | every step `data OK loss OK`, **RESUME DRILL: PASS** |
| BPB on held-out procedures | ≈0.13 |
| Reference config param count | **128.40M** (`train/model.py train/cfg-125m.yaml`) |

## Lab C — adapt
| Stage | Expected |
|---|---|
| Template parity / packing | both **PASS** |
| Audit sample | n=59 for 5% defect @ 95% |
| SFT (LoRA r=16) | loss 4.8 → ≈1.0; adapter ≈0.24 MB |
| DPO (β=0.1) | margin>0 on ≈74/80 pairs |
| GRPO vs checker | mean reward ≈0.56 → ≈0.73 |
| Gaming audit | top transcripts converge on **8PSK3/4** — the model chasing the efficiency term. Read them: this is the audit the book requires. |
| Harness (merged) | overall ≈35.6 |

## Lab D — retrieval
| Stage | Recall@10 |
|---|---|
| BM25 baseline | ≈0.48 |
| Dense, untuned encoder | ≈0.09 |
| + InfoNCE fine-tune | ≈0.45 |
| + hard negatives (continued from r1) | ≈0.53 |
| **Hybrid fusion (dense + BM25)** | **≈0.62** |

Judge agreement: Cohen's κ ≈0.8 (above the 0.6 rubric threshold).
PQ index (`m=8`) costs a few points of recall for a large memory saving.

**If round 2 makes things worse**, check you passed `--base ckpt/emb-r1.pt`.
Without it the round retrains from scratch — a real bug found while building
this repo, and exactly the "mining hurt us" symptom Chapter 9 warns about.

## Lab E — compress, serve, field
| Stage | Expected |
|---|---|
| AWQ int4 (activation-aware) | parity with parent (≈35.6), all gates ok |
| GPTQ int4 (error-compensated) | traceability −66.7pp → **GATE FAIL** |
| QAT repair | recovers partially and *damages another suite* — re-run the FULL report; a repair validated only on its target task is not validated (Ch. 14) |
| Distilled student (2L, d96) | ≈35.6 — parity with its teacher on these narrow tasks |
| GGUF export | real container, 39 tensors |
| Soak | sustained tok/s logged to `soak/q4.csv` |
| Bundle drills | corrupted bundle **refused**; failed-BIT **auto-reverts**, active slot unchanged |

## Honest notes
- The lab model is ~1.5M parameters. It learns formats, citations, and
  identifier patterns; it does **not** reason. Suites like `standards_qa`
  score 0 because exact-match on a specific number is beyond it. That is
  expected and is why the harness reports per-suite rather than an average.
- int4 is survivable here only because AWQ protects salient channels; at this
  scale quantization is far more brittle than it is at 8B.
- Guardrail results are structural, not behavioural: the suite exists so the
  gate is wired and the both-direction discipline is visible.
