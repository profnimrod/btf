#!/usr/bin/env python
"""Run every laboratory end to end on CPU with the synthetic corpus.

    python labs/run_all.py            full run   (~30-60 min, real results)
    python labs/run_all.py --smoke    wiring check (~5 min, results meaningless)
    python labs/run_all.py --from d   resume at a stage (setup, b, c, d, e)

Cross-platform: Windows (PowerShell/cmd), Linux, macOS. No bash needed.
Each stage prints the numbers docs/EXPECTED.md tells you to expect.
"""
from __future__ import annotations
import os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def run(*args, check=True, capture=False):
    cmd = [PY] + [str(a) for a in args]
    print("$", " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, text=True, encoding="utf-8",
                       capture_output=capture)
    if check and r.returncode != 0:
        sys.exit(f"[run_all] stage failed: {' '.join(cmd[1:])}")
    return r


def setup(P, scratch):
    print("############ SETUP ############")
    run("tests/env_check.py")
    run("data/make_synthetic.py", "--scale", "small")
    run("corpus/build.py", "--sources", "data/samples", "--dedup", "minhash:0.9",
        "--decontam", "eval/JUDGED-v1", "--out", "corpus/v1")
    run("tok/train.py", "--corpus", "corpus/v1", "--vocab", "4000", "--digits", "split",
        "--out", "tok/domain-32k")
    gen = scratch / "gen"; gen.mkdir()
    for f in (ROOT / "corpus/v1/text").glob("GEN-*.txt"):
        shutil.copy(f, gen / f.name)
    run("tok/train.py", "--corpus", gen, "--vocab", "2000", "--digits", "keep", "--out", "tok/general")
    run("tok/fragmentation.py", "--tok", "tok/domain-32k", "--ref", "tok/general",
        "--terms", "eval/domain-terms.txt")

def lab_b(P, scratch):
    print("############ LAB B: pretrain a small domain model ############")
    run("train/pretrain.py", "--config", "train/cfg-lab.yaml", "--data", "corpus/v1",
        "--max-steps", P["PRE_STEPS"], "--ckpt-every", P["CKPT_EVERY"], "--ckpt-dir", "ckpt/lab",
        "--eval-heartbeat", "eval/canary-prompts.jsonl", "--device", "cpu", "--log", "logs/lab.jsonl")
    run("train/verify-resume.py", "--ckpt", f"ckpt/lab/{P['RESUME_CKPT']}", "--config", "train/cfg-lab.yaml",
        "--data", "corpus/v1", "--log", "logs/lab.jsonl", "--steps", "3")
    held = scratch / "heldout"; held.mkdir()
    for f in (ROOT / "corpus/v1/text").glob("PROC-2*.txt"):
        shutil.copy(f, held / f.name)
    run("eval/bpb.py", "--ckpt", f"ckpt/lab/{P['PRE_CKPT']}", "--tok", "tok/domain-32k",
        "--text", held, "--label", "lab-model")
    run("train/model.py", "train/cfg-125m.yaml")

def lab_c(P, scratch):
    print("############ LAB C: fine-tune and preference-tune ############")
    run("tests/template_parity.py", "--tokenizer", "tok/domain-32k")
    run("tests/packing_inspect.py")
    run("data/audit-sample.py", "--data", "data/sft-v1/train.jsonl")
    run("train/sft_local.py", "--base", f"ckpt/lab/{P['PRE_CKPT']}", "--data", "data/sft-v1/train.jsonl",
        "--lora-r", "16", "--epochs", P["SFT_EPOCHS"], "--lr", "2e-3",
        "--oversample", "Recommend a MODCOD:6", "--out", "ckpt/sft-adapter.pt")
    run("train/dpo_local.py", "--base", "ckpt/sft-adapter.pt", "--beta", "0.1", "--epochs", "1",
        "--limit", P["DPO_LIMIT"], "--out", "ckpt/dpo-adapter.pt")
    run("train/grpo_local.py", "--base", "ckpt/dpo-adapter.pt", "--group", "6",
        "--iters", P["GRPO_ITERS"], "--temperature", "0.7", "--out", "ckpt/grpo-adapter.pt")
    run("quant/merge.py", "--adapters", "sft,dpo,grpo", "--out", "ckpt/merged.pt")
    run("eval/harness.py", "--ckpt", "ckpt/merged.pt", "--name", "parent", "--limit", P["HL"], "--lb-limit", P["LBL"])
    run("eval/guardrails.py", "--ckpt", "ckpt/merged.pt", "--both-directions")

def lab_d(P, scratch):
    print("############ LAB D: embedder and retrieval ############")
    run("pairs/mine.py", "--corpus", "corpus/v1/text", "--out", "data/pairs-mined.jsonl")
    run("pairs/synth.py", "--corpus", "corpus/v1/text", "--decontam", "eval/JUDGED-v1", "--out", "data/pairs-synth.jsonl")
    run("pairs/dedup.py", "data/pairs-mined.jsonl", "data/pairs-synth.jsonl", "--semantic", "0.95", "--out", "data/pairs-v1.jsonl")
    run("index/build.py", "--src", "corpus/v1/text", "--bm25", "--out", "run/index/baseline")
    run("eval/retrieval.py", "--index", "run/index/baseline", "--mode", "bm25", "--k", "10")
    run("train/embedder.py", "--pairs", "data/pairs-v1.jsonl", "--epochs", "0", "--out", "ckpt/emb-untuned.pt")
    run("index/build.py", "--src", "corpus/v1/text", "--dense", "ckpt/emb-untuned.pt", "--out", "run/index/dense0")
    run("eval/retrieval.py", "--index", "run/index/dense0", "--model", "ckpt/emb-untuned.pt", "--mode", "dense")
    run("train/embedder.py", "--pairs", "data/pairs-v1.jsonl", "--batch", "32", "--mrl", "128,64",
        "--epochs", P["EMB_EPOCHS"], "--lr", "1e-3", "--out", "ckpt/emb-r1.pt")
    run("index/build.py", "--src", "corpus/v1/text", "--dense", "ckpt/emb-r1.pt", "--out", "run/index/dense1")
    run("eval/retrieval.py", "--index", "run/index/dense1", "--model", "ckpt/emb-r1.pt", "--mode", "dense")
    run("pairs/mine-hard.py", "--model", "ckpt/emb-r1.pt", "--pairs", "data/pairs-v1.jsonl",
        "--margin-filter", "--denoise", "cross-encoder", "--out", "data/hard-negs.json")
    run("train/embedder.py", "--base", "ckpt/emb-r1.pt", "--pairs", "data/pairs-v1.jsonl",
        "--hard-negs", "data/hard-negs.json", "--batch", "32", "--epochs", P["EMB_R2_EPOCHS"],
        "--lr", "3e-4", "--out", "ckpt/emb-r2.pt")
    run("index/build.py", "--src", "corpus/v1/text", "--dense", "ckpt/emb-r2.pt", "--bm25",
        "--acl-metadata", "--out", "run/index/v1")
    run("eval/retrieval.py", "--index", "run/index/v1", "--model", "ckpt/emb-r2.pt", "--mode", "dense")
    run("eval/retrieval.py", "--index", "run/index/v1", "--model", "ckpt/emb-r2.pt", "--mode", "hybrid",
        "--save", "eval/reports/retrieval-hybrid.json")
    run("eval/pool.py", "--queries", "eval/JUDGED-v1/queries.jsonl", "--index", "run/index/v1",
        "--systems", "bm25,dense", "--model", "ckpt/emb-r2.pt", "--out", "eval/pool.jsonl")
    run("eval/judge-sheet.py", "--pool", "eval/pool.jsonl", "--judges", "2", "--overlap", "0.25")
    run("train/rerank.py", "--pairs", "data/pairs-v1.jsonl", "--hard-negs", "data/hard-negs.json",
        "--epochs", "1", "--out", "ckpt/reranker.pt")
    run("eval/bins.py", "--index", "run/index/v1", "--model", "ckpt/emb-r2.pt", "--save", "eval/reports/bins.json")
    run("index/build.py", "--src", "corpus/v1/text", "--dense", "ckpt/emb-r2.pt", "--pq", "m=8", "--out", "run/index/v1-pq")
    run("eval/retrieval.py", "--index", "run/index/v1-pq", "--model", "ckpt/emb-r2.pt", "--mode", "dense")

def lab_e(P, scratch):
    print("############ LAB E: compress, serve, field ############")
    run("calib/build.py", "--n", P["CALIB_N"], "--len", "192", "--mix", "deploy", "--decontam", "eval/JUDGED-v1", "--out", "calib/v1")
    run("quant/awq.py", "--native", "--model", "ckpt/merged.pt", "--calib", "calib/v1", "--group", "64", "--bits", "4", "--out", "ckpt/awq-int4.pt")
    run("eval/harness.py", "--ckpt", "ckpt/awq-int4.pt", "--name", "awq-int4", "--limit", P["HL"], "--lb-limit", P["LBL"])
    run("quant/gptq.py", "--native", "--model", "ckpt/merged.pt", "--calib", "calib/v1", "--group", "64", "--bits", "4", "--out", "ckpt/gptq-int4.pt")
    run("eval/harness.py", "--ckpt", "ckpt/gptq-int4.pt", "--name", "gptq-int4", "--limit", P["HL"], "--lb-limit", P["LBL"])
    r = run("eval/delta.py", "--parent", "parent", "--children", "awq-int4,gptq-int4", "--gates", check=False)
    if r.returncode != 0:
        print("[expected] a gate failure here is the lesson (Ch. 14)")
    run("quant/qat.py", "--master", "ckpt/merged.pt", "--fake-quant", "int4:g64", "--data", "data/sft-v1/train.jsonl",
        "--focus", "REQ-SC", "--steps", P["QAT_STEPS"], "--out", "ckpt/qat-master.pt")
    run("eval/harness.py", "--ckpt", "ckpt/qat-master.pt", "--name", "qat-master", "--limit", P["HL"], "--lb-limit", P["LBL"])
    run("distill/gen.py", "--teacher", "ckpt/merged.pt", "--n", P["DISTILL_N"], "--filter", "checker", "--out", "data/distill-v1.jsonl")
    run("distill/train.py", "--data", "data/distill-v1.jsonl", "--student-layers", "2", "--student-dim", "96",
        "--epochs", P["DISTILL_EPOCHS"], "--lr", "2e-3", "--out", "ckpt/student.pt")
    run("eval/harness.py", "--ckpt", "ckpt/student.pt", "--name", "student", "--limit", P["HL"], "--lb-limit", P["LBL"])
    run("eval/select.py", "--top", "3")
    run("eval/mcnemar.py", "--a", "eval/reports/retrieval-hybrid.json", "--b", "eval/reports/retrieval-hybrid.json")
    run("export/gguf.py", "--model", "ckpt/qat-master.pt", "--dtype", "f16", "--out", "gguf/asst-f16.gguf", "--name", "btf-lab")
    print("# production path: llama.cpp/build/bin/llama-quantize gguf/asst-f16.gguf gguf/asst-q4.gguf Q4_K_M")
    run("bench/soak.py", "--model", "ckpt/awq-int4.pt", "--minutes", P["SOAK_MIN"], "--log", "soak/q4.csv")
    run("bundle/seal.py", "--model", "ckpt/awq-int4.pt", "--index", "run/index/v1-pq",
        "--scaffold", "serve/template.py", "--out", "bundle/ops-edge-v1")
    r = run("registry/publish.py", "--artifact", "ckpt/awq-int4.pt", "--type", "model",
            "--datasets", "satcom-sft-v1", "--license", "open", capture=True)
    print(r.stdout, end="")
    digest = r.stdout.strip().splitlines()[-1].strip()
    run("registry/fetch.py", "--digest", digest, "--verify")
    run("registry/bom.py", "--digest", digest)
    run("deploy/stage.py", "--target", "run/fleet", "bundle/ops-edge-v1")

STAGES = ["setup", "b", "c", "d", "e"]


def main():
    args = sys.argv[1:]
    smoke = "--smoke" in args
    start = "setup"
    if "--from" in args:
        start = args[args.index("--from") + 1].lower()
        if start not in STAGES:
            sys.exit(f"--from must be one of {STAGES}")
    if smoke:
        P = dict(PRE_STEPS=120, CKPT_EVERY="60steps", RESUME_CKPT="step00000060.pt",
                 PRE_CKPT="step00000120.pt", SFT_EPOCHS=2, DPO_LIMIT=20, GRPO_ITERS=6,
                 EMB_EPOCHS=2, EMB_R2_EPOCHS=1, DISTILL_N=40, DISTILL_EPOCHS=3,
                 HL=8, LBL=3, QAT_STEPS=15, SOAK_MIN=0.1, CALIB_N=16)
        print("### SMOKE PROFILE - wiring check only; results are not meaningful ###")
    else:
        P = dict(PRE_STEPS=1200, CKPT_EVERY="600steps", RESUME_CKPT="step00000600.pt",
                 PRE_CKPT="step00001200.pt", SFT_EPOCHS=6, DPO_LIMIT=80, GRPO_ITERS=40,
                 EMB_EPOCHS=8, EMB_R2_EPOCHS=3, DISTILL_N=200, DISTILL_EPOCHS=12,
                 HL=30, LBL=8, QAT_STEPS=60, SOAK_MIN=0.5, CALIB_N=64)
    scratch = Path(tempfile.mkdtemp(prefix="btf-"))
    order = {"setup": setup, "b": lab_b, "c": lab_c, "d": lab_d, "e": lab_e}
    for name in STAGES[STAGES.index(start):]:
        order[name](P, scratch)
    shutil.rmtree(scratch, ignore_errors=True)
    print("############ ALL LABS COMPLETE ############")
    print("Compare your numbers against docs/EXPECTED.md.")


if __name__ == "__main__":
    main()
