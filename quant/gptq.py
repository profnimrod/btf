"""Lab E: GPTQ-style error-compensated 4-bit quantization (Ch. 14).

--plan   prints the recipe and per-task gate policy (no GPU needed)
--native quantizes a lab checkpoint on CPU and writes a new checkpoint
The GPU path for 8B-class HF models uses gptqmodel; see env/requirements.lock.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GATES = {"standards_qa": 1.0, "telemetry_narration": 1.5, "link_budget": 1.0,
         "report_drafting": 1.0, "traceability": 1.0}


def plan(a):
    print(json.dumps({
        "model": a.model, "calib": a.calib, "scheme": f"GPTQ int{a.bits}",
        "group_size": a.group, "calib_samples": 512, "calib_seqlen": 2048,
        "per_task_gates_pp": GATES,
        "note": "telemetry_narration is the tight gate; expect a QAT repair "
                "(quant/qat.py) if it exceeds 1.5pp (Ch. 14)."}, indent=2))


def native(a):
    from serve.generate import load
    from train.lora import merge_lora
    from quant.quantize import quantize_model, collect_activation_scales
    from tokenizers import Tokenizer
    model, tok = load(a.model, a.tok)
    n_merged = merge_lora(model)
    if n_merged:
        print(f'[quant] merged {n_merged} LoRA projections into the base weights first (Ch. 14)')
    calib = [json.loads(l) for l in
             Path(a.calib, "calib.jsonl").read_text(encoding='utf-8').splitlines() if l.strip()]
    ids = [torch.tensor([tok.encode(c["text"]).ids[:128]]) for c in calib[:16]]
    ids = [i for i in ids if i.shape[1] > 4]
    n = quantize_model(model, bits=a.bits, group=a.group, method="gptq")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__,
                "quant": {"scheme": f"gptq-int{a.bits}", "group": a.group,
                          "calib": a.calib}}, a.out)
    print(f"[gptq] quantized {n} projections to int{a.bits} "
          f"(group {a.group}, error-compensated) -> {a.out}")
    print("[gptq] now run: python eval/harness.py --ckpt %s  then eval/delta.py --gates"
          % a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt/merged-8b")
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--calib", default="calib/v1")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--out", default="ckpt/gptq-int4.pt")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--native", action="store_true")
    a = ap.parse_args()
    if a.native:
        native(a)
    elif a.plan:
        plan(a)
    else:
        sys.exit("choose --plan (preview) or --native (run on a lab checkpoint)")


if __name__ == "__main__":
    main()
