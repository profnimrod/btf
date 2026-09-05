"""Lab C: QLoRA SFT over the SATCOM instruction set (Ch. 11). Thin,
honest wrapper over TRL/PEFT. --plan prints the exact configuration and
the template-parity + packing invariants without needing a GPU; the real
run requires CUDA + bitsandbytes.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def plan(a):
    cfg = {
        "base": a.base, "data": a.data, "method": "QLoRA (4-bit NF4)",
        "lora": {"r": 32, "alpha": 64, "dropout": 0.05,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"]},
        "optim": {"lr": 2e-4, "epochs": 2, "warmup_ratio": 0.03,
                  "scheduler": "cosine", "packing": True,
                  "loss": "response-only"},
        "general_mix": a.general_mix,
        "eval_every_steps": 200,
        "invariants": ["tests/template_parity.py", "tests/packing_inspect.py"],
    }
    print(json.dumps(cfg, indent=2))
    print("\n[plan] run on a CUDA host with: transformers peft trl bitsandbytes")
    print("[plan] checkpoint selection: eval/select.py --top 3 (Ch. 11)")


def run(a):  # pragma: no cover  (GPU path)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig
    except Exception as e:
        sys.exit(f"[sft] GPU deps unavailable: {e}\n     use --plan on CPU.")
    if not torch.cuda.is_available():
        sys.exit("[sft] no CUDA device; use --plan.")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, quantization_config=bnb,
                                                 device_map="auto")
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # dataset loading + SFTTrainer(config=SFTConfig(packing=True, ...)).train()
    print("[sft] configured; wire dataset and call trainer.train()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/cfg-sft-qlora.yaml")
    ap.add_argument("--base", default="ckpt/cpt-8b")
    ap.add_argument("--data", default="data/sft-v1")
    ap.add_argument("--general-mix", type=float, default=0.2)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    plan(a) if a.plan else run(a)


if __name__ == "__main__":
    main()
