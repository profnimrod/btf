"""Load a lab checkpoint (optionally with a LoRA adapter) and generate.
Shared by the SFT/DPO/GRPO labs, the BIT canaries, and the RAG demo."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.model import Model, ModelConfig  # noqa: E402
from train.lora import apply_lora  # noqa: E402
from serve.template import render_serving  # noqa: E402


def load(ckpt: str, tok_dir: str):
    from tokenizers import Tokenizer
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "adapter" in st:                       # LoRA adapter on a frozen base
        base = torch.load(st["base"], map_location="cpu", weights_only=False)
        cfg = ModelConfig(**st["cfg"])
        m = Model(cfg)
        m.load_state_dict(base["model"], strict=False)
        apply_lora(m, r=st.get("lora_r", 8), alpha=2 * st.get("lora_r", 8))
        m.load_state_dict(st["adapter"], strict=False)
    else:
        cfg = ModelConfig(**st["cfg"])
        m = Model(cfg)
        m.load_state_dict(st["model"], strict=False)
    m.eval()
    tok = Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))
    return m, tok


def answer(model, tok, question: str, max_new: int = 40, temperature: float = 0.0):
    prompt = render_serving({"messages": [{"role": "user", "content": question}]})
    ids = torch.tensor([tok.encode(prompt).ids])
    out = model.generate(ids, max_new=max_new, temperature=temperature)[0].tolist()
    text = tok.decode(out)
    return text[len(tok.decode(ids[0].tolist())):].strip()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok", default="tok/domain-32k")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    m, tk = load(a.ckpt, a.tok)
    print(answer(m, tk, a.prompt, temperature=a.temperature))
