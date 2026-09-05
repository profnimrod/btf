"""Lab E: export a lab checkpoint to GGUF for llama.cpp-family runtimes
(Ch. 19). Writes a real GGUF container with llama-architecture tensor
names. For 8B-class HF models, use llama.cpp's convert script instead."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import torch

# this file is named gguf.py: drop its own directory from the path so that
# `import gguf` finds the installed package, not this script
_here = os.path.dirname(os.path.abspath(__file__))
sys.path = [q for q in sys.path if os.path.abspath(q or ".") != _here]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="btf-lab")
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    a = ap.parse_args()
    import gguf
    import numpy as np
    st = torch.load(a.model, map_location="cpu", weights_only=False)
    sd, cfg = st["model"], st["cfg"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(a.out, "llama")
    w.add_name(a.name)
    w.add_context_length(cfg["context"])
    w.add_embedding_length(cfg["d_model"])
    w.add_block_count(cfg["n_layers"])
    w.add_feed_forward_length(cfg["d_ff"])
    w.add_head_count(cfg["n_heads"])
    w.add_head_count_kv(cfg["n_kv_heads"])
    w.add_rope_freq_base(cfg["rope_theta"])
    w.add_layer_norm_rms_eps(1e-5)
    name_map = {"emb.weight": "token_embd.weight",
                "norm_f.weight": "output_norm.weight",
                "lm_head.weight": "output.weight"}
    for i in range(cfg["n_layers"]):
        name_map |= {
            f"blocks.{i}.n1.weight": f"blk.{i}.attn_norm.weight",
            f"blocks.{i}.n2.weight": f"blk.{i}.ffn_norm.weight",
            f"blocks.{i}.attn.wq.weight": f"blk.{i}.attn_q.weight",
            f"blocks.{i}.attn.wk.weight": f"blk.{i}.attn_k.weight",
            f"blocks.{i}.attn.wv.weight": f"blk.{i}.attn_v.weight",
            f"blocks.{i}.attn.wo.weight": f"blk.{i}.attn_output.weight",
            f"blocks.{i}.mlp.gate.weight": f"blk.{i}.ffn_gate.weight",
            f"blocks.{i}.mlp.up.weight": f"blk.{i}.ffn_up.weight",
            f"blocks.{i}.mlp.down.weight": f"blk.{i}.ffn_down.weight"}
    n = 0
    for k, v in sd.items():
        if k in name_map:
            arr = v.detach().float().numpy()
            w.add_tensor(name_map[k], arr.astype(np.float16 if a.dtype == 'f16' else np.float32))
            n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    mb = Path(a.out).stat().st_size / 1e6
    print(f"[gguf] wrote {n} tensors -> {a.out} ({mb:.2f} MB)")


if __name__ == "__main__":
    main()
