"""Minimal LoRA for the lab model (Ch. 11): W' = W + (alpha/r) B A.
Adapters are the only trainable tensors and save as a small file."""
from __future__ import annotations
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def apply_lora(model, r=8, alpha=16, targets=("wq", "wk", "wv", "wo")):
    n = 0
    for block in model.blocks:
        for name in targets:
            parent = block.attn if hasattr(block.attn, name) else block.mlp
            if hasattr(parent, name) and isinstance(getattr(parent, name), nn.Linear):
                setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha))
                n += 1
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad = True
            m.B.requires_grad = True
    return n


def adapter_state(model):
    return {k: v for k, v in model.state_dict().items()
            if k.endswith(".A") or k.endswith(".B")}


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def merge_lora(model):
    """Fold adapters into their base weights: W <- W + (alpha/r) B A.
    Lab C publishes adapters unmerged; Lab E merges for the single-artifact
    tiers before quantization and export (Chs. 11, 14)."""
    import torch.nn as nn
    merged = 0
    for block in model.blocks:
        for parent in (block.attn, block.mlp):
            for name, mod in list(parent.named_children()):
                if isinstance(mod, LoRALinear):
                    base = mod.base
                    base.weight.data = base.weight.data + \
                        (mod.B @ mod.A).detach() * mod.scale
                    setattr(parent, name, base)
                    merged += 1
    for p_ in model.parameters():
        p_.requires_grad = True
    return merged
