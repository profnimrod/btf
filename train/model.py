"""Decoder-only transformer for the Beyond the Frontier labs (Lab B).

Deliberately small and readable (~250 lines): RMSNorm, RoPE, GQA,
SwiGLU, tied embeddings, optional ternary (absmean, straight-through)
weights per Ch. 4/7. This is the file Lab B asks you to read in full.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_head: int = 64
    d_ff: int = 3072
    rope_theta: float = 10000.0
    context: int = 2048
    quant: str = "none"          # "none" | "ternary"
    tie_embeddings: bool = True
    init_std: float = 0.02

    @staticmethod
    def from_yaml(path: str) -> "ModelConfig":
        import yaml
        with open(path, encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        m = raw.get("model", raw)
        keys = ModelConfig.__dataclass_fields__
        return ModelConfig(**{k: m[k] for k in m if k in keys})

    def describe(self) -> str:
        q = " (ternary: absmean STE)" if self.quant == "ternary" else ""
        return (f"{self.n_layers}L d{self.d_model} h{self.n_heads}/kv{self.n_kv_heads} "
                f"ff{self.d_ff} vocab{self.vocab} ctx{self.context}{q}")


def ternary_ste(w: torch.Tensor) -> torch.Tensor:
    """Absmean ternary quantization with a straight-through estimator.

    W_t = clip(round(W/gamma), -1, 1) * gamma,  gamma = mean(|W|).
    Forward uses W_t; backward passes gradients to W unchanged (Ch. 4).
    """
    gamma = w.abs().mean().clamp_min(1e-8)
    wq = (w / gamma).round().clamp_(-1, 1) * gamma
    return w + (wq - w).detach()


class BitLinear(nn.Linear):
    def __init__(self, d_in: int, d_out: int):
        super().__init__(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, ternary_ste(self.weight))


def make_linear(cfg: ModelConfig, d_in: int, d_out: int) -> nn.Module:
    if cfg.quant == "ternary":
        return BitLinear(d_in, d_out)
    return nn.Linear(d_in, d_out, bias=False)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def rope_tables(d_head: int, context: int, theta: float):
    inv = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(context).float()
    freqs = torch.outer(t, inv)                      # [T, d_head/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: [B, T, H, d_head]; rotate pairs (x0,x1),(x2,x3),...
    T = x.shape[1]
    c = cos[:T].to(x.dtype)[None, :, None, :]
    s = sin[:T].to(x.dtype)[None, :, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wq = make_linear(cfg, cfg.d_model, cfg.n_heads * cfg.d_head)
        self.wk = make_linear(cfg, cfg.d_model, cfg.n_kv_heads * cfg.d_head)
        self.wv = make_linear(cfg, cfg.d_model, cfg.n_kv_heads * cfg.d_head)
        self.wo = make_linear(cfg, cfg.n_heads * cfg.d_head, cfg.d_model)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        cfg = self.cfg
        q = self.wq(x).view(B, T, cfg.n_heads, cfg.d_head)
        k = self.wk(x).view(B, T, cfg.n_kv_heads, cfg.d_head)
        v = self.wv(x).view(B, T, cfg.n_kv_heads, cfg.d_head)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        g = cfg.n_heads // cfg.n_kv_heads                # GQA groups
        k = k.repeat_interleave(g, dim=2)
        v = v.repeat_interleave(g, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B,H,T,dh]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, cfg.n_heads * cfg.d_head)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = make_linear(cfg, cfg.d_model, cfg.d_ff)
        self.up = make_linear(cfg, cfg.d_model, cfg.d_ff)
        self.down = make_linear(cfg, cfg.d_ff, cfg.d_model)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.emb.weight
        cos, sin = rope_tables(cfg.d_head, cfg.context, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        self.grad_checkpoint = False

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def param_count(self) -> int:
        seen, n = set(), 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
        return n

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    b, x, self.cos, self.sin, use_reentrant=False)
            else:
                x = b(x, self.cos, self.sin)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new: int = 24, temperature: float = 0.0):
        self.eval()
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.context:])
            logits = logits[:, -1, :]
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def save_config(cfg: ModelConfig, path: str):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(asdict(cfg), f, indent=2)


if __name__ == "__main__":
    import sys
    cfg = ModelConfig.from_yaml(sys.argv[1]) if len(sys.argv) > 1 else ModelConfig()
    m = Model(cfg)
    print(cfg.describe())
    print(f"parameters: {m.param_count()/1e6:.2f}M")
