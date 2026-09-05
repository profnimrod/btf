"""Weight quantization for the lab model (Ch. 14): groupwise round-to-
nearest, activation-aware (AWQ-style) scaling, and error-compensated
(GPTQ-style) column updates. Quantized weights are stored dequantized so
any evaluation path can load them unchanged."""
from __future__ import annotations
import torch
import torch.nn as nn


def quantize_tensor(W: torch.Tensor, bits=4, group=128, scales=None):
    """Groupwise symmetric quantization along the input dimension."""
    out, inp = W.shape
    g = min(group, inp)
    Wq = W.clone()
    qmax = 2 ** (bits - 1) - 1
    if scales is not None:
        Wq = Wq * scales.view(1, -1)          # AWQ: protect salient channels
    for s in range(0, inp, g):
        blk = Wq[:, s:s + g]
        sc = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        blk_q = torch.clamp(torch.round(blk / sc), -qmax - 1, qmax) * sc
        Wq[:, s:s + g] = blk_q
    if scales is not None:
        Wq = Wq / scales.view(1, -1)
    return Wq


def gptq_tensor(W: torch.Tensor, bits=4, group=128):
    """Error-compensated column-wise quantization: the residual from each
    column is pushed onto the columns not yet quantized."""
    out, inp = W.shape
    Wq = W.clone()
    qmax = 2 ** (bits - 1) - 1
    for s in range(0, inp, group):
        e = s + min(group, inp - s)
        blk = Wq[:, s:e].clone()
        sc = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
        err = torch.zeros(out, device=W.device)
        for j in range(blk.shape[1]):
            col = blk[:, j] + err
            q = torch.clamp(torch.round(col / sc.squeeze(1)), -qmax - 1, qmax) \
                * sc.squeeze(1)
            err = (col - q) * 0.5             # damped compensation
            blk[:, j] = q
        Wq[:, s:e] = blk
    return Wq


@torch.no_grad()
def collect_activation_scales(model, calib_ids, alpha=0.5):
    """AWQ-style: per-input-channel activation magnitude, used to rescale
    salient channels before rounding."""
    acts = {}
    hooks = []

    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach().abs().mean(dim=(0, 1))
            acts[name] = acts.get(name, 0) + x
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(mk(name)))
    for ids in calib_ids:
        model(ids)
    for h in hooks:
        h.remove()
    return {k: (v / v.mean().clamp_min(1e-8)).pow(alpha).clamp(0.2, 5.0)
            for k, v in acts.items()}


@torch.no_grad()
def quantize_model(model, bits=4, group=128, method="rtn", act_scales=None):
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "lm_head" not in name:
            W = mod.weight.data
            if method == "gptq":
                mod.weight.data = gptq_tensor(W, bits, group)
            else:
                sc = act_scales.get(name) if act_scales else None
                mod.weight.data = quantize_tensor(W, bits, group, sc)
            n += 1
    return n
