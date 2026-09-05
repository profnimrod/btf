"""Sequence packing with segment ids and response-only loss masks (Ch. 11)."""
from __future__ import annotations


def pack_examples(examples, seq_len: int, eot_id: int = 0):
    ids, seg = [], []
    s = 0
    for prompt, resp in examples:
        toks = list(prompt) + list(resp) + [eot_id]
        if len(ids) + len(toks) > seq_len:
            break
        ids.extend(toks)
        seg.extend([s] * len(toks))
        s += 1
    n_real = len(ids)
    ids += [eot_id] * (seq_len - len(ids))
    seg += [-1] * (seq_len - len(seg))
    return ids, seg, n_real


def loss_mask(seg, prompt_lens, resp_lens, n_real=None):
    """1 for response tokens only; 0 for prompt tokens, padding, EOT.

    Only the examples that actually fit (bounded by n_real) are scored.
    """
    mask = [0] * len(seg)
    pos = 0
    limit = n_real if n_real is not None else len(seg)
    for pl, rl in zip(prompt_lens, resp_lens):
        if pos + pl + rl + 1 > limit:      # this example did not fully fit
            break
        pos += pl                          # skip prompt
        for j in range(rl):                # score response tokens
            mask[pos + j] = 1
        pos += rl + 1                      # skip response + eot
    return mask
