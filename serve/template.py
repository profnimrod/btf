"""Chat template — the single source of truth for how roles and messages
become tokens, used identically by training and serving (Ch. 11)."""
from __future__ import annotations

USER, ASST, EOT = "<|user|>", "<|assistant|>", "<|eot|>"


def _fmt(messages) -> str:
    out = []
    for m in messages:
        tag = USER if m["role"] == "user" else ASST
        out.append(f"{tag}\n{m['content']}{EOT}")
    return "\n".join(out)


def render_serving(record: dict) -> str:
    """Everything up to and including the final assistant tag (no target)."""
    msgs = [m for m in record["messages"] if m["role"] != "assistant"]
    return _fmt(msgs) + f"\n{ASST}\n"


def render_training(record: dict) -> str:
    """Serving prefix + the assistant target + EOT."""
    asst = [m for m in record["messages"] if m["role"] == "assistant"][-1]
    return render_serving(record) + f"{asst['content']}{EOT}"


def render_parts(record: dict):
    """Prompt text and response text as SEPARATE strings.

    Training must encode these separately and concatenate the token ids.
    Re-tokenizing the joined string lets BPE merge across the boundary, so
    the training sequence would no longer start with the exact tokens the
    server sends at inference — the silent defect tests/template_parity.py
    exists to catch (Ch. 11).
    """
    asst = [m for m in record["messages"] if m["role"] == "assistant"][-1]
    return render_serving(record), f"{asst['content']}{EOT}"
