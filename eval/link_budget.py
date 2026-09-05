"""Deterministic satellite link-budget checker — the RLVR reward source
of Ch. 12 and the self-verification tool of Ch. 18/Lab C.

score(prompt, response) -> Reward: recompute margin from scenario,
grade against the recommended MODCOD's required Eb/N0, gate unparseable
answers, penalize margin overclaim, reward throughput efficiency.
"""
from __future__ import annotations
import json, math, re
from dataclasses import dataclass

# DVB-S2X-style required Es/N0 (dB) at reference rate — illustrative table.
MODCOD_REQ = {
    "QPSK1/4": -2.35, "QPSK1/2": 1.00, "QPSK3/4": 4.03, "QPSK9/10": 6.42,
    "8PSK3/4": 7.91, "8PSK9/10": 10.98, "16APSK3/4": 10.21,
    "16APSK9/10": 13.13, "32APSK9/10": 16.05,
}
MODCOD_BPS = {  # spectral efficiency (bits/sym), for the efficiency term
    "QPSK1/4": 0.49, "QPSK1/2": 0.99, "QPSK3/4": 1.49, "QPSK9/10": 1.79,
    "8PSK3/4": 2.23, "8PSK9/10": 2.68, "16APSK3/4": 2.97,
    "16APSK9/10": 3.57, "32APSK9/10": 4.45,
}


@dataclass
class Reward:
    value: float
    parsed: bool
    margin_db: float
    detail: str


def compute_margin(s: dict, modcod: str) -> float:
    """CN0 = EIRP - FSPL - atmos + G/T + 228.6 ; margin = Es/N0 - required."""
    fspl = (92.45 + 20 * math.log10(max(s["range_km"], 1e-6))
            + 20 * math.log10(max(s["freq_ghz"], 1e-6)))
    cn0 = (s["eirp_dbw"] - fspl - s.get("atmos_db", 0.0)
           + s["g_t_dbk"] + 228.6)
    esn0 = cn0 - 10 * math.log10(s["symbol_rate_hz"])
    return esn0 - MODCOD_REQ[modcod]


def parse_response(text: str):
    mc = re.search(r"(QPSK|8PSK|16APSK|32APSK)\s*(\d/\d+)", text.replace(" ", ""))
    mg = re.search(r"margin[^-\d]*(-?\d+(?:\.\d+)?)", text, re.I)
    modcod = (mc.group(1) + mc.group(2)) if mc else None
    claimed = float(mg.group(1)) if mg else None
    return modcod, claimed


def score(prompt, response) -> Reward:
    scen = prompt if isinstance(prompt, dict) else json.loads(prompt)
    modcod, claimed = parse_response(response)
    if modcod is None or modcod not in MODCOD_REQ:
        return Reward(0.0, False, float("nan"), "format gate: no valid MODCOD")
    true_margin = compute_margin(scen, modcod)
    if true_margin < 0:
        return Reward(0.05, True, true_margin,
                      f"{modcod} does not close (margin {true_margin:.2f} dB)")
    # graded credit: closes -> base; efficiency term rewards throughput;
    # overclaim penalty if the response inflates the margin.
    base = 0.6
    eff = 0.4 * (MODCOD_BPS[modcod] / max(MODCOD_BPS.values()))
    penalty = 0.0
    if claimed is not None and claimed - true_margin > 1.0:
        penalty = min(0.5, 0.2 * (claimed - true_margin))
    val = max(0.0, base + eff - penalty)
    return Reward(round(val, 4), True, round(true_margin, 3),
                  f"{modcod} closes; margin {true_margin:.2f} dB; "
                  f"eff {eff:.2f}; overclaim penalty {penalty:.2f}")


if __name__ == "__main__":
    demo = {"eirp_dbw": 55.0, "range_km": 38000, "freq_ghz": 20.0,
            "atmos_db": 3.0, "g_t_dbk": 18.0, "symbol_rate_hz": 30e6}
    for r in ["Recommend QPSK3/4, margin 2.1 dB",
              "Use 16APSK9/10 with margin 1.0 dB",
              "MODCOD QPSK3/4 margin 25 dB",
              "let's just wing it"]:
        out = score(demo, r)
        print(f"{out.value:5.2f}  parsed={out.parsed}  {out.detail}")
