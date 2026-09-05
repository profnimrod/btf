"""Generate the synthetic SATCOM proxy corpus and every derived dataset the
labs consume. Deterministic (seeded) so results are reproducible.

Writes:
  data/samples/<type>/*.txt      source documents for corpus/build.py
  data/sft-v1/train.jsonl        instruction pairs (Lab C)
  data/pref-v1/train.jsonl       preference comparisons (Lab C)
  data/lb-prompts.jsonl          link-budget scenarios (Lab C GRPO)
  eval/JUDGED-v1/queries.jsonl   judged retrieval set (Lab D)
  eval/domain-terms.txt          term list for tokenizer fragmentation
  eval/canary-prompts.jsonl      heartbeat prompts (Lab B)
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

BANDS = {"C": 6.0, "X": 8.0, "Ku": 14.0, "Ka": 30.0}
MODCODS = ["QPSK1/4", "QPSK1/2", "QPSK3/4", "QPSK9/10", "8PSK3/4",
           "8PSK9/10", "16APSK3/4", "16APSK9/10", "32APSK9/10"]
SUBSYS = ["EPS", "TTC", "PAYLOAD", "ADCS", "THERMAL", "PROP"]
CHANNELS = {
    "EPS": ["bus_voltage_v", "array_current_a", "battery_soc_pct", "shunt_temp_c"],
    "TTC": ["uplink_agc_dbm", "downlink_pwr_dbw", "cmd_reject_count", "sync_loss_count"],
    "PAYLOAD": ["twta_helix_current_ma", "carrier_cn0_dbhz", "gain_step", "input_bo_db"],
    "ADCS": ["wheel_rpm", "quat_err_mdeg", "sun_angle_deg", "gyro_bias_dps"],
    "THERMAL": ["radiator_temp_c", "heater_duty_pct", "panel_delta_t_c", "louver_pos_pct"],
    "PROP": ["tank_press_kpa", "thruster_on_s", "cat_bed_temp_c", "dv_applied_ms"],
}
REGIMES = ["eclipse entry", "eclipse exit", "full sun", "station keeping",
           "payload peak load", "safe mode recovery"]


def w(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def gen_standards(rng, out, n):
    facts = []
    for i in range(n):
        sid = f"STD-{700 + i}"
        band = rng.choice(list(BANDS))
        rate = rng.choice([2, 5, 10, 20, 45])
        clause = rng.randint(3, 9)
        title = f"{sid} Section {clause}: {band}-band carrier establishment"
        body = (
            f"{title}\n\n"
            f"{clause}.1 Scope. This section specifies the procedure for "
            f"establishing a {band}-band forward carrier at a symbol rate of "
            f"{rate} Msps against a geostationary transponder.\n\n"
            f"{clause}.2 Nominal parameters. The reference uplink frequency is "
            f"{BANDS[band]:.1f} GHz. The default modulation and coding scheme "
            f"is {rng.choice(MODCODS)}. Received Es/N0 shall be measured at the "
            f"demodulator input over a {rng.choice([10, 30, 60])} second window.\n\n"
            f"{clause}.3 Acquisition. The terminal shall sweep +/- "
            f"{rng.choice([25, 50, 100])} kHz about the assigned centre "
            f"frequency. Lock shall be declared when the frame synchroniser "
            f"reports {rng.choice([3, 5, 8])} consecutive valid frames.\n\n"
            f"{clause}.4 Degradation. If measured margin falls below "
            f"{rng.choice([0.5, 1.0, 1.5])} dB for more than "
            f"{rng.choice([15, 30, 45])} seconds, the network control element "
            f"shall command a step to a lower-order MODCOD.\n"
        )
        w(out / "standards" / f"{sid}.txt", body)
        facts.append((f"{sid}", f"What symbol rate does {sid} specify for "
                      f"{band}-band carrier establishment?", f"{rate} Msps"))
    return facts


def gen_requirements(rng, out, n):
    facts = []
    for i in range(n):
        rid = f"REQ-SC-{100 + i}"
        sub = rng.choice(SUBSYS)
        val = rng.choice([95, 98, 99, 99.5])
        dur = rng.choice([24, 48, 72, 168])
        body = (
            f"{rid} — {sub} availability requirement\n\n"
            f"Requirement. The {sub} subsystem shall maintain nominal "
            f"operation for not less than {val} percent of any {dur} hour "
            f"period, excluding scheduled maintenance windows.\n\n"
            f"Rationale. Derived from mission availability allocation in "
            f"REQ-SC-{100 + (i * 7) % n}. Verification is by analysis and by "
            f"{rng.choice(['test', 'demonstration', 'inspection'])}.\n\n"
            f"Traces to. {rng.choice(['STD-701', 'STD-703', 'STD-705'])}; "
            f"procedure PROC-{200 + (i * 3) % 40}.\n"
        )
        w(out / "requirements" / f"{rid}.txt", body)
        facts.append((rid, f"What availability does {rid} require?",
                      f"{val} percent over {dur} hours"))
    return facts


def gen_procedures(rng, out, n):
    facts = []
    for i in range(n):
        pid = f"PROC-{200 + i}"
        sub = rng.choice(SUBSYS)
        act = rng.choice(["interference response", "safe mode recovery",
                          "carrier reacquisition", "MODCOD step-down",
                          "thermal excursion response"])
        limit = rng.choice([3, 5, 10, 15])
        body = (
            f"{pid} — {act} ({sub})\n\n"
            f"Purpose. This procedure governs operator response to {act} "
            f"events affecting the {sub} subsystem.\n\n"
            f"Preconditions. Operator holds current certification. The "
            f"affected carrier is identified and logged.\n\n"
            f"Steps.\n"
            f"1. Confirm the anomaly against at least two independent "
            f"telemetry channels.\n"
            f"2. Notify the network controller within {limit} minutes of "
            f"confirmation.\n"
            f"3. Record the pre-event configuration before any change.\n"
            f"4. Apply the corrective action authorised for the current "
            f"authority tier; do not exceed the written envelope.\n"
            f"5. Verify recovery over a {rng.choice([5, 10, 20])} minute "
            f"observation window and file the disposition report.\n\n"
            f"Escalation. If recovery is not confirmed, escalate to "
            f"engineering support and reference {pid}.\n"
        )
        w(out / "procedures" / f"{pid}.txt", body)
        facts.append((pid, f"Within how many minutes must the network "
                      f"controller be notified under {pid}?",
                      f"{limit} minutes"))
    return facts


def gen_telemetry(rng, out, n):
    facts = []
    for i in range(n):
        tid = f"TLM-{300 + i}"
        sub = rng.choice(SUBSYS)
        ch = rng.choice(CHANNELS[sub])
        reg = rng.choice(REGIMES)
        base = round(rng.uniform(10, 90), 2)
        dev = round(rng.uniform(2, 12), 2)
        lines = []
        for t in range(8):
            v = base + rng.gauss(0, 1.2) + (dev if t == 5 else 0)
            lines.append(f"  t+{t * 30:04d}s  {ch} = {v:8.2f}")
        body = (
            f"{tid} — {sub} telemetry window during {reg}\n\n"
            f"Window summary. Channel {ch} on subsystem {sub} sampled at 30 "
            f"second cadence during {reg}. Nominal band for this regime is "
            f"{base - 4:.2f} to {base + 4:.2f}.\n\n"
            + "\n".join(lines) + "\n\n"
            f"Assessment. A single-sample excursion of {dev:.2f} units above "
            f"the regime baseline was observed at t+0150s and returned to "
            f"nominal on the following sample. Screener disposition: "
            f"{'promote for review' if dev > 7 else 'no action'}.\n"
        )
        w(out / "telemetry" / f"{tid}.txt", body)
        facts.append((tid, f"Which channel does {tid} report?", ch))
    return facts


def gen_tickets(rng, out, n):
    facts, qa = [], []
    for i in range(n):
        kid = f"TKT-{400 + i}"
        sub = rng.choice(SUBSYS)
        sym = rng.choice(["intermittent carrier loss", "elevated command rejects",
                          "unexplained gain step", "margin degradation",
                          "sync loss during handover"])
        cause = rng.choice(["adjacent-satellite interference",
                            "uplink power control misconfiguration",
                            "terminal LNB drift", "rain fade beyond design margin",
                            "stale MODCOD table on the network controller"])
        fix = rng.choice(["applied a one-step MODCOD reduction and revalidated margin",
                          "corrected the uplink power control setpoint",
                          "replaced the terminal LNB and re-peaked the antenna",
                          "restored the MODCOD table from the configuration baseline",
                          "coordinated a frequency offset with the affected operator"])
        body = (
            f"{kid} — {sym} on {sub}\n\n"
            f"Reported symptom. Operators observed {sym} affecting the "
            f"assigned carrier over a {rng.choice([20, 45, 90])} minute period.\n\n"
            f"Investigation. Telemetry review and link analysis identified "
            f"{cause} as the proximate cause. Adjacent channels were checked "
            f"and found nominal.\n\n"
            f"Resolution. Engineering {fix}. Margin recovered to within "
            f"{rng.choice([0.4, 0.8, 1.2])} dB of the predicted value.\n\n"
            f"Follow-up. Reference procedure PROC-{200 + (i * 5) % 40}.\n"
        )
        w(out / "tickets" / f"{kid}.txt", body)
        facts.append((kid, f"What was the proximate cause in {kid}?", cause))
        qa.append((f"What caused {sym} in ticket {kid}, and how was it resolved?",
                   f"Investigation identified {cause}. Engineering {fix}. "
                   f"Source: {kid}."))
    return facts, qa


def gen_linkbudgets(rng, out, n):
    facts, scenarios = [], []
    for i in range(n):
        lid = f"LNK-{500 + i}"
        band = rng.choice(list(BANDS))
        scen = {
            "eirp_dbw": round(rng.uniform(48, 62), 1),
            "range_km": rng.choice([35786, 37000, 38500, 40000]),
            "freq_ghz": BANDS[band],
            "atmos_db": round(rng.uniform(0.5, 6.0), 1),
            "g_t_dbk": round(rng.uniform(10, 25), 1),
            "symbol_rate_hz": rng.choice([5e6, 10e6, 20e6, 30e6]),
        }
        body = (
            f"{lid} — {band}-band link budget worksheet\n\n"
            f"Transmit EIRP: {scen['eirp_dbw']} dBW\n"
            f"Slant range: {scen['range_km']} km\n"
            f"Frequency: {scen['freq_ghz']} GHz\n"
            f"Atmospheric loss: {scen['atmos_db']} dB\n"
            f"Receive G/T: {scen['g_t_dbk']} dB/K\n"
            f"Symbol rate: {scen['symbol_rate_hz'] / 1e6:.0f} Msps\n\n"
            f"Method. Compute free space path loss, subtract atmospheric "
            f"loss, add receive G/T and the Boltzmann constant term, then "
            f"convert carrier-to-noise density to Es/N0 at the given symbol "
            f"rate. Compare against the required Es/N0 for the candidate "
            f"MODCOD and report the margin.\n"
        )
        w(out / "linkbudget" / f"{lid}.txt", body)
        facts.append((lid, f"What is the slant range used in {lid}?",
                      f"{scen['range_km']} km"))
        scenarios.append({"id": lid, "scenario": scen})
    return facts, scenarios


GENERAL = (
    "The committee met on Tuesday to review the annual budget and to discuss "
    "the schedule for the coming year. Several members asked whether the "
    "proposal would affect ordinary operations, and the chair explained that "
    "the changes were modest. A short summary of the discussion was circulated "
    "afterwards so that everyone could read the details at their own pace. "
    "Weather in the region had been unusually mild, which made travel easier "
    "for those attending in person. The next meeting will be held in the same "
    "building, and refreshments will be provided as usual. "
)


def gen_general(rng, out, n):
    """Generic English prose: the replay slice, and the reference corpus for
    the tokenizer fragmentation comparison (Ch. 6)."""
    words = GENERAL.split()
    for i in range(n):
        rng.shuffle(words)
        body = " ".join(words[j % len(words)] for j in range(rng.randint(200, 320)))
        w(out / "general" / f"GEN-{600 + i}.txt", f"General note {600 + i}\n\n{body}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--scale", default="small", choices=["small", "full"])
    ap.add_argument("--out", default="data/samples")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    n = {"small": 1, "full": 3}[a.scale]

    f1 = gen_standards(rng, out, 12 * n)
    f2 = gen_requirements(rng, out, 30 * n)
    f3 = gen_procedures(rng, out, 40 * n)
    f4 = gen_telemetry(rng, out, 40 * n)
    f5, ticket_qa = gen_tickets(rng, out, 40 * n)
    f6, scenarios = gen_linkbudgets(rng, out, 25 * n)
    gen_general(rng, out, 30 * n)
    facts = f1 + f2 + f3 + f4 + f5 + f6

    # ---- question phrasings, then a DISJOINT train/eval split by document.
    # Operational Q&A history becomes training pairs; a held-out slice is
    # quarantined as the judged set (Chs. 9, 13, 15).
    def phrasings(q, doc_id):
        stem = q.rstrip("?")
        return [q,
                f"{stem} according to {doc_id}?",
                f"looking at {doc_id}, {stem[0].lower() + stem[1:]}?"]

    rng.shuffle(facts)
    cut = int(0.55 * len(facts))
    train_facts, eval_facts = facts[:cut], facts[cut:]

    jd = Path("eval/JUDGED-v1")
    jd.mkdir(parents=True, exist_ok=True)
    with (jd / "queries.jsonl").open("w") as f:
        for doc_id, q, ans in eval_facts:
            f.write(json.dumps({"query": q, "gold": doc_id, "answer": ans}) + "\n")

    # Q&A history: query-style pairs the miner will pick up (never eval docs)
    with open("data/qa-history.jsonl", "w") as f:
        for doc_id, q, ans in train_facts:
            for ph in phrasings(q, doc_id):
                f.write(json.dumps({"query": ph, "doc_id": doc_id,
                                    "answer": ans}) + "\n")

    # ---- SFT instruction pairs
    sft = Path("data/sft-v1")
    sft.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc_id, q, ans in train_facts:
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"{ans} (source: {doc_id})"}]})
    for q, ansr in ticket_qa:
        rows.append({"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": ansr}]})
    for s in scenarios:
        sc = s["scenario"]
        rows.append({"messages": [
            {"role": "user", "content":
                f"Recommend a MODCOD and report margin. EIRP {sc['eirp_dbw']} dBW, "
                f"range {sc['range_km']} km, freq {sc['freq_ghz']} GHz, atmos "
                f"{sc['atmos_db']} dB, G/T {sc['g_t_dbk']} dB/K, symbol rate "
                f"{sc['symbol_rate_hz'] / 1e6:.0f} Msps."},
            {"role": "assistant", "content":
                f"MODCOD {rng.choice(MODCODS[:5])} margin "
                f"{rng.uniform(0.5, 4):.1f} dB (source: {s['id']})"}]})
    rng.shuffle(rows)
    with (sft / "train.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- preference pairs: chosen cites and hedges; rejected overclaims
    pref = Path("data/pref-v1")
    pref.mkdir(parents=True, exist_ok=True)
    with (pref / "train.jsonl").open("w") as f:
        for doc_id, q, ans in train_facts:
            f.write(json.dumps({
                "prompt": q,
                "chosen": f"{ans} (source: {doc_id})",
                "rejected": rng.choice([
                    f"{ans}.",                       # no citation
                    "That is not covered in the provided material.",
                    f"Definitely {ans}, no need to check the source.",
                ])}) + "\n")

    # ---- GRPO scenarios
    with open("data/lb-prompts.jsonl", "w") as f:
        for s in scenarios:
            f.write(json.dumps(s) + "\n")

    # ---- domain term list + canaries
    terms = sorted({t for t in
                    [m for m in MODCODS] + list(CHANNELS["EPS"]) +
                    list(CHANNELS["TTC"]) + list(CHANNELS["PAYLOAD"]) +
                    ["geostationary transponder", "symbol rate", "Es/N0",
                     "frame synchroniser", "MODCOD step-down", "slant range",
                     "carrier-to-noise density", "uplink power control",
                     "adjacent-satellite interference", "rain fade"]})
    Path("eval/domain-terms.txt").write_text("\n".join(terms) + "\n")
    with open("eval/canary-prompts.jsonl", "w") as f:
        for p in ["The link budget", "MODCOD step-down", "PROC-201 states",
                  "Telemetry channel"]:
            f.write(json.dumps({"prompt": p}) + "\n")

    n_docs = sum(1 for _ in out.rglob("*.txt"))
    print(f"[synth] {n_docs} documents across {len(list(out.iterdir()))} types")
    print(f"[synth] judged queries: {len(eval_facts)} (train facts: {len(train_facts)})  sft rows: {len(rows)}  "
          f"pref rows: {len(facts)}  lb scenarios: {len(scenarios)}")


if __name__ == "__main__":
    main()
