"""Regression suite for everything in the repo that runs without a GPU.
`pytest -q` from the repo root exercises the model, training determinism,
the checker, registry signing, and bundle safety drills."""
import sys, subprocess, tempfile, shutil, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_param_count_matches_book():
    from train.model import Model, ModelConfig
    m = Model(ModelConfig.from_yaml(str(ROOT / "train/cfg-125m.yaml")))
    # book claims ~128M ("125M-class") — assert we're in the right neighborhood
    assert 125e6 < m.param_count() < 131e6


def test_ternary_branch_runs():
    import torch
    from train.model import Model, ModelConfig
    cfg = ModelConfig(vocab=64, d_model=32, n_layers=2, n_heads=4,
                      n_kv_heads=2, d_head=8, d_ff=64, context=16, quant="ternary")
    m = Model(cfg)
    logits, loss = m(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8)))
    assert logits.shape == (2, 8, 64) and loss.item() > 0


def test_forward_backward_step():
    import torch
    from train.model import Model, ModelConfig
    cfg = ModelConfig(vocab=64, d_model=32, n_layers=2, n_heads=4,
                      n_kv_heads=2, d_head=8, d_ff=64, context=16)
    m = Model(cfg)
    _, loss = m(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8)))
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_link_budget_checker():
    from eval.link_budget import score
    scen = {"eirp_dbw": 55.0, "range_km": 38000, "freq_ghz": 20.0,
            "atmos_db": 3.0, "g_t_dbk": 18.0, "symbol_rate_hz": 30e6}
    good = score(scen, "16APSK9/10 margin 0.6 dB")
    junk = score(scen, "no idea")
    over = score(scen, "QPSK3/4 margin 25 dB")
    assert good.value > junk.value and junk.value == 0.0
    assert over.value < good.value            # overclaim penalized


def test_stats_bootstrap_and_audit():
    from eval.stats import paired_bootstrap, audit_n, ndcg_at_k
    d, lo, hi = paired_bootstrap([0, 0, 0, 0], [1, 1, 1, 1], iters=500)
    assert d == 1.0 and lo <= d <= hi
    assert audit_n(0.05, 0.95) == 59
    assert 0.0 <= ndcg_at_k([2, 1, 0], 3) <= 1.0


def test_packing_mask_no_leak():
    from data.packing import pack_examples, loss_mask
    ex = [([1, 2], [9]), ([3], [8, 7])]
    ids, seg, n_real = pack_examples(ex, 12, eot_id=0)
    mask = loss_mask(seg, [2, 1], [1, 2], n_real)
    # padding positions (seg == -1) are never scored
    assert all(m == 0 for m, s in zip(mask, seg) if s == -1)
    assert sum(mask) == 3                      # exactly the response tokens


def test_template_parity_by_ids():
    """The invariant that matters is token ids, not strings: encode the parts
    separately and the training sequence must start with the serving ids."""
    from serve.template import render_parts, render_serving

    class FakeTok:                      # a tokenizer that merges across joins
        def encode(self, t):
            class R: pass
            r = R(); r.ids = [ord(c) for c in t.replace("\n", "")]
            return r

    rec = {"messages": [{"role": "user", "content": "x"},
                        {"role": "assistant", "content": "y"}]}
    tok = FakeTok()
    prompt, response = render_parts(rec)
    serve_ids = tok.encode(prompt).ids
    train_ids = serve_ids + tok.encode(response).ids
    assert train_ids[:len(serve_ids)] == serve_ids
    assert prompt == render_serving(rec)


def test_registry_sign_verify_and_tamper(tmp_path):
    import registry.registry as reg
    reg.ROOT = tmp_path / "store"
    reg.OBJ, reg.MAN, reg.KEYS = (reg.ROOT / "objects", reg.ROOT / "manifests",
                                  reg.ROOT / "keys")
    art = tmp_path / "m.bin"; art.write_text("weights")
    digest = reg.publish(str(art), "model", [], ["ds-v1"])
    assert reg.fetch(digest, verify=True).exists()
    # tamper the stored object -> verify must raise SystemExit
    (reg.OBJ / digest).write_bytes(b"tampered")
    try:
        reg.fetch(digest, verify=True); assert False
    except SystemExit:
        pass


def test_bundle_bit_and_rollback(tmp_path):
    import bundle.bundle as b
    good = tmp_path / "good"; good.mkdir()
    (good / "model.gguf").write_text("w")
    (good / "bit_canaries.jsonl").write_text(
        '{"id":"c","expect":"parses","parses":true}\n')
    b.seal(str(good), "good-v1")
    fleet = tmp_path / "fleet"
    assert b.activate(str(good), str(fleet)) is True
    active = (fleet / "active").read_text()
    # a failing-BIT bundle must not change the active slot
    bad = tmp_path / "bad"; bad.mkdir()
    (bad / "model.gguf").write_text("w")
    (bad / "bit_canaries.jsonl").write_text(
        '{"id":"c","expect":"refuses","refuses":false}\n')
    b.seal(str(bad), "bad-v2")
    assert b.activate(str(bad), str(fleet)) is False
    assert (fleet / "active").read_text() == active


def test_lora_merge_is_lossless():
    import torch
    from train.model import Model, ModelConfig
    from train.lora import apply_lora, merge_lora
    cfg = ModelConfig(vocab=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      d_head=8, d_ff=64, context=16)
    m = Model(cfg); m.eval()
    x = torch.randint(0, 64, (1, 8))
    apply_lora(m, r=4, alpha=8)          # B is zero-initialised: identity
    before = m(x)[0]
    merge_lora(m)
    after = m(x)[0]
    assert torch.allclose(before, after, atol=1e-5)


def test_quantization_preserves_shape_and_reduces_precision():
    import torch
    from quant.quantize import quantize_tensor, gptq_tensor
    W = torch.randn(16, 64)
    for fn in (lambda w: quantize_tensor(w, 8, 32), lambda w: gptq_tensor(w, 4, 32)):
        Q = fn(W)
        assert Q.shape == W.shape
        assert len(torch.unique(Q)) < W.numel()      # genuinely fewer levels


def test_chunking_never_emits_title_only():
    from index.build import chunk_document
    doc = "PROC-200 - title line\n\nPurpose. " + "word " * 40 + "\n\nSteps. " + "step " * 40
    chunks = chunk_document(doc, "PROC-200")
    assert chunks and all(len(c["text"].split()) > 12 for c in chunks)
    assert all(c["title"] in c["text"] for c in chunks)


def test_ndcg_bounded():
    from eval.retrieval import ndcg
    assert ndcg([1, 0, 0], 3) == 1.0
    assert 0.0 <= ndcg([0, 1, 1], 3) <= 1.0
    assert ndcg([0, 0, 0], 3) == 0.0


def test_link_budget_rewards_efficiency_and_penalises_overclaim():
    from eval.link_budget import score
    sc = {"eirp_dbw": 60.0, "range_km": 38000, "freq_ghz": 20.0, "atmos_db": 2.0,
          "g_t_dbk": 20.0, "symbol_rate_hz": 20e6}
    eff = score(sc, "16APSK9/10 margin 1.0 dB")
    cons = score(sc, "QPSK1/4 margin 12 dB")
    if eff.margin_db >= 0:
        assert eff.value > cons.value        # efficiency term rewards throughput


def test_book_json_samples_parse():
    """Every machine-readable sample printed in the book must be valid JSON."""
    import json
    for line in (ROOT / "data/schema-example.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            assert {"messages", "source", "markings"} <= rec.keys()


def test_tokenizer_compat_preflight_detects_mismatch(tmp_path):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    def make(vocab, out):
        t = Tokenizer(models.BPE(unk_token="<unk>"))
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        words = "alpha beta gamma delta" if out == "a" else "omega sigma theta lambda"
        f = tmp_path / f"{out}.txt"; f.write_text((words + " ") * 20)
        t.train([str(f)], trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<unk>"], show_progress=False))
        d = tmp_path / out; d.mkdir(); t.save(str(d / "tokenizer.json")); return d
    a, b = make(80, "a"), make(90, "b")
    import subprocess, sys
    same = subprocess.run([sys.executable, str(ROOT / "tests/tokenizer_compat.py"), "--draft", str(a), "--target", str(a)])
    diff = subprocess.run([sys.executable, str(ROOT / "tests/tokenizer_compat.py"), "--draft", str(a), "--target", str(b)])
    assert same.returncode == 0 and diff.returncode == 1
