"""
Phase X2 tests.

The recorder's whole value rests on two claims: it exposes everything from one
pass, and it leaves the model exactly as it found it. Both are tested directly,
including on the exception path, which is where a context manager usually fails.

Tests that need the model load one verified checkpoint read-only on CPU and use a
tiny synthetic batch. Nothing is written anywhere.

    env_win/Scripts/python.exe src/publication/explainability/test_dual_path_hooks.py --project-root .
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import dual_path_hooks as X2
from fusion_v2_gradcam import load_verified_model

PASSED, FAILED = [], []
_MODEL = {}


def model_and_x(root):
    if "m" not in _MODEL:
        m, _d, sha = load_verified_model(root, seed=42, device="cpu")
        m.eval()
        g = torch.Generator().manual_seed(20260926)
        _MODEL["m"] = m
        _MODEL["sha"] = sha
        _MODEL["x"] = torch.rand((2, 3, 224, 224), generator=g) * 2.0 - 1.0
    return _MODEL["m"], _MODEL["x"]


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        FAILED.append((name, "assertion: %s" % e))
        print("  FAIL  %s\n          %s" % (name, e), flush=True)
    except Exception as e:
        FAILED.append((name, "%s: %s" % (type(e).__name__, e)))
        print("  ERROR %s\n          %s: %s" % (name, type(e).__name__, e), flush=True)
    else:
        PASSED.append(name)
        print("  ok    %s" % name, flush=True)


def rejects(fn, *a, **k):
    try:
        fn(*a, **k)
    except X2.X2Error as e:
        return str(e)
    raise AssertionError("accepted something it should have rejected")


ROOT = None


# ------------------------------------------------------------------ tests ---

def t01_exposes_all_groups_from_one_pass():
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x)
    s = o.shapes()
    assert s["fused_logits"] == [2, 4] and s["fused_probs"] == [2, 4], s
    assert s["aux_cnn_logits"] == [2, 4] and s["aux_vit_logits"] == [2, 4], s
    assert s["gate"] == [2, 2], s
    assert s["cnn_maps[3]"] == [2, 40, 28, 28], s
    assert s["cnn_maps[5]"] == [2, 112, 14, 14], s
    assert s["cnn_maps[8]"] == [2, 1280, 7, 7], s
    assert s["vit_attention"] == "12 x [2, 12, 197, 197]", s
    assert s["target_index"] == [2], s


def t02_gate_valid():
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x)
    g = o.gate
    assert torch.isfinite(g).all(), "gate not finite"
    assert (g >= 0).all(), "gate has negative mass"
    assert torch.allclose(g.sum(-1), torch.ones(2), atol=X2.GATE_SUM_TOL), g.sum(-1)


def t03_attention_rows_sum_to_one():
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x)
    assert len(o.vit_attention) == 12, len(o.vit_attention)
    for t in o.vit_attention:
        assert torch.isfinite(t).all()
        s = t.sum(-1)
        assert torch.allclose(s, torch.ones_like(s), atol=X2.ATTN_SUM_TOL)


def t04_instrumentation_does_not_change_predictions():
    m, x = model_and_x(ROOT)
    with torch.no_grad():
        plain = torch.softmax(m(x), -1)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x)
    d = float((o.fused_probs - plain).abs().max())
    assert d <= X2.NEUTRALITY_TOL, "instrumentation shifted probabilities by %.3e" % d
    assert torch.equal(o.fused_probs.argmax(-1), plain.argmax(-1))


def t05_restores_attention_implementation_exactly():
    m, x = model_and_x(ROOT)
    before = m.vit.config._attn_implementation
    with torch.no_grad():
        plain = torch.softmax(m(x), -1)
    with X2.DualPathRecorder(m) as rec:
        during = rec.attn_impl
        rec.run(x)
    after = m.vit.config._attn_implementation
    assert before == "sdpa", before
    assert during == "eager", during
    assert after == before, "attn impl not restored: %s -> %s" % (before, after)
    with torch.no_grad():
        again = torch.softmax(m(x), -1)
    assert torch.equal(again, plain), "restore did not reproduce exactly"


def t06_removes_hooks_on_exit():
    m, x = model_and_x(ROOT)
    rec = X2.DualPathRecorder(m)
    with rec:
        assert rec.n_hooks == 3, rec.n_hooks
        rec.run(x)
    assert rec.n_hooks == 0, "hooks retained after exit"
    for s in X2.CNN_STAGES:
        assert len(m.cnn_stages[s]._forward_hooks) == 0, \
            "module %d still carries a forward hook" % s


def t07_restores_even_when_body_raises():
    """The usual way a context manager fails: the happy path restores, the
    exception path does not."""
    m, x = model_and_x(ROOT)
    before = m.vit.config._attn_implementation
    rec = X2.DualPathRecorder(m)
    try:
        with rec:
            rec.run(x)
            raise RuntimeError("deliberate failure inside the context")
    except RuntimeError as e:
        assert "deliberate failure" in str(e), "exception was swallowed"
    else:
        raise AssertionError("context manager swallowed the exception")
    assert rec.n_hooks == 0, "hooks retained after an exception"
    assert m.vit.config._attn_implementation == before, \
        "attn impl not restored after an exception"
    for s in X2.CNN_STAGES:
        assert len(m.cnn_stages[s]._forward_hooks) == 0


def t08_no_gradients_retained():
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x)
    for name in ("fused_logits", "fused_probs", "gate", "aux_cnn_logits",
                 "aux_vit_logits", "tta_probs"):
        t = getattr(o, name)
        assert t.grad_fn is None, "%s retains a graph" % name
        assert not t.requires_grad, "%s requires grad" % name
    for s, t in o.cnn_maps.items():
        assert t.grad_fn is None and not t.requires_grad, "cnn stage %d retains grad" % s
    for t in o.vit_attention:
        assert t.grad_fn is None and not t.requires_grad, "attention retains grad"


def t09_target_index_matches_fused_argmax():
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x, target_mode="predicted")
    assert torch.equal(o.target_index, o.fused_probs.argmax(-1))
    with X2.DualPathRecorder(m) as rec:
        o2 = rec.run(x, target_mode="fixed", fixed_index=2)
    assert torch.equal(o2.target_index, torch.full((2,), 2, dtype=torch.long))


def t10_rejects_run_outside_context():
    m, x = model_and_x(ROOT)
    rec = X2.DualPathRecorder(m)
    msg = rejects(rec.run, x)
    assert "outside its context" in msg, msg


def t11_rejects_unknown_target_mode():
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        msg = rejects(rec.run, x, target_mode="whatever")
        assert "unknown target_mode" in msg, msg
        msg2 = rejects(rec.run, x, target_mode="fixed")
        assert "requires fixed_index" in msg2, msg2


def t12_validation_detects_bad_gate():
    """Positive control plus a deliberate corruption."""
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x)
    good = X2.validate_outputs(o, 2)
    assert good["gate_rows_sum_to_one"] and good["gate_finite"] and good["gate_nonnegative"]
    assert good["attention_rows_sum_to_one"] and good["cnn_stage_shapes_ok"]
    assert good["all_groups_present"] and good["no_grad_retained"]

    o.gate = o.gate * 2.0                      # rows now sum to 2
    bad = X2.validate_outputs(o, 2)
    assert not bad["gate_rows_sum_to_one"], "a gate summing to 2 passed validation"


def t13_validation_detects_bad_attention():
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x)
    o.vit_attention = tuple([t * 0.5 for t in o.vit_attention])
    bad = X2.validate_outputs(o, 2)
    assert not bad["attention_rows_sum_to_one"], "halved attention passed validation"


def t14_validation_detects_missing_attention():
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x)
    o.vit_attention = tuple()
    bad = X2.validate_outputs(o, 2)
    assert not bad["attention_shape_ok"] and not bad["attention_rows_sum_to_one"]


def t15_disabling_attention_is_detected_not_silent():
    """Under sdpa the flag returns None. The recorder must raise, not hand back
    an empty tuple that a later phase would treat as 'no attention'."""
    m, x = model_and_x(ROOT)
    rec = X2.DualPathRecorder(m, enable_vit_attention=True)
    with rec:
        # force the sdpa path back while inside the context
        m.vit.set_attn_implementation("sdpa")
        msg = rejects(rec.run, x)
        assert "attention requested but not captured" in msg, msg
        m.vit.set_attn_implementation("eager")
    assert m.vit.config._attn_implementation == "sdpa", "restore failed after the probe"


def t16_tta_probs_match_uninstrumented_tta():
    from fusion_v2 import tta_logits
    m, x = model_and_x(ROOT)
    with torch.no_grad():
        plain = tta_logits(m, x)
    with X2.DualPathRecorder(m) as rec:
        o = rec.run(x, with_tta=True)
    d = float((o.tta_probs - plain).abs().max())
    assert d <= X2.NEUTRALITY_TOL, "TTA probabilities shifted by %.3e" % d


def t17_two_runs_in_one_context_are_independent():
    """Stale captures from a previous run must not leak into the next."""
    m, x = model_and_x(ROOT)
    with X2.DualPathRecorder(m) as rec:
        a = rec.run(x)
        b = rec.run(x)
    assert torch.equal(a.fused_probs, b.fused_probs)
    assert torch.equal(a.cnn_maps[8], b.cnn_maps[8])
    assert len(b.vit_attention) == 12


def t18_tolerances_are_prespecified():
    assert X2.NEUTRALITY_TOL == 1e-6
    assert X2.GATE_SUM_TOL == 1e-6
    assert X2.ATTN_SUM_TOL == 1e-4
    assert X2.ACC_TOL_PP == 0.0
    assert X2.CNN_STAGES == (3, 5, 8)
    assert X2.CLASSES == sorted(X2.CLASSES)


def t19_writes_only_under_permitted_roots():
    src = open(HERE / "dual_path_hooks.py", encoding="utf-8").read()
    assert X2.OUT_ROOT.startswith("publication_results/explainability/x2_instrumentation")
    for line in src.splitlines():
        if "open(" in line and ('"w"' in line or "'w'" in line):
            assert "out /" in line, "unaudited write: %s" % line.strip()


def t20_raw_tensors_not_persisted():
    src = open(HERE / "dual_path_hooks.py", encoding="utf-8").read()
    for bad in ("torch.save", "np.save", "np.savez", ".npz"):
        assert bad not in src, "module persists raw tensors via %s" % bad


def t21_protected_code_unchanged():
    root = Path(ROOT)
    exp = {"src/fusion_v2.py": "dd1a1180459b9eb1",
           "src/train_ablation.py": "ed8bf84c4953a3c4"}
    for rel, pre in exp.items():
        h = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        assert h.startswith(pre), "%s changed" % rel


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()
    ROOT = str(Path(args.project_root).resolve())

    print("Phase X2 - instrumentation tests")
    print("=" * 74)
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("t") and len(k) > 1 and k[1].isdigit() and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print("=" * 74)
    print("%d passed, %d failed" % (len(PASSED), len(FAILED)))
    for n, w in FAILED:
        print("  - %s :: %s" % (n, w))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
