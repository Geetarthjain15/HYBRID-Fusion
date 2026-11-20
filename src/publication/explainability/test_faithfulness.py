"""Tests for Phase X5 faithfulness machinery.

    env_win/Scripts/python.exe src/publication/explainability/test_faithfulness.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import faithfulness as FA                                       # noqa: E402
import composite as CO                                          # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    print("=" * 78 + "\n  PHASE X5 TESTS\n" + "=" * 78)

    for rel, want in {
        "src/fusion_v2.py": "dd1a1180459b9eb1a3e1a02716d0406722f97d2e0eb1b9922ac551d0cdba1292",
        "src/train_ablation.py": "ed8bf84c4953a3c4bc4ac035784241d022316f04e4d0f6cd295c98c1126ab15c",
    }.items():
        check("t01_frozen_%s" % os.path.basename(rel), sha(ROOT / rel) == want)

    # --- the lock is the contract -----------------------------------------
    V = FA.verify_lock(ROOT)
    check("t02_protocol_lock_hash", V["protocol_lock_sha256"] == FA.PROTOCOL_LOCK_SHA256,
          V["protocol_lock_sha256"][:16])
    proto = V["protocol"]
    check("t03_methods_match_lock",
          [m["id"] for m in proto["methods"]] == FA.METHODS, str(len(FA.METHODS)))
    check("t04_steps_20", proto["deletion"]["steps"] == 20 and proto["insertion"]["steps"] == 20)
    check("t05_topk", proto["top_k_percent"] == [1, 5, 10, 20, 50])
    check("t06_bootstrap", proto["bootstrap"]["iterations"] == 10000
          and proto["bootstrap"]["seed"] == 20260927)
    check("t07_grid", proto["common_grid"] == [14, 14] and FA.NCELL == 196)
    check("t08_mask_value_zero", proto["masking"]["mask_value_normalised_tensor"] == 0.0)

    for m in ("faithfulness.py", "composite.py"):
        g = FA.source_gate(HERE / m)
        check("t09_source_gate_%s" % m, all(g.values()), str(g))

    # --- composites --------------------------------------------------------
    c = torch.rand(14, 14); v = torch.rand(14, 14)
    e = CO.equal_weight(c, v)
    check("t10_composite_equal_range", float(e.min()) == 0.0 and abs(float(e.max()) - 1.0) < 1e-6)
    gw = CO.gate_weighted(c, v, 1.0, 0.0)
    check("t11_gate_all_cnn_recovers_cnn",
          torch.allclose(gw, CO.to_common(c), atol=1e-6),
          "max diff %.2e" % float((gw - CO.to_common(c)).abs().max()))
    gw2 = CO.gate_weighted(c, v, 0.0, 1.0)
    check("t12_gate_all_vit_recovers_vit", torch.allclose(gw2, CO.to_common(v), atol=1e-6))
    cp = CO.centre_prior()
    pk = np.unravel_index(int(cp.argmax()), cp.shape)
    check("t13_centre_prior_centred", pk in ((6, 6), (6, 7), (7, 6), (7, 7)), str(pk))
    r1 = CO.random_map(torch.Generator().manual_seed(1))
    r2 = CO.random_map(torch.Generator().manual_seed(1))
    check("t14_random_map_deterministic", torch.equal(r1, r2))

    # --- resample-then-normalise, and 7->14 --------------------------------
    small = torch.arange(49, dtype=torch.float32).reshape(7, 7)
    up = CO.to_common(small)
    check("t15_cnn_7_to_14", tuple(up.shape) == (14, 14))
    check("t16_normalised_to_unit", float(up.min()) == 0.0 and abs(float(up.max()) - 1.0) < 1e-6)
    check("t17_constant_map_becomes_zero",
          float(CO.to_common(torch.ones(14, 14)).abs().max()) == 0.0)

    # --- masking geometry --------------------------------------------------
    x = torch.ones(1, 3, 224, 224)
    order = torch.arange(196)
    m0 = FA.build_masked(x, order, [0], False)[0]
    m_all = FA.build_masked(x, order, [196], False)[0]
    check("t18_deletion_0_is_unmodified", torch.equal(m0, x[0]))
    check("t19_deletion_all_is_fully_masked", float(m_all.abs().max()) == 0.0)
    i0 = FA.build_masked(x, order, [0], True)[0]
    i_all = FA.build_masked(x, order, [196], True)[0]
    check("t20_insertion_0_is_fully_masked", float(i0.abs().max()) == 0.0)
    check("t21_insertion_all_is_unmodified", torch.equal(i_all, x[0]))
    half = FA.build_masked(x, order, [98], False)[0]
    frac = float((half == 0).float().mean())
    check("t22_half_masks_exactly_half", abs(frac - 0.5) < 1e-6, "%.6f" % frac)
    one = FA.build_masked(x, order, [1], False)[0]
    check("t23_one_cell_is_16x16_block",
          int((one == 0).float().sum() / 3) == 256, str(int((one == 0).float().sum() / 3)))

    # --- AUC ---------------------------------------------------------------
    check("t24_auc_constant_one", abs(FA.curve_auc(np.ones(21)) - 1.0) < 1e-9)
    check("t25_auc_linear_down", abs(FA.curve_auc(np.linspace(1, 0, 21)) - 0.5) < 1e-9)

    # --- Spearman ----------------------------------------------------------
    a = torch.arange(196, dtype=torch.float32)
    check("t26_spearman_self", abs(FA.spearman(a, a) - 1.0) < 1e-9)
    check("t27_spearman_reversed", abs(FA.spearman(a, -a) + 1.0) < 1e-9)
    check("t28_spearman_constant_nan", np.isnan(FA.spearman(a, torch.zeros(196))))

    # --- transforms are invertible on the map ------------------------------
    g = torch.Generator().manual_seed(0)
    m = torch.rand(14, 14)
    check("t29_hflip_invertible",
          torch.allclose(FA.invert_map_transform(torch.flip(m, dims=[1]), "hflip"), m))
    x2 = torch.rand(1, 3, 224, 224)
    check("t30_hflip_transform",
          torch.equal(FA.apply_transform(x2, "hflip", g), torch.flip(x2, dims=[3])))
    for tid in ("rot_plus10", "rot_minus10", "scale_plus5", "scale_minus5", "noise"):
        y = FA.apply_transform(x2, tid, g)
        ok = y.shape == x2.shape and torch.isfinite(y).all()
        check("t31_transform_%s" % tid, ok, str(tuple(y.shape)))
    check("t32_noise_is_small",
          float((FA.apply_transform(x2, "noise", torch.Generator().manual_seed(0)) - x2)
                .abs().mean()) < 0.02)
    check("t33_unknown_transform_raises",
          _raises(lambda: FA.apply_transform(x2, "nope", g), FA.X5Error))

    # --- lock tamper detection --------------------------------------------
    check("t34_wrong_lock_hash_raises",
          _raises(lambda: _verify_with_bad_hash(), FA.X5Error))

    n = len(RES) - sum(RES)
    print("\n  %d tests, %d passed, %d failed -> %s"
          % (len(RES), sum(RES), n, "PASS" if n == 0 else "FAIL"))
    return 1 if n else 0


def _raises(fn, exc):
    try:
        fn()
        return False
    except exc:
        return True
    except Exception:
        return False


def _verify_with_bad_hash():
    real = FA.PROTOCOL_LOCK_SHA256
    FA.PROTOCOL_LOCK_SHA256 = "0" * 64
    try:
        FA.verify_lock(ROOT)
    finally:
        FA.PROTOCOL_LOCK_SHA256 = real


if __name__ == "__main__":
    sys.exit(main())
