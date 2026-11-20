"""Tests for Phase X4 sanity checks.

    env_win/Scripts/python.exe src/publication/explainability/test_sanity_checks.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import x4_sanity_checks as SC                                       # noqa: E402
import cnn_attribution as CA                                     # noqa: E402

FROZEN = {
    "src/fusion_v2.py": "dd1a1180459b9eb1a3e1a02716d0406722f97d2e0eb1b9922ac551d0cdba1292",
    "src/train_ablation.py": "ed8bf84c4953a3c4bc4ac035784241d022316f04e4d0f6cd295c98c1126ab15c",
}
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
    print("=" * 78 + "\n  PHASE X4 TESTS\n" + "=" * 78)
    for rel, want in FROZEN.items():
        check("t01_frozen_%s" % os.path.basename(rel), sha(ROOT / rel) == want)

    # thresholds must be the prespecified ones
    check("t02_determinism_tol_exact", SC.DET_TOL == 0.0)
    check("t03_batch_rel_tol", SC.BATCH_REL_TOL == 1e-4)
    check("t04_equiv_tol_preapproved", SC.EQUIV_TOL == 1e-5)
    check("t05_grids", SC.CNN_GRID == 7 and SC.VIT_GRID == 14)
    check("t06_sample_size", SC.SAMPLE_PER_CLASS * 4 == 16)

    for m in ("x4_sanity_checks.py", "cnn_attribution.py", "vit_attribution.py"):
        g = SC.source_gate(HERE / m)
        check("t07_source_gate_%s" % m, all(g.values()), str(g))
    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "b.py")
        open(bad, "w").write("m.set_attn_implementation('eager')\nm(x, output_attentions=True)\n")
        check("t08_source_gate_can_fail", not any(SC.source_gate(bad).values()))

    # --- orientation primitives -------------------------------------------
    v = SC.vit_raster_order_ok()
    check("t09_vit_raster_row_major", v["vit_reshape_row_major"],
          "m[0,1]=%.0f m[1,0]=%.0f (row-major -> 1 and 14)" % (v["vit_m_0_1"], v["vit_m_1_0"]))
    check("t10_vit_not_transposed", v["vit_m_0_1"] == 1.0 and v["vit_m_1_0"] == 14.0)
    c = SC.cnn_spatial_correspondence_ok()
    check("t11_cnn_peak_preserved", c["cnn_peak_preserved"],
          "corners: %s %s %s" % (c["peak_0_0"], c["peak_0_6"], c["peak_6_0"]))
    u = SC.upsample_orientation_ok()
    check("t12_upsample_orientation", u["orientation_preserved"],
          "TL=%s TR=%s BL=%s" % (u["top_left"], u["top_right"], u["bottom_left"]))
    check("t13_no_axis_transpose", u["no_axis_transpose"], "asym(1,5)->%s" % u["asym_1_5"])

    # the orientation test must be able to FAIL: feed a flipped map
    m = np.zeros((7, 7), np.float32); m[0, 0] = 1.0
    big = CA.rescale_like_library(m)
    r0, c0 = np.unravel_index(int(big.argmax()), big.shape)
    bigf = CA.rescale_like_library(np.fliplr(m))
    r1, c1 = np.unravel_index(int(bigf.argmax()), bigf.shape)
    check("t14_orientation_check_is_sensitive", (c0 < 112) and (c1 > 112),
          "(%d,%d) vs flipped (%d,%d)" % (r0, c0, r1, c1))

    # --- baselines ---------------------------------------------------------
    cp = SC.centre_prior((14, 14))
    pk = np.unravel_index(int(cp.argmax()), cp.shape)
    check("t15_centre_prior_peaks_centre", pk in ((6, 6), (6, 7), (7, 6), (7, 7)), str(pk))
    check("t16_centre_prior_deterministic",
          np.array_equal(cp, SC.centre_prior((14, 14))))
    r1 = SC.random_map(np.random.default_rng(SC.BASELINE_SEED), (7, 7))
    r2 = SC.random_map(np.random.default_rng(SC.BASELINE_SEED), (7, 7))
    check("t17_random_map_deterministic", np.array_equal(r1, r2))

    # --- similarity --------------------------------------------------------
    a = np.arange(49, dtype=np.float32).reshape(7, 7)
    s_self = SC.similarity(a, a)
    check("t18_similarity_self", abs(s_self["pearson"] - 1.0) < 1e-9
          and s_self["normalised_l1"] == 0.0)
    s_neg = SC.similarity(a, -a)
    check("t19_similarity_antiparallel", abs(s_neg["pearson"] + 1.0) < 1e-9,
          "%.4f" % s_neg["pearson"])
    s_const = SC.similarity(a, np.zeros((7, 7), np.float32))
    check("t20_similarity_constant_is_none", s_const["pearson"] is None)

    # --- sampling ----------------------------------------------------------
    import json
    man = json.loads((ROOT / SC.RECON_ROOT / "RECONSTRUCTION_MANIFEST.json")
                     .read_text(encoding="utf-8"))
    s1 = [e["filename"] for e in SC.stratified_sample(man, 4)]
    s2 = [e["filename"] for e in SC.stratified_sample(man, 4)]
    check("t21_sample_deterministic", s1 == s2, "%d files" % len(s1))
    check("t22_sample_stratified", len(s1) == 16 and len(set(s1)) == 16)
    by = {}
    for e in SC.stratified_sample(man, 4):
        by[e["manifest_class"]] = by.get(e["manifest_class"], 0) + 1
    check("t23_four_per_class", all(v == 4 for v in by.values()), str(by))

    n = len(RES) - sum(RES)
    print("\n  %d tests, %d passed, %d failed -> %s"
          % (len(RES), sum(RES), n, "PASS" if n == 0 else "FAIL"))
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
