"""Tests for Phase X3 (amended) synchronized branch attribution.

    env_win/Scripts/python.exe src/publication/explainability/test_attribution.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_ablation as TA                                       # noqa: E402
from fusion_v2_gradcam import load_verified_model, gradcam_plus_plus  # noqa: E402
import cnn_attribution as CA                                      # noqa: E402
import vit_attribution as VA                                      # noqa: E402
import run_attribution as RA                                      # noqa: E402
import sdpa_token_attribution as X2B                              # noqa: E402

FROZEN = {
    "src/fusion_v2.py": "dd1a1180459b9eb1a3e1a02716d0406722f97d2e0eb1b9922ac551d0cdba1292",
    "src/train_ablation.py": "ed8bf84c4953a3c4bc4ac035784241d022316f04e4d0f6cd295c98c1126ab15c",
}
RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print("  [%s] %-50s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def images(n=2):
    xs = []
    for c in RA.CLASSES:
        d = ROOT / RA.RECON_ROOT / c
        xs.append(TA.eval_transform(Image.open(d / sorted(os.listdir(d))[0]).convert("RGB")))
        if len(xs) == n:
            break
    return torch.stack(xs)


def main():
    print("=" * 78 + "\n  PHASE X3 TESTS\n" + "=" * 78)
    for rel, want in FROZEN.items():
        check("t01_frozen_%s" % os.path.basename(rel), sha(ROOT / rel) == want)
    for m in RA.MODULES:
        g = RA.source_gate(HERE / m)
        check("t02_source_gate_%s" % m, all(g.values()), str(g))

    with tempfile.TemporaryDirectory() as td:
        bad = os.path.join(td, "b.py")
        open(bad, "w").write("m.set_attn_implementation('eager')\nm(x, output_attentions=True)\n")
        g = RA.source_gate(bad)
    check("t03_source_gate_can_fail", not any(g.values()))

    # Grad-CAM++ weight maths on a hand-checkable case
    a = np.ones((1, 1, 2, 2), np.float32); gr = np.full((1, 1, 2, 2), 0.5, np.float32)
    g2, g3 = 0.25, 0.125
    aij = g2 / (2 * g2 + 4 * g3 + CA.EPS)
    w = CA.gradcam_pp_weights(a, gr)
    check("t04_weight_formula_eq19", abs(float(w[0, 0]) - 4 * 0.5 * aij) < 1e-7,
          "%.8f" % float(w[0, 0]))
    neg = CA.gradcam_pp_native(a, -gr)
    check("t05_relu_on_cam", float(neg.min()) >= 0.0)

    model, dev, ck = load_verified_model(ROOT, seed=42, device="cpu")
    model.eval()
    sd0 = RA.state_dict_digest(model)
    x = images(2)
    with torch.no_grad():
        p_plain = model(x).softmax(-1)

    with RA.SynchronizedAttributor(model) as s:
        check("t06_two_hooks", s.n_hooks == 2)
        o = s.attribute(x)
        check("t07_sdpa_during", s.attn_impl() == "sdpa")
    check("t08_hooks_removed", s.n_hooks == 0)
    check("t09_cnn_native_7x7", o["cnn_maps"].shape == (2, 7, 7), str(o["cnn_maps"].shape))
    check("t10_cnn_act_is_stage8", o["cnn_act_shape"] == (2, 1280, 7, 7), str(o["cnn_act_shape"]))
    check("t11_vit_14x14", o["vit_maps"].shape == (2, 14, 14))
    check("t12_maps_finite_nonconstant",
          all(np.isfinite(m).all() and m.max() > m.min()
              for m in list(o["cnn_maps"]) + list(o["vit_maps"])))
    d = float((o["probs"] - p_plain).abs().max())
    check("t13_neutral_exact_zero", d == 0.0, "%.3e" % d)
    check("t14_param_grads_none", all(p.grad is None for p in model.parameters()))

    # ViT map must equal the verified X2b method bit-for-bit
    with X2B.TokenAttributor(model) as t:
        ref = t.attribute(x, target_index=o["target_index"])["maps"].numpy()
    check("t15_vit_identical_to_x2b", np.array_equal(ref, o["vit_maps"]),
          "max diff %.3e" % float(np.abs(ref - o["vit_maps"]).max()))

    # Native CNN map vs existing verified helper (amendment §3)
    diffs = []
    for k in range(2):
        helper = gradcam_plus_plus(model, x[k:k + 1], dev, int(o["target_index"][k]))
        diffs.append(float(np.abs(CA.rescale_like_library(o["cnn_maps"][k]) - helper).max()))
    model.zero_grad(set_to_none=True)
    check("t16_gradcam_equivalence_1e-5", max(diffs) <= RA.EQUIV_TOL,
          "max %.3e" % max(diffs))

    # Synchronization: both maps respond to the target
    with RA.SynchronizedAttributor(model) as s:
        o0 = s.attribute(x, target_index=torch.tensor([0, 0]))
        o3 = s.attribute(x, target_index=torch.tensor([3, 3]))
    check("t17_target_changes_both_maps",
          np.abs(o0["vit_maps"] - o3["vit_maps"]).max() > 0
          and np.abs(o0["cnn_maps"] - o3["cnn_maps"]).max() > 0)

    b = RA.SynchronizedAttributor(model)
    try:
        with b:
            raise RuntimeError("x")
    except RuntimeError:
        pass
    check("t18_hooks_removed_on_raise", b.n_hooks == 0)
    try:
        RA.SynchronizedAttributor(model).attribute(x)
        outside = False
    except RA.X3Error:
        outside = True
    check("t19_refuses_outside_context", outside)
    check("t20_state_dict_unchanged", RA.state_dict_digest(model) == sd0)
    check("t21_sdpa_after", model.vit.config._attn_implementation == "sdpa")
    check("t22_npz_schema_has_no_composite",
          not any(k for k in RA.NPZ_KEYS if "composite" in k or "fused_map" in k)
          and RA.NPZ_KEYS == {"cnn_map", "vit_map", "target_index", "target_mode",
                              "target_class", "filename", "true_label"})

    # --- verdict / exit-code logic (regression: the prose carries a caveat
    #     suffix, so comparing it to "PASS" wrongly failed a passing run) -----
    g_skip, p_skip = RA.verdict({"a": True, "b": True, "preservation": None}, True)
    check("t36_skip_preservation_still_passes", g_skip is True, p_skip)
    check("t37_prose_carries_caveat", "NOT VERIFIED" in p_skip)
    check("t38_prose_is_not_the_verdict", p_skip != "PASS" and g_skip is True,
          "exit would be %d" % (0 if g_skip else 1))
    g_ok, p_ok = RA.verdict({"a": True, "preservation": True}, False)
    check("t39_full_pass_is_plain_PASS", g_ok is True and p_ok == "PASS")
    g_bad, p_bad = RA.verdict({"a": True, "b": False, "preservation": None}, True)
    check("t40_failed_gate_fails_verdict", g_bad is False and p_bad.startswith("FAIL"))
    g_pres, _ = RA.verdict({"a": True, "preservation": False}, False)
    check("t41_failed_preservation_fails_verdict", g_pres is False)
    check("t42_none_is_excluded_not_counted_true",
          RA.verdict({"preservation": None}, True)[0] is True
          and RA.verdict({"preservation": False}, False)[0] is False)

    # --- multi-seed reporting: equivalence is None for seeds 43/44 by design.
    #     Formatting it with %.3e crashed the grid after seed 42 completed. ----
    check("t43_fmt_equiv_handles_none", RA.fmt_equiv(None) == "n/a", RA.fmt_equiv(None))
    check("t44_fmt_equiv_formats_value", RA.fmt_equiv(6.557e-06) == "6.557e-06",
          RA.fmt_equiv(6.557e-06))

    def seedrow(sd, eq, n_eq):
        return {"seed": sd, "n_images": 848, "max_abs_prob_diff_single": 0.0,
                "max_abs_prob_diff_tta": 0.0, "pred_label_changes_vs_plain": 0,
                "pred_label_changes_vs_recorded": 0, "accuracy_diff_pp": 0.0,
                "cnn_maps_valid": 848, "vit_maps_valid": 848,
                "gradcam_equivalence_max_abs_diff": eq,
                "gradcam_equivalence_n": n_eq, "attribution_s_per_image": 0.024,
                "attn_impls_observed": ["sdpa"], "targets_synchronized": True,
                "filename_mismatches": 0, "true_label_mismatches": 0}

    per = [seedrow(42, 6.557e-06, 848), seedrow(43, None, 0), seedrow(44, None, 0)]
    # the exact format strings that crashed, over all three seeds
    try:
        for r in per:
            "| %d | %.3e | %.3e | %d/%d | %d/%d | %d/%d | %s | %.3f |" % (
                r["seed"], r["max_abs_prob_diff_single"], r["max_abs_prob_diff_tta"],
                r["pred_label_changes_vs_plain"], r["pred_label_changes_vs_recorded"],
                r["cnn_maps_valid"], r["n_images"], r["vit_maps_valid"], r["n_images"],
                RA.fmt_equiv(r["gradcam_equivalence_max_abs_diff"]),
                r["attribution_s_per_image"])
            "gradcam equiv max=%s | %.3f s/img" % (
                RA.fmt_equiv(r["gradcam_equivalence_max_abs_diff"]),
                r["attribution_s_per_image"])
        rendered = True
    except TypeError:
        rendered = False
    check("t45_three_seed_lines_render", rendered)

    s5 = (any(r["seed"] == 42 for r in per) and all(
        (r["gradcam_equivalence_max_abs_diff"] is not None
         and r["gradcam_equivalence_max_abs_diff"] <= RA.EQUIV_TOL
         and r["gradcam_equivalence_n"] == r["n_images"])
        if r["seed"] == 42 else r["gradcam_equivalence_n"] == 0 for r in per))
    check("t46_S5_passes_with_seed42_only_equivalence", s5 is True)
    per_bad = [seedrow(42, 2e-05, 848), seedrow(43, None, 0), seedrow(44, None, 0)]
    s5b = (any(r["seed"] == 42 for r in per_bad) and all(
        (r["gradcam_equivalence_max_abs_diff"] is not None
         and r["gradcam_equivalence_max_abs_diff"] <= RA.EQUIV_TOL
         and r["gradcam_equivalence_n"] == r["n_images"])
        if r["seed"] == 42 else r["gradcam_equivalence_n"] == 0 for r in per_bad))
    check("t47_S5_fails_when_seed42_exceeds_tol", s5b is False)

    n = len(RES) - sum(RES)
    print("\n  %d tests, %d passed, %d failed -> %s" % (len(RES), sum(RES), n,
                                                      "PASS" if n == 0 else "FAIL"))
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
