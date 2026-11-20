"""Tests for Phase X2b SDPA-preserving token attribution.

Run:
    env_win/Scripts/python.exe src/publication/explainability/test_sdpa_token_attribution.py
"""

from __future__ import annotations

import hashlib
import os
import sys
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
from fusion_v2_gradcam import load_verified_model                 # noqa: E402
import sdpa_token_attribution as X                                # noqa: E402

FROZEN = {
    "src/fusion_v2.py":
        "dd1a1180459b9eb1a3e1a02716d0406722f97d2e0eb1b9922ac551d0cdba1292",
    "src/train_ablation.py":
        "ed8bf84c4953a3c4bc4ac035784241d022316f04e4d0f6cd295c98c1126ab15c",
}

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_images(n=2):
    recon = ROOT / X.RECON_ROOT
    xs, fns = [], []
    for c in X.CLASSES:
        d = recon / c
        for f in sorted(os.listdir(d))[:1]:
            xs.append(TA.eval_transform(Image.open(d / f).convert("RGB")))
            fns.append(f)
            if len(xs) >= n:
                return torch.stack(xs), fns
    return torch.stack(xs), fns


def main() -> int:
    print("=" * 78 + "\n  PHASE X2b TESTS\n" + "=" * 78)

    # --- static checks -----------------------------------------------------
    for rel, want in FROZEN.items():
        check("t01_frozen_%s" % os.path.basename(rel), sha(ROOT / rel) == want, want[:16])

    src = X.module_source_clean()
    check("t02_no_set_attn_implementation", src["no_set_attn_implementation_call"])
    check("t03_no_output_attentions", src["no_output_attentions_request"])
    check("t04_class_order", X.CLASSES == ["glioma", "meningioma", "notumor", "pituitary"])
    check("t05_expected_counts_sum", sum(X.EXPECTED_COUNTS.values()) == X.EXPECTED_N,
          "%d" % X.EXPECTED_N)
    check("t06_tolerance_not_widened", X.NEUTRALITY_TOL == 1e-6 and X.ACC_TOL_PP == 0.0)
    check("t07_grid", X.N_PATCH == 196 and X.GRID == 14 and X.GRID ** 2 == X.N_PATCH)

    # --- model -------------------------------------------------------------
    model, _dev, ck = load_verified_model(ROOT, seed=42, device="cpu")
    model.eval()
    check("t08_checkpoint_verified", ck.startswith("9b328ba83df5d44b"), ck[:16])
    check("t09_sdpa_loaded",
          model.vit.config._attn_implementation == "sdpa",
          model.vit.config._attn_implementation)

    x, fns = load_images(2)
    sd0 = X.state_dict_digest(model)
    check("t10_state_digest_deterministic", X.state_dict_digest(model) == sd0, sd0[:16])

    # --- the degenerate site (documents why the site moved) ----------------
    d = X.assert_final_patch_grad_is_zero(model, x[:1])
    check("t11_final_patch_grad_exactly_zero",
          d["final_patch_grad_abs_sum"] == 0.0 and d["final_patch_grad_nonzero"] == 0,
          "sum=%.3e nonzero=%d/%d" % (d["final_patch_grad_abs_sum"],
                                      d["final_patch_grad_nonzero"],
                                      d["final_patch_grad_numel"]))
    check("t12_final_cls_grad_nonzero", d["final_cls_grad_abs_sum"] > 0.0,
          "%.6e" % d["final_cls_grad_abs_sum"])

    # --- attribution -------------------------------------------------------
    att = X.TokenAttributor(model)
    check("t13_site_resolves", hasattr(att, "_module") and att._module() is not None,
          X.ATTR_SITE)
    try:
        att.attribute(x)
        outside = False
    except X.X2bError:
        outside = True
    check("t14_refuses_outside_context", outside)

    with torch.no_grad():
        p_plain = torch.softmax(model(x), -1)

    with X.TokenAttributor(model) as a:
        check("t15_one_hook_registered", a.n_hooks == 1, "%d" % a.n_hooks)
        o = a.attribute(x)
        check("t16_sdpa_during", a.attn_impl() == "sdpa", a.attn_impl())
    check("t17_hook_removed_on_exit", a.n_hooks == 0)

    check("t18_map_shape", tuple(o["maps"].shape) == (2, 14, 14),
          str(tuple(o["maps"].shape)))
    check("t19_maps_finite", bool(torch.isfinite(o["maps"]).all()))
    check("t20_maps_nonconstant",
          all(float(o["maps"][k].max()) > float(o["maps"][k].min()) for k in range(2)))
    check("t21_patch_grad_positive", bool((o["patch_grad_abs_sum"] > 0).all()),
          "%.6e" % float(o["patch_grad_abs_sum"].min()))
    check("t22_no_grad_retained_on_outputs",
          all(getattr(o[k], "grad_fn", None) is None for k in o))

    # --- neutrality --------------------------------------------------------
    dmax = float((o["probs"] - p_plain).abs().max())
    check("t23_grad_forward_matches_nograd", dmax <= X.NEUTRALITY_TOL, "%.6e" % dmax)
    check("t24_pred_labels_match",
          bool(torch.equal(o["probs"].argmax(-1), p_plain.argmax(-1))))

    # --- no parameter gradient was ever accumulated ------------------------
    check("t25_param_grads_none", all(p.grad is None for p in model.parameters()))
    check("t26_state_dict_unchanged", X.state_dict_digest(model) == sd0)
    check("t27_sdpa_after", model.vit.config._attn_implementation == "sdpa")
    check("t28_model_still_eval", not model.training)

    # --- cleanup on exception ---------------------------------------------
    b = X.TokenAttributor(model)
    try:
        with b:
            raise RuntimeError("deliberate")
    except RuntimeError:
        pass
    check("t29_hook_removed_when_body_raises", b.n_hooks == 0)
    check("t30_sdpa_intact_after_raise",
          model.vit.config._attn_implementation == "sdpa")

    # --- target selection is honoured -------------------------------------
    with X.TokenAttributor(model) as a:
        t_a = torch.tensor([0, 0])
        t_b = torch.tensor([3, 3])
        ma = a.attribute(x, target_index=t_a)["maps"].clone()
        mb = a.attribute(x, target_index=t_b)["maps"].clone()
    check("t31_target_changes_map", float((ma - mb).abs().max()) > 0.0,
          "%.6e" % float((ma - mb).abs().max()))

    # --- batching independence (numerical tolerance, not a scientific gate) -
    with X.TokenAttributor(model) as a:
        single = torch.cat([a.attribute(x[k:k + 1], target_index=t_a[k:k + 1])["maps"]
                            for k in range(2)])
    scale = float(ma.abs().max())
    rel = float((single - ma).abs().max()) / scale if scale > 0 else 0.0
    check("t32_batch_matches_single", rel <= 1e-4, "rel max diff %.3e" % rel)

    # --- reshape is row-major over the 14x14 patch raster ------------------
    with X.TokenAttributor(model) as a:
        o1 = a.attribute(x[:1], target_index=torch.tensor([0]))
    flat = o1["maps"].reshape(-1)
    check("t33_reshape_row_major", flat.numel() == 196
          and bool(torch.equal(flat.reshape(14, 14), o1["maps"][0])))

    # --- the source gate must actually be able to fail (not vacuous) --------
    import tempfile
    bad = ("def f(m, x):\n"
           "    m.vit.set_attn_implementation('eager')\n"
           "    return m.vit(x, output_attentions=True)\n")
    with tempfile.TemporaryDirectory() as td:
        bp = os.path.join(td, "bad.py")
        open(bp, "w", encoding="utf-8").write(bad)
        g = X.module_source_clean(bp)
    check("t34_source_gate_detects_violation",
          not g["no_set_attn_implementation_call"]
          and not g["no_output_attentions_request"])
    ok_src = ("KEYS = ['set_attn_implementation', 'output_attentions']\n"
              "'''mentions set_attn_implementation and output_attentions'''\n")
    with tempfile.TemporaryDirectory() as td:
        gp = os.path.join(td, "ok.py")
        open(gp, "w", encoding="utf-8").write(ok_src)
        g2 = X.module_source_clean(gp)
    check("t35_source_gate_ignores_mere_mentions",
          g2["no_set_attn_implementation_call"] and g2["no_output_attentions_request"])

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("\n" + "=" * 78)
    print("  %d tests, %d passed, %d failed -> %s"
          % (len(RESULTS), len(RESULTS) - n_fail, n_fail,
             "PASS" if n_fail == 0 else "FAIL"))
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
