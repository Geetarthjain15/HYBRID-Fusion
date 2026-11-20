"""Phase X4 - explanation sanity checks for the X3 attribution maps.

Tests whether the CNN Grad-CAM++ (7x7) and ViT patch-embedding Gradient x Input
(14x14) maps are technically valid attribution outputs. It does NOT test whether
they are faithful, stable, mutually consistent, or transferable across sources -
those are Phases X5 and X6 and are deliberately absent here.

Checks 1-8 are pass/fail gates. Their tolerances are fixed in the constants
below and are NOT adjustable after observing results.

Checks 9-11 (parameter randomization, random-map baseline, centre-prior
baseline) are DESCRIPTIVE. They report similarity numbers and carry no
threshold and no verdict, because no decision rule for them was prespecified.
Reporting them as "low" or "high" would be a post-hoc judgement, so the numbers
are reported as observed values only.

Nothing here regenerates or modifies the X3 maps, the checkpoints, or any other
existing artifact. SDPA is never changed and attention matrices are never read.

Usage
-----
    env_win/Scripts/python.exe src/publication/explainability/x4_sanity_checks.py --project-root . --smoke
    env_win/Scripts/python.exe src/publication/explainability/x4_sanity_checks.py --project-root . --full
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
SRC = PUB.parent
for _p in (str(HERE), str(PUB), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_ablation as TA                                   # noqa: E402
from fusion_v2_gradcam import load_verified_model, gradcam_plus_plus  # noqa: E402
import cnn_attribution as CA                                  # noqa: E402
import vit_attribution as VA                                  # noqa: E402
from run_attribution import SynchronizedAttributor, state_dict_digest  # noqa: E402

# ---------------------------------------------------------------- constants --
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_SEED = 42                 # checkpoint used for all X4 work
SAMPLE_SEED = 42                # RNG seed for the stratified sample
SAMPLE_PER_CLASS = 4            # -> 16 images
SMOKE_PER_CLASS = 1             # -> 4 images
X3_BATCH = 32                   # the batch setting X3 used
RANDOMIZE_SEED = 20260927       # parameter-randomization generator
BASELINE_SEED = 20260927        # random-map baseline generator

# --- gates 1-8: fixed BEFORE execution, never adjusted afterwards ------------
DET_TOL = 0.0                   # (1) repeated extraction must be bit-identical
BATCH_REL_TOL = 1e-4            # (2) float-accumulation tolerance, relative to
#                                     map scale. Precedent: the same bound was
#                                     used as a numerical (not scientific) check
#                                     in test_sdpa_token_attribution t32, set
#                                     before any X4 result was observed.
SENSITIVITY_MIN = 0.0           # (3) strictly greater than this
EQUIV_TOL = 1e-5                # (8) pre-approved in the X3 amendment §3
CNN_GRID, VIT_GRID = 7, 14      # (5)

RECON_ROOT = "publication_runs/explainability/reconstructed_clean_test"
X3_ROOT = "publication_results/explainability/x3_attribution"
OUT_ROOT = "publication_results/explainability/x4_sanity"
BASELINE_MANIFEST = "publication_results/audit/baseline_manifest.json"


class X4Error(RuntimeError):
    """A Phase X4 stop condition was hit."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stamp(m):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m), flush=True)


def _rel(p, root) -> str:
    try:
        return os.path.relpath(str(p), str(root)).replace("\\", "/")
    except ValueError:
        return os.path.abspath(str(p)).replace("\\", "/")


def source_gate(path) -> dict:
    """No call to set_attn_implementation, no output_attentions request."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    calls = kws = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            nm = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            calls += nm == "set_attn_implementation"
        if isinstance(n, ast.keyword) and n.arg == "output_attentions":
            kws += 1
        if isinstance(n, ast.Attribute) and n.attr == "output_attentions":
            kws += 1
    return {"no_set_attn_implementation_call": calls == 0,
            "no_output_attentions_request": kws == 0}


# ------------------------------------------------------- deterministic parts --

def vit_raster_order_ok() -> dict:
    """(6a) 196 patch values must reshape row-major into the 14x14 grid."""
    rel = torch.arange(VIT_GRID * VIT_GRID, dtype=torch.float32).reshape(1, -1)
    grad = torch.zeros(1, VIT_GRID * VIT_GRID + 1, 768)
    hidden = torch.zeros(1, VIT_GRID * VIT_GRID + 1, 768)
    # place the value in channel 0 so grad*hidden summed over channels == value
    grad[0, 1:, 0] = 1.0
    hidden[0, 1:, 0] = rel[0]
    m = VA.patch_relevance(grad, hidden)[0].numpy()
    expect = np.arange(VIT_GRID * VIT_GRID, dtype=np.float32).reshape(VIT_GRID, VIT_GRID)
    return {"vit_reshape_row_major": bool(np.array_equal(m, expect)),
            "vit_m_0_1": float(m[0, 1]), "vit_m_1_0": float(m[1, 0])}


def cnn_spatial_correspondence_ok() -> dict:
    """(6b) A peak in the stage-8 activation must appear at the same (r,c)."""
    out = {}
    ok = True
    for (r, c) in ((0, 0), (0, CNN_GRID - 1), (CNN_GRID - 1, 0), (3, 5)):
        a = np.full((1, 2, CNN_GRID, CNN_GRID), 0.1, np.float32)
        a[0, :, r, c] = 10.0
        g = np.full((1, 2, CNN_GRID, CNN_GRID), 0.5, np.float32)
        cam = CA.gradcam_pp_native(a, g)[0]
        got = np.unravel_index(int(cam.argmax()), cam.shape)
        out["peak_%d_%d" % (r, c)] = "%d,%d" % got
        ok &= got == (r, c)
    out["cnn_peak_preserved"] = bool(ok)
    return out


def upsample_orientation_ok() -> dict:
    """(6c) Resizing 7x7 -> 224x224 must not flip L-R, T-B or transpose."""
    out = {}
    ok = True
    corners = {"top_left": (0, 0), "top_right": (0, CNN_GRID - 1),
               "bottom_left": (CNN_GRID - 1, 0), "bottom_right": (CNN_GRID - 1, CNN_GRID - 1)}
    for name, (r, c) in corners.items():
        m = np.zeros((CNN_GRID, CNN_GRID), np.float32)
        m[r, c] = 1.0
        big = CA.rescale_like_library(m)
        br, bc = np.unravel_index(int(big.argmax()), big.shape)
        # the peak must stay in the same half vertically and horizontally
        same_v = (br < 112) == (r < CNN_GRID / 2)
        same_h = (bc < 112) == (c < CNN_GRID / 2)
        out[name] = "%d,%d" % (br, bc)
        ok &= bool(same_v and same_h)
    # transpose guard: an off-diagonal marker must not land at its mirror
    m = np.zeros((CNN_GRID, CNN_GRID), np.float32)
    m[1, 5] = 1.0
    big = CA.rescale_like_library(m)
    br, bc = np.unravel_index(int(big.argmax()), big.shape)
    out["asym_1_5"] = "%d,%d" % (br, bc)
    no_transpose = br < bc          # row index stayed smaller than column index
    out["no_axis_transpose"] = bool(no_transpose)
    out["orientation_preserved"] = bool(ok and no_transpose)
    return out


def random_map(rng, shape) -> np.ndarray:
    return rng.random(shape, dtype=np.float32)


def centre_prior(shape) -> np.ndarray:
    """Deterministic centre-weighted map; a sanity baseline, nothing more."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sig = max(h, w) / 4.0
    return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))).astype(np.float32)


def similarity(a: np.ndarray, b: np.ndarray) -> dict:
    """Descriptive similarity between two maps. No threshold, no verdict."""
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    out = {}
    if x.std() > 0 and y.std() > 0:
        out["pearson"] = float(np.corrcoef(x, y)[0, 1])
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        out["spearman"] = float(np.corrcoef(rx, ry)[0, 1])
    else:
        out["pearson"] = None
        out["spearman"] = None

    def norm(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)

    out["normalised_l1"] = float(np.abs(norm(x) - norm(y)).mean())
    return out


def randomized_copy(model):
    """In-memory parameter randomization. The checkpoint is never touched."""
    rnd = copy.deepcopy(model)
    g = torch.Generator().manual_seed(RANDOMIZE_SEED)
    n = 0
    with torch.no_grad():
        for p in rnd.parameters():
            std = float(p.detach().float().std()) if p.numel() > 1 else 1.0
            if not np.isfinite(std) or std == 0:
                std = 1.0
            p.copy_(torch.normal(0.0, std, size=p.shape, generator=g,
                                 dtype=p.dtype))
            n += 1
    rnd.eval()
    return rnd, n


# ------------------------------------------------------------------- driver --

def stratified_sample(manifest, per_class):
    by = {c: [] for c in CLASSES}
    for e in sorted(manifest["files"], key=lambda e: e["filename"]):
        by[e["manifest_class"]].append(e)
    rng = random.Random(SAMPLE_SEED)
    out = []
    for c in CLASSES:
        out.extend(sorted(rng.sample(by[c], per_class),
                          key=lambda e: e["filename"]))
    return out


def load_batch(root, entries):
    from PIL import Image
    xs = [TA.eval_transform(Image.open(root / e["destination_path"]).convert("RGB"))
          for e in entries]
    return torch.stack(xs)


def run(root: Path, per_class: int) -> dict:
    model, dev, ck = load_verified_model(root, seed=MODEL_SEED, device="cpu")
    model.eval()
    impl0 = model.vit.config._attn_implementation
    if impl0 != "sdpa":
        raise X4Error("attention implementation %r != 'sdpa'" % impl0)
    sd0 = state_dict_digest(model)

    man_path = root / RECON_ROOT / "RECONSTRUCTION_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    entries = stratified_sample(man, per_class)
    x = load_batch(root, entries)
    n = len(entries)
    stamp("sample: %d images (%d per class), seed %d" % (n, per_class, SAMPLE_SEED))

    # ---- target classes from the same rule X3 used: predicted ---------------
    with torch.no_grad():
        probs = model(x).softmax(-1)
    pred = probs.argmax(-1)
    alt = probs.argmin(-1)              # explicit, deterministic non-predicted

    rows, sims = [], []
    impls = {impl0}

    # ---- (1) determinism, (3) target sensitivity, (4)(5) validity ----------
    with SynchronizedAttributor(model) as att:
        a1 = att.attribute(x[:1], target_index=pred[:1])
        a2 = att.attribute(x[:1], target_index=pred[:1])
        impls.add(att.attn_impl())
        det_cnn = float(np.abs(a1["cnn_maps"] - a2["cnn_maps"]).max())
        det_vit = float(np.abs(a1["vit_maps"] - a2["vit_maps"]).max())

        # single-image pass over the whole sample
        single_cnn, single_vit, single_tgt = [], [], []
        for k in range(n):
            o = att.attribute(x[k:k + 1], target_index=pred[k:k + 1])
            single_cnn.append(o["cnn_maps"][0]); single_vit.append(o["vit_maps"][0])
            single_tgt.append(int(o["target_index"][0]))
        # batched pass at the X3 batch setting
        ob = att.attribute(x[:min(X3_BATCH, n)], target_index=pred[:min(X3_BATCH, n)])
        # maps for an explicitly selected non-predicted class
        oalt = att.attribute(x, target_index=alt)
        impls.add(att.attn_impl())
        hooks_inside = att.n_hooks
    hooks_after = att.n_hooks

    single_cnn = np.stack(single_cnn); single_vit = np.stack(single_vit)
    batch_cnn, batch_vit = ob["cnn_maps"], ob["vit_maps"]

    def rel_diff(a, b):
        sc = float(np.abs(a).max())
        return float(np.abs(a - b).max()) / sc if sc > 0 else 0.0

    batch_rel_cnn = rel_diff(single_cnn, batch_cnn)
    batch_rel_vit = rel_diff(single_vit, batch_vit)
    batch_tgt_same = single_tgt == [int(t) for t in ob["target_index"]]

    # ---- (9) parameter randomization --------------------------------------
    rnd_model, n_params = randomized_copy(model)
    with SynchronizedAttributor(rnd_model) as att:
        ornd = att.attribute(x, target_index=pred)
    del rnd_model

    # ---- per-image records -------------------------------------------------
    rng = np.random.default_rng(BASELINE_SEED)
    cp_cnn = centre_prior((CNN_GRID, CNN_GRID))
    cp_vit = centre_prior((VIT_GRID, VIT_GRID))
    finite_ok = nonconst_ok = shape_ok = sens_ok = True
    equiv_max = 0.0

    for k, e in enumerate(entries):
        c, v = single_cnn[k], single_vit[k]
        ca, va = oalt["cnn_maps"][k], oalt["vit_maps"][k]
        shape_ok &= (c.shape == (CNN_GRID, CNN_GRID) and v.shape == (VIT_GRID, VIT_GRID))
        finite_ok &= bool(np.isfinite(c).all() and np.isfinite(v).all())
        nonconst_ok &= bool(c.max() > c.min() and v.max() > v.min()
                            and np.abs(c).sum() > 0 and np.abs(v).sum() > 0)
        d_cnn = float(np.abs(c - ca).max()); d_vit = float(np.abs(v - va).max())
        sens_ok &= (d_cnn > SENSITIVITY_MIN and d_vit > SENSITIVITY_MIN)

        # (8) CNN equivalence against the existing verified helper
        helper = gradcam_plus_plus(model, x[k:k + 1], dev, int(pred[k]))
        eq = float(np.abs(CA.rescale_like_library(c) - helper).max())
        equiv_max = max(equiv_max, eq)

        rows.append({
            "filename": e["filename"], "true_label": e["manifest_class"],
            "pred_class": CLASSES[int(pred[k])], "alt_class": CLASSES[int(alt[k])],
            "cnn_shape": "x".join(map(str, c.shape)),
            "vit_shape": "x".join(map(str, v.shape)),
            "cnn_finite": bool(np.isfinite(c).all()),
            "vit_finite": bool(np.isfinite(v).all()),
            "cnn_nonconstant": bool(c.max() > c.min()),
            "vit_nonconstant": bool(v.max() > v.min()),
            "cnn_target_sensitivity_maxabs": d_cnn,
            "vit_target_sensitivity_maxabs": d_vit,
            "gradcam_equivalence_max_abs_diff": eq,
        })

        rc, rv = random_map(rng, (CNN_GRID, CNN_GRID)), random_map(rng, (VIT_GRID, VIT_GRID))
        for branch, real, randm, cp in (("cnn", c, rc, cp_cnn), ("vit", v, rv, cp_vit)):
            base = {"filename": e["filename"], "branch": branch}
            for label, other in (("parameter_randomized",
                                  ornd["cnn_maps"][k] if branch == "cnn" else ornd["vit_maps"][k]),
                                 ("random_map", randm),
                                 ("centre_prior", cp)):
                s = similarity(real, other)
                sims.append({**base, "comparison": label, **s})

    model.zero_grad(set_to_none=True)
    impl_after = model.vit.config._attn_implementation
    grads_clean = all(p.grad is None for p in model.parameters())
    sd_after = state_dict_digest(model)

    ori = {}
    ori.update(vit_raster_order_ok())
    ori.update(cnn_spatial_correspondence_ok())
    ori.update(upsample_orientation_ok())

    gates = {
        "C1_deterministic_repeat": det_cnn == DET_TOL and det_vit == DET_TOL,
        "C2_batch_vs_single": (batch_rel_cnn <= BATCH_REL_TOL
                               and batch_rel_vit <= BATCH_REL_TOL
                               and batch_tgt_same),
        "C3_target_class_sensitivity": bool(sens_ok),
        "C4_finite_and_nonconstant": bool(finite_ok and nonconst_ok),
        "C5_native_shapes": bool(shape_ok),
        "C6_orientation_preserved": bool(ori["vit_reshape_row_major"]
                                         and ori["cnn_peak_preserved"]
                                         and ori["orientation_preserved"]),
        "C7_hook_and_gradient_cleanup": (hooks_after == 0 and grads_clean
                                         and not model.training
                                         and sd_after == sd0
                                         and impl_after == "sdpa"
                                         and sorted(str(i) for i in impls) == ["sdpa"]),
        "C8_cnn_equivalence_le_1e-5": equiv_max <= EQUIV_TOL,
    }

    return {
        "n_images": n, "checkpoint_sha256": ck,
        "sample": [e["filename"] for e in entries],
        "gates": gates,
        "measured": {
            "determinism_max_abs_diff_cnn": det_cnn,
            "determinism_max_abs_diff_vit": det_vit,
            "batch_vs_single_rel_cnn": batch_rel_cnn,
            "batch_vs_single_rel_vit": batch_rel_vit,
            "batch_target_classes_identical": batch_tgt_same,
            "gradcam_equivalence_max_abs_diff": equiv_max,
            "hooks_inside": hooks_inside, "hooks_after_exit": hooks_after,
            "param_grads_all_none": grads_clean,
            "state_dict_unchanged": sd_after == sd0,
            "attn_impls_observed": sorted(str(i) for i in impls),
            "randomized_parameter_tensors": n_params,
        },
        "orientation_detail": ori,
        "_rows": rows, "_sims": sims,
    }


def summarise_sims(sims):
    out = {}
    for branch in ("cnn", "vit"):
        for comp in ("parameter_randomized", "random_map", "centre_prior"):
            sel = [s for s in sims if s["branch"] == branch and s["comparison"] == comp]
            for metric in ("pearson", "spearman", "normalised_l1"):
                vals = [s[metric] for s in sel if s[metric] is not None]
                if vals:
                    out["%s_%s_%s" % (branch, comp, metric)] = {
                        "mean": float(np.mean(vals)), "median": float(np.median(vals)),
                        "min": float(np.min(vals)), "max": float(np.max(vals)),
                        "n": len(vals)}
    return out


def preservation(root):
    man = json.loads((root / BASELINE_MANIFEST).read_text(encoding="utf-8"))
    inv = man["inventory"]
    ent = inv["prediction_csvs"] + inv["metric_files"]
    altered = [e["path"] for e in ent if sha256(root / e["path"]) != e["sha256"]]
    return {"checked": len(ent), "altered": len(altered), "altered_paths": altered}


def render_md(R):
    L = ["# Phase X4 - Explanation Sanity Checks", "",
         "**Overall: %s**" % R["overall"], "",
         "Mode `%s`, %d images, checkpoint seed %d, device cpu, attention `sdpa`."
         % (R["mode"], R["n_images"], MODEL_SEED), "",
         "## Gates (1-8)", "", "| check | result | measured |", "|---|---|---|"]
    m = R["measured"]
    detail = {
        "C1_deterministic_repeat": "max abs diff cnn %.3e / vit %.3e"
                                   % (m["determinism_max_abs_diff_cnn"], m["determinism_max_abs_diff_vit"]),
        "C2_batch_vs_single": "rel cnn %.3e / vit %.3e, target classes identical: %s"
                              % (m["batch_vs_single_rel_cnn"], m["batch_vs_single_rel_vit"],
                                 m["batch_target_classes_identical"]),
        "C3_target_class_sensitivity": "all images differ from the non-predicted class map",
        "C4_finite_and_nonconstant": "all maps finite, non-constant, non-zero",
        "C5_native_shapes": "cnn 7x7, vit 14x14",
        "C6_orientation_preserved": "raster order, peak correspondence, upsample orientation",
        "C7_hook_and_gradient_cleanup": "%d hooks after exit, grads clean %s, state_dict unchanged %s"
                                        % (m["hooks_after_exit"], m["param_grads_all_none"], m["state_dict_unchanged"]),
        "C8_cnn_equivalence_le_1e-5": "max abs diff %.3e (tolerance 1e-5)"
                                      % m["gradcam_equivalence_max_abs_diff"],
    }
    for k, v in R["gates"].items():
        L.append("| %s | %s | %s |" % (k, "PASS" if v else "FAIL", detail.get(k, "")))
    L += ["", "## Descriptive baselines (9-11) - no threshold, no verdict", "",
          "These are observed similarity values between the real maps and three",
          "reference maps. No decision rule was prespecified for them, so they are",
          "reported as measurements only. They are not evidence of faithfulness,",
          "which is measured solely by the Phase X5 protocol.", "",
          "| branch | comparison | Pearson mean | Spearman mean | normalised L1 mean |",
          "|---|---|---|---|---|"]
    s = R["similarity_summary"]
    for branch in ("cnn", "vit"):
        for comp in ("parameter_randomized", "random_map", "centre_prior"):
            p = s.get("%s_%s_pearson" % (branch, comp), {})
            sp = s.get("%s_%s_spearman" % (branch, comp), {})
            l1 = s.get("%s_%s_normalised_l1" % (branch, comp), {})
            L.append("| %s | %s | %s | %s | %s |" % (
                branch, comp.replace("_", " "),
                "%.4f" % p["mean"] if p else "n/a",
                "%.4f" % sp["mean"] if sp else "n/a",
                "%.4f" % l1["mean"] if l1 else "n/a"))
    L += ["", "## Scope", "",
          "X4 tests technical validity only. It does not compute deletion,",
          "insertion, confidence drop, stability, CNN-ViT agreement, composites or",
          "external-source results, and it makes no claim about anatomical",
          "localisation, tumour detection or clinical validity. No region of any",
          "map may be described as tumour. No attention matrix was read.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase X4 sanity checks")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if not (a.smoke or a.full):
        ap.error("choose --smoke or --full")
    root = Path(a.project_root).resolve()

    per_class = SMOKE_PER_CLASS if a.smoke else SAMPLE_PER_CLASS
    sfx = "_smoke" if a.smoke else ""
    out = root / OUT_ROOT
    targets = {k: out / (v % sfx) for k, v in
               {"json": "x4_sanity%s.json", "md": "X4_SANITY_REPORT%s.md",
                "rows": "per_image_sanity%s.csv",
                "sims": "similarity_baselines%s.csv"}.items()}
    clash = [str(p) for p in targets.values() if p.exists()]
    if clash:
        raise X4Error("refusing to overwrite: %s" % clash)

    gates_src = {m: source_gate(HERE / m) for m in
                 ("x4_sanity_checks.py", "cnn_attribution.py", "vit_attribution.py")}
    stamp("source gates: %s" % gates_src)
    if not all(all(v.values()) for v in gates_src.values()):
        raise X4Error("eager-attention route reachable from X4 code")

    R = run(root, per_class)
    pres = preservation(root)
    gates_ok = all(R["gates"].values()) and pres["altered"] == 0
    overall = "PASS" if gates_ok else "FAIL"

    man_path = root / RECON_ROOT / "RECONSTRUCTION_MANIFEST.json"
    x3_json = root / X3_ROOT / "x3_attribution.json"
    report = {
        "phase": "X4", "mode": "smoke" if a.smoke else "full",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall": overall, "gates_passed": gates_ok,
        "n_images": R["n_images"], "sample_filenames": R["sample"],
        "sampling": {"stratified_per_class": per_class, "sample_seed": SAMPLE_SEED,
                     "selected_before_viewing_maps": True},
        "model": {"checkpoint_seed": MODEL_SEED,
                  "checkpoint_sha256": R["checkpoint_sha256"],
                  "attn_implementation": "sdpa", "device": "cpu"},
        "thresholds_fixed_before_execution": {
            "determinism_tol": DET_TOL, "batch_rel_tol": BATCH_REL_TOL,
            "target_sensitivity_min": SENSITIVITY_MIN,
            "cnn_equivalence_tol": EQUIV_TOL,
            "checks_9_to_11": "descriptive only - no threshold, no verdict"},
        "seeds": {"sample": SAMPLE_SEED, "parameter_randomization": RANDOMIZE_SEED,
                  "random_map_baseline": BASELINE_SEED},
        "inputs": {
            "reconstruction_manifest": _rel(man_path, root),
            "reconstruction_manifest_sha256": sha256(man_path),
            "x3_attribution_json": _rel(x3_json, root),
            "x3_attribution_json_sha256": sha256(x3_json),
            "x3_maps_read": False,
            "note": "X4 recomputes maps from the verified checkpoint using the "
                    "X3 code path; it never reads, regenerates or modifies the "
                    "2544 stored X3 .npz maps."},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "platform": platform.platform(),
                        "cuda_available": torch.cuda.is_available()},
        "source_gates": gates_src,
        "gates": R["gates"], "measured": R["measured"],
        "orientation_detail": R["orientation_detail"],
        "similarity_summary": summarise_sims(R["_sims"]),
        "preservation": pres,
        "output_schema": {
            "per_image_sanity.csv": list(R["_rows"][0].keys()),
            "similarity_baselines.csv": list(R["_sims"][0].keys())},
        "scope_exclusions": ["deletion", "insertion", "confidence_drop", "stability",
                             "cnn_vit_agreement", "composite", "external_source"],
    }

    out.mkdir(parents=True, exist_ok=True)
    targets["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    targets["md"].write_text(render_md(report), encoding="utf-8")
    with open(targets["rows"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(R["_rows"][0].keys())); w.writeheader(); w.writerows(R["_rows"])
    with open(targets["sims"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(R["_sims"][0].keys())); w.writeheader(); w.writerows(R["_sims"])
    mani = {_rel(p, root): sha256(p) for p in targets.values()}
    (out / ("sha256_manifest%s.json" % sfx)).write_text(json.dumps(mani, indent=2),
                                                        encoding="utf-8")

    print("\nX4 %s: %s" % (report["mode"], overall))
    for k, v in R["gates"].items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    print("  [%s] preservation (%d checked, %d altered)"
          % ("PASS" if pres["altered"] == 0 else "FAIL", pres["checked"], pres["altered"]))
    return 0 if gates_ok else 1


if __name__ == "__main__":
    sys.exit(main())
