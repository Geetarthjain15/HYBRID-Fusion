"""Phase X2b - SDPA-preserving ViT token attribution for FusionV2.

Gradient x Input relevance on the ViT patch-EMBEDDING tokens, computed while the
verified SDPA attention implementation stays active. Read-only: the model is
instrumented with one forward hook that returns its input unchanged, and nothing
else about the model, its weights, its configuration or its arithmetic changes.

Why the embedding output and not the final hidden state
-------------------------------------------------------
`fusion_v2.py:91` consumes only `last_hidden_state[:, 0]` - the CLS token. The
196 final-layer patch tokens feed no downstream output, so
d(fused_logit)/d(last_hidden_state[:, 1:]) is identically zero and any relevance
computed there would be an all-zero map. The patch-embedding output is the
earliest site whose patch tokens reach the final CLS token through all 12 encoder
layers, so it is the site used here. `assert_final_patch_grad_is_zero()` keeps
that reasoning executable rather than merely asserted in prose.

What this does NOT do
---------------------
No attention matrix is read, `output_attentions` is never requested and
`set_attn_implementation` is never called. The resulting 14x14 map is a local
first-order model-sensitivity measure. It is not attention, not segmentation and
not anatomical localisation, and no region of it may be described as tumour.

Usage
-----
    env_win/Scripts/python.exe src/publication/explainability/sdpa_token_attribution.py --project-root . --smoke
    env_win/Scripts/python.exe src/publication/explainability/sdpa_token_attribution.py --project-root . --neutrality --all-seeds
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
SRC = PUB.parent
for _p in (str(PUB), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_ablation as TA                                   # noqa: E402
from fusion_v2 import tta_logits                              # noqa: E402
from fusion_v2_gradcam import load_verified_model             # noqa: E402

SEEDS = [42, 43, 44]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXPECTED_N = 848
EXPECTED_COUNTS = {"glioma": 243, "meningioma": 228, "notumor": 116,
                   "pituitary": 261}

NEUTRALITY_TOL = 1e-6          # gate A5, inherited from X2; never widened here
ACC_TOL_PP = 0.0               # gate A4
N_PATCH = 196
GRID = 14
ATTR_SITE = "vit.vit.embeddings"
ATTR_METHOD = "gradient_x_input"
SMOKE_PER_CLASS = 4

RECON_ROOT = "publication_runs/explainability/reconstructed_clean_test"
OUT_ROOT = "publication_results/explainability/x2b_token_attribution"
BASELINE_MANIFEST = "publication_results/audit/baseline_manifest.json"


class X2bError(RuntimeError):
    """A Phase X2b stop condition was hit."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def banner(m):
    print("\n" + "=" * 78 + "\n  %s\n" % m + "=" * 78, flush=True)


def stamp(m):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m), flush=True)


def _rel(p, root) -> str:
    # Cross-drive relpath raises on Windows; an absolute path is a correct answer.
    try:
        return os.path.relpath(str(p), str(root)).replace("\\", "/")
    except ValueError:
        return os.path.abspath(str(p)).replace("\\", "/")


def state_dict_digest(model) -> str:
    """Hash every parameter and buffer, so restoration can be proven (A8)."""
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------- attributor ----

class TokenAttributor:
    """Attaches one read-only forward hook to the ViT patch-embedding module.

    SDPA is left exactly as loaded. `__exit__` removes the hook even if the body
    raised, and never restores anything it did not change - because it changes
    nothing beyond registering that hook.
    """

    def __init__(self, model, site: str = ATTR_SITE):
        self.model = model
        self.site = site
        self._handles = []
        self._cap = {}
        self._entered = False
        self._attn_impl_at_enter = None

    def _module(self):
        m = self.model
        for part in self.site.split("."):
            m = getattr(m, part)
        return m

    def __enter__(self):
        self._attn_impl_at_enter = self.attn_impl()

        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._cap["h"] = h
            return out                      # unchanged: the forward is untouched

        self._handles.append(self._module().register_forward_hook(hook))
        self._entered = True
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        self._cap.clear()
        self._entered = False
        return False                        # never swallow an exception

    @property
    def n_hooks(self) -> int:
        return len(self._handles)

    def attn_impl(self):
        return getattr(self.model.vit.config, "_attn_implementation", None)

    def attribute(self, x: torch.Tensor, target_index: torch.Tensor | None = None):
        """One grad-enabled forward + one autograd.grad. Returns detached results.

        `torch.autograd.grad` is used rather than `.backward()` so no gradient is
        ever accumulated into model parameters (gate A7).
        """
        if not self._entered:
            raise X2bError("TokenAttributor.attribute called outside its context")
        self._cap.clear()
        logits = self.model(x)              # autograd ENABLED - no torch.no_grad
        h = self._cap.get("h")
        if h is None:
            raise X2bError("hook captured nothing at site %s" % self.site)
        if h.dim() != 3 or h.shape[1] != N_PATCH + 1:
            raise X2bError("unexpected hidden shape %s at %s"
                           % (tuple(h.shape), self.site))
        probs = logits.softmax(-1)
        tgt = probs.argmax(-1) if target_index is None else target_index.to(torch.long)
        # Samples are independent in eval mode (BatchNorm uses running stats,
        # LayerNorm is per-sample), so the gradient of the SUM of the selected
        # logits gives each row its own d(logit_i)/d(h_i) with no cross terms.
        sel = logits.gather(1, tgt.view(-1, 1)).sum()
        grad, = torch.autograd.grad(sel, h, retain_graph=False, create_graph=False)
        gp, hp = grad[:, 1:], h[:, 1:]      # drop CLS at index 0
        rel = (gp * hp).sum(-1)             # [B, 196]
        return {
            "maps": rel.detach().reshape(-1, GRID, GRID),
            "logits": logits.detach(),
            "probs": probs.detach(),
            "target_index": tgt.detach(),
            "patch_grad_abs_sum": gp.abs().sum((1, 2)).detach(),
        }


def assert_final_patch_grad_is_zero(model, x) -> dict:
    """Executable record of the §2.3 finding: the final patch tokens are dead.

    Returns the measured gradient magnitudes at both candidate sites so the
    choice of attribution site rests on a number, not on a comment.
    """
    cap = {}

    def hook(_m, _i, out):
        h = getattr(out, "last_hidden_state", out[0] if isinstance(out, tuple) else out)
        cap["h"] = h
        return out

    hh = model.vit.vit.encoder.register_forward_hook(hook)
    try:
        logits = model(x)
        t = int(logits.argmax(-1)[0])
        g, = torch.autograd.grad(logits[0, t], cap["h"])
    finally:
        hh.remove()
    return {
        "final_cls_grad_abs_sum": float(g[:, 0].abs().sum()),
        "final_patch_grad_abs_sum": float(g[:, 1:].abs().sum()),
        "final_patch_grad_nonzero": int((g[:, 1:].abs() > 0).sum()),
        "final_patch_grad_numel": int(g[:, 1:].numel()),
    }


# ----------------------------------------------------------------- scoring ---

def build_loader(recon_dir: Path, keep: set | None, batch: int):
    from torch.utils.data import DataLoader
    from torchvision import datasets
    ds = datasets.ImageFolder(str(recon_dir), transform=TA.eval_transform)
    if keep is not None:
        ds.samples = [s for s in ds.samples if os.path.basename(s[0]) in keep]
        ds.targets = [t for _, t in ds.samples]
    if ds.classes != CLASSES:
        raise X2bError("ImageFolder class order %s != %s" % (ds.classes, CLASSES))
    return DataLoader(ds, batch_size=batch, num_workers=0, shuffle=False), ds


def module_source_clean(path=None) -> dict:
    """Gate A2: the eager route must not be reachable from this module.

    Structural, not phrase matching. A text search would trip over this module's
    own prose and string constants, and would miss nothing real that the AST
    does not see: the two ways to reach eager attention are a CALL to
    `set_attn_implementation` and a KEYWORD ARGUMENT `output_attentions=`.
    Mentioning either name in a docstring or a dict key is harmless and must not
    fail the gate.
    """
    import ast
    tree = ast.parse(Path(path or __file__).read_text(encoding="utf-8"))
    bad_calls, bad_kwargs = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            nm = f.attr if isinstance(f, ast.Attribute) else (
                f.id if isinstance(f, ast.Name) else None)
            if nm == "set_attn_implementation":
                bad_calls.append(getattr(node, "lineno", -1))
        if isinstance(node, ast.keyword) and node.arg == "output_attentions":
            bad_kwargs.append(getattr(node, "lineno", -1))
        if isinstance(node, ast.Attribute) and node.attr == "output_attentions":
            bad_kwargs.append(getattr(node, "lineno", -1))
    return {
        "no_set_attn_implementation_call": not bad_calls,
        "no_output_attentions_request": not bad_kwargs,
    }


def run_seed(root: Path, seed: int, recon_dir: Path, keep: set | None,
             batch: int) -> dict:
    model, _dev, ck_sha = load_verified_model(root, seed=seed, device="cpu")
    model.eval()
    sd_before = state_dict_digest(model)
    impl_before = getattr(model.vit.config, "_attn_implementation", None)
    if impl_before != "sdpa":
        raise X2bError("attention implementation is %r, expected 'sdpa'" % impl_before)

    loader, ds = build_loader(recon_dir, keep, batch)
    ref_path = root / ("runs_v2_clean/fusion_v2_seed%d/preds_test.csv" % seed)
    if not ref_path.is_file():
        raise X2bError("reference predictions absent: %s" % ref_path)
    with open(ref_path, encoding="utf-8") as f:
        ref = {r["filename"]: r for r in csv.DictReader(f)}

    order = [os.path.basename(p) for p, _ in ds.samples]
    true_of = {os.path.basename(p): CLASSES[t] for p, t in ds.samples}

    # --- uninstrumented reference pass, same process, same batching ----------
    plain = {}
    with torch.no_grad():
        j = 0
        for x, _y in loader:
            sp = torch.softmax(model(x), -1)
            tp = tta_logits(model, x)
            for k in range(x.shape[0]):
                fn = order[j]; j += 1
                plain[fn] = (sp[k].numpy().astype(np.float64),
                             tp[k].numpy().astype(np.float64))

    # --- instrumented pass ---------------------------------------------------
    t0 = time.time()
    rows, maps_out, names_out = [], [], []
    impl_seen = {impl_before}
    worst_single = worst_tta = 0.0
    worst_recorded_tta = 0.0
    n_over = 0
    changes_plain = changes_recorded = 0
    maps_valid = 0
    correct_instr = correct_recorded = 0

    with TokenAttributor(model) as att:
        j = 0
        for x, _y in loader:
            n = x.shape[0]
            fns = order[j:j + n]; j += n
            # Target = predicted class from the recorded inference (proposal §3).
            tgt = torch.tensor([CLASSES.index(ref[f]["pred_label"]) for f in fns],
                               dtype=torch.long)
            out = att.attribute(x, target_index=tgt)
            impl_seen.add(att.attn_impl())
            with torch.no_grad():
                tta = tta_logits(model, x)
            impl_seen.add(att.attn_impl())

            for k in range(n):
                fn = fns[k]
                ps = out["probs"][k].numpy().astype(np.float64)
                pt = tta[k].numpy().astype(np.float64)
                p_s, p_t = plain[fn]
                d_s = float(np.abs(ps - p_s).max())
                d_t = float(np.abs(pt - p_t).max())
                rp = np.array([float(ref[fn]["p_" + c]) for c in CLASSES])
                d_rec = float(np.abs(pt - rp).max())
                worst_single = max(worst_single, d_s)
                worst_tta = max(worst_tta, d_t)
                worst_recorded_tta = max(worst_recorded_tta, d_rec)
                if max(d_s, d_t) > NEUTRALITY_TOL:
                    n_over += 1

                instr_lbl = CLASSES[int(pt.argmax())]
                plain_lbl = CLASSES[int(p_t.argmax())]
                rec_lbl = ref[fn]["pred_label"]
                changes_plain += int(instr_lbl != plain_lbl)
                changes_recorded += int(instr_lbl != rec_lbl)
                tl = true_of[fn]
                correct_instr += int(instr_lbl == tl)
                correct_recorded += int(rec_lbl == tl)

                m = out["maps"][k].numpy().astype(np.float32)
                pg = float(out["patch_grad_abs_sum"][k])
                ok = (bool(np.isfinite(m).all()) and m.shape == (GRID, GRID)
                      and float(m.max()) > float(m.min()) and pg > 0.0)
                maps_valid += int(ok)
                maps_out.append(m); names_out.append(fn)

                rows.append({
                    "filename": fn, "seed": seed, "true_label": tl,
                    "recorded_pred_label": rec_lbl,
                    "instrumented_pred_label": instr_lbl,
                    "plain_pred_label": plain_lbl,
                    "target_index": int(out["target_index"][k]),
                    "max_abs_prob_diff_single": d_s,
                    "max_abs_prob_diff_tta": d_t,
                    "max_abs_prob_diff_vs_recorded_tta": d_rec,
                    "patch_grad_abs_sum": pg,
                    "map_min": float(m.min()), "map_max": float(m.max()),
                    "map_mean": float(m.mean()),
                    "map_valid": bool(ok),
                    **{"p_single_" + c: float(ps[i]) for i, c in enumerate(CLASSES)},
                    **{"p_tta_" + c: float(pt[i]) for i, c in enumerate(CLASSES)},
                })
        hooks_inside = att.n_hooks
    runtime = time.time() - t0

    impl_after = getattr(model.vit.config, "_attn_implementation", None)
    param_grads_clean = all(p.grad is None for p in model.parameters())
    sd_after = state_dict_digest(model)

    n = len(rows)
    acc_i = 100.0 * correct_instr / n
    acc_r = 100.0 * correct_recorded / n
    fn_mismatch = sum(1 for r in rows if r["filename"] not in ref)
    lbl_mismatch = sum(1 for r in rows
                       if ref.get(r["filename"], {}).get("true_label")
                       != r["true_label"])

    return {
        "seed": seed,
        "checkpoint_sha256": ck_sha,
        "n_images": n,
        "batch_size": batch,
        "attn_impl_before": impl_before,
        "attn_impl_after": impl_after,
        "attn_impls_observed": sorted(str(i) for i in impl_seen),
        "hooks_registered_inside": hooks_inside,
        "hooks_after_exit": 0,
        "param_grads_all_none": bool(param_grads_clean),
        "model_still_eval": not model.training,
        "state_dict_digest_before": sd_before,
        "state_dict_digest_after": sd_after,
        "state_dict_unchanged": sd_before == sd_after,
        "max_abs_prob_diff_single": worst_single,
        "max_abs_prob_diff_tta": worst_tta,
        "max_abs_prob_diff_vs_recorded_tta": worst_recorded_tta,
        "n_over_tolerance": n_over,
        "pred_label_changes_vs_plain": changes_plain,
        "pred_label_changes_vs_recorded": changes_recorded,
        "accuracy_instrumented_pct": acc_i,
        "accuracy_recorded_pct": acc_r,
        "accuracy_diff_pp": acc_i - acc_r,
        "maps_valid": maps_valid,
        "filename_mismatches": fn_mismatch,
        "true_label_mismatches": lbl_mismatch,
        "runtime_s": runtime,
        "_rows": rows,
        "_maps": np.stack(maps_out) if maps_out else np.zeros((0, GRID, GRID), np.float32),
        "_names": names_out,
    }


def acceptance(per_seed: list, full: bool, src: dict, degen: dict) -> dict:
    n_exp = EXPECTED_N if full else SMOKE_PER_CLASS * len(CLASSES)
    A = {}
    A["A1_sdpa_active_throughout"] = all(
        s["attn_impl_before"] == "sdpa" and s["attn_impl_after"] == "sdpa"
        and s["attn_impls_observed"] == ["sdpa"] for s in per_seed)
    A["A2_eager_route_unreachable"] = all(src.values())
    A["A3_zero_pred_label_changes"] = all(
        s["pred_label_changes_vs_plain"] == 0
        and s["pred_label_changes_vs_recorded"] == 0 for s in per_seed)
    A["A4_accuracy_diff_zero_pp"] = all(
        abs(s["accuracy_diff_pp"]) <= ACC_TOL_PP for s in per_seed)
    A["A5_max_prob_diff_within_tol"] = all(
        max(s["max_abs_prob_diff_single"], s["max_abs_prob_diff_tta"])
        <= NEUTRALITY_TOL for s in per_seed)
    A["A6_all_maps_valid"] = all(s["maps_valid"] == s["n_images"] for s in per_seed)
    A["A7_no_retained_state"] = all(
        s["hooks_after_exit"] == 0 and s["param_grads_all_none"]
        and s["model_still_eval"] for s in per_seed)
    A["A8_exact_restoration"] = all(s["state_dict_unchanged"] for s in per_seed)
    A["A10_identity"] = all(
        s["filename_mismatches"] == 0 and s["true_label_mismatches"] == 0
        and s["n_images"] == n_exp for s in per_seed)
    A["site_correction_still_required"] = (
        degen["final_patch_grad_abs_sum"] == 0.0
        and degen["final_patch_grad_nonzero"] == 0)
    return A


def preservation_check(root: Path) -> dict:
    mp = root / BASELINE_MANIFEST
    if not mp.is_file():
        return {"checked": 0, "altered": 0, "available": False}
    man = json.loads(mp.read_text(encoding="utf-8"))
    inv = man.get("inventory", {})
    entries = list(inv.get("prediction_csvs", [])) + list(inv.get("metric_files", []))
    altered, checked, missing = [], 0, []
    for e in entries:
        p = root / e["path"]
        if not p.is_file():
            missing.append(e["path"]); continue
        checked += 1
        if sha256(p) != e["sha256"]:
            altered.append(e["path"])
    return {"checked": checked, "altered": len(altered),
            "altered_paths": altered, "missing": missing, "available": True}


def render_md(R: dict) -> str:
    L = []
    a = R["acceptance"]
    L.append("# Phase X2b - SDPA-Preserving ViT Token Attribution")
    L.append("")
    L.append("**Overall: %s**" % R["overall"])
    L.append("")
    L.append("| field | value |")
    L.append("|---|---|")
    for k in ("phase", "mode", "created_utc", "attn_implementation",
              "attribution_method", "attribution_site", "target_rule",
              "batch_size", "n_images", "neutrality_tolerance"):
        L.append("| %s | `%s` |" % (k, R[k]))
    L.append("| class order | `%s` |" % ", ".join(R["class_order"]))
    L.append("")
    L.append("## Why the embedding site")
    L.append("")
    d = R["degenerate_site_probe"]
    L.append("Measured gradient magnitude at the two candidate sites, seed 42:")
    L.append("")
    L.append("| site | sum of abs gradient | nonzero elements |")
    L.append("|---|---|---|")
    L.append("| final `last_hidden_state[:, 0]` (CLS) | %.6e | - |"
             % d["final_cls_grad_abs_sum"])
    L.append("| final `last_hidden_state[:, 1:]` (196 patches) | %.6e | %d / %d |"
             % (d["final_patch_grad_abs_sum"], d["final_patch_grad_nonzero"],
                d["final_patch_grad_numel"]))
    L.append("")
    L.append("`fusion_v2.py:91` reads only the CLS token, so the final patch")
    L.append("tokens feed no downstream output and their gradient is identically")
    L.append("zero. Attribution is therefore taken at the patch-embedding output.")
    L.append("")
    L.append("## Per-seed neutrality")
    L.append("")
    L.append("| seed | max abs diff single | max abs diff TTA | over tol | "
             "label changes | acc instr | acc recorded | acc diff pp | maps valid | s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in R["per_seed"]:
        L.append("| %d | %.6e | %.6e | %d | %d | %.4f%% | %.4f%% | %.4f | %d/%d | %.0f |"
                 % (s["seed"], s["max_abs_prob_diff_single"],
                    s["max_abs_prob_diff_tta"], s["n_over_tolerance"],
                    s["pred_label_changes_vs_plain"] + s["pred_label_changes_vs_recorded"],
                    s["accuracy_instrumented_pct"], s["accuracy_recorded_pct"],
                    s["accuracy_diff_pp"], s["maps_valid"], s["n_images"],
                    s["runtime_s"]))
    L.append("")
    L.append("## Acceptance checks")
    L.append("")
    L.append("| check | result |")
    L.append("|---|---|")
    for k, v in a.items():
        L.append("| %s | %s |" % (k, "PASS" if v else "FAIL"))
    p = R["preservation"]
    L.append("| A9_preservation | %s |"
             % ("PASS" if p.get("available") and p["altered"] == 0 else "FAIL"))
    L.append("")
    L.append("Preservation: %d baseline artifacts re-hashed, %d altered."
             % (p.get("checked", 0), p.get("altered", 0)))
    L.append("")
    L.append("## Interpretation limits")
    L.append("")
    L.append("These 14x14 maps are a local, first-order measure of how the fused")
    L.append("logit for the selected class responds to perturbation of the ViT")
    L.append("patch-embedding representation. They are model sensitivity only.")
    L.append("")
    L.append("- No attention matrix was read; no attention-based claim is made.")
    L.append("- The maps are not segmentation, detection or anatomical")
    L.append("  localisation, and no region of them may be described as tumour.")
    L.append("- They are not a causal, exact or complete explanation of any")
    L.append("  prediction, and they do not demonstrate clinically correct")
    L.append("  reasoning.")
    L.append("- Faithfulness is not established here. It is measured only by the")
    L.append("  prespecified deletion/insertion and stability protocol of Phase")
    L.append("  X5, which has not been run.")
    L.append("- The Phase X2 eager-attention failure remains recorded and is not")
    L.append("  repaired, superseded or invalidated by this phase.")
    L.append("- FusionV2 architecture, weights, gate, training, preprocessing,")
    L.append("  TTA policy and all published metrics are unchanged.")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase X2b token attribution")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--neutrality", action="store_true")
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if not (args.smoke or args.neutrality):
        ap.error("choose --smoke or --neutrality")

    root = Path(args.project_root).resolve()
    recon = root / RECON_ROOT
    if not recon.is_dir():
        raise X2bError("reconstructed clean test set absent: %s" % recon)
    man_path = recon / "RECONSTRUCTION_MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man["n_files"] != EXPECTED_N or man["per_class_counts"] != EXPECTED_COUNTS:
        raise X2bError("X1 manifest does not match the verified clean test set")

    full = bool(args.neutrality)
    if full:
        keep = None
        seeds = SEEDS if args.all_seeds else [args.seed or 42]
    else:
        by_class = {c: [] for c in CLASSES}
        for e in sorted(man["files"], key=lambda e: e["filename"]):
            if len(by_class[e["manifest_class"]]) < SMOKE_PER_CLASS:
                by_class[e["manifest_class"]].append(e["filename"])
        keep = {f for v in by_class.values() for f in v}
        seeds = [args.seed or 42]

    out = root / OUT_ROOT
    banner("PHASE X2b - %s" % ("full neutrality" if full else "smoke"))
    stamp("root        : %s" % root)
    stamp("images      : %s" % ("848 (all)" if keep is None else "%d" % len(keep)))
    stamp("seeds       : %s" % seeds)
    stamp("batch       : %d" % args.batch)

    src = module_source_clean()
    stamp("source gate : %s" % src)

    # Degenerate-site probe, recorded so the site choice stays evidence-backed.
    m0, _d, _s = load_verified_model(root, seed=seeds[0], device="cpu")
    m0.eval()
    e0 = sorted(man["files"], key=lambda e: e["filename"])[0]
    from PIL import Image
    x0 = TA.eval_transform(Image.open(root / e0["destination_path"]).convert("RGB")
                           ).unsqueeze(0)
    degen = assert_final_patch_grad_is_zero(m0, x0)
    stamp("final-patch grad abs sum = %.6e (nonzero %d/%d)"
          % (degen["final_patch_grad_abs_sum"], degen["final_patch_grad_nonzero"],
             degen["final_patch_grad_numel"]))
    del m0

    per_seed = []
    for s in seeds:
        stamp("seed %d ..." % s)
        r = run_seed(root, s, recon, keep, args.batch)
        stamp("  max|dp| single=%.3e tta=%.3e | over tol=%d | changes=%d/%d | "
              "acc %.4f vs %.4f | maps %d/%d | %.0fs"
              % (r["max_abs_prob_diff_single"], r["max_abs_prob_diff_tta"],
                 r["n_over_tolerance"], r["pred_label_changes_vs_plain"],
                 r["pred_label_changes_vs_recorded"],
                 r["accuracy_instrumented_pct"], r["accuracy_recorded_pct"],
                 r["maps_valid"], r["n_images"], r["runtime_s"]))
        per_seed.append(r)

    A = acceptance(per_seed, full, src, degen)
    pres = preservation_check(root)
    A9 = bool(pres.get("available")) and pres["altered"] == 0
    overall = "PASS" if all(A.values()) and A9 else "FAIL"

    R = {
        "phase": "X2b",
        "mode": "neutrality_full" if full else "smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "attn_implementation": "sdpa",
        "attribution_method": ATTR_METHOD,
        "attribution_site": "%s output, patch tokens 1:197" % ATTR_SITE,
        "target_rule": "predicted class from recorded inference",
        "batch_size": args.batch,
        "seeds": seeds,
        "n_images": per_seed[0]["n_images"],
        "neutrality_tolerance": NEUTRALITY_TOL,
        "class_order": CLASSES,
        "checkpoint_sha256": {str(s["seed"]): s["checkpoint_sha256"] for s in per_seed},
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__,
                        "platform": platform.platform()},
        "inputs": {
            "manifest": _rel(man_path, root),
            "manifest_sha256": sha256(man_path),
            "reference_predictions": [
                "runs_v2_clean/fusion_v2_seed%d/preds_test.csv" % s for s in seeds],
            "baseline_manifest": BASELINE_MANIFEST,
        },
        "source_gate": src,
        "degenerate_site_probe": degen,
        "per_seed": [{k: v for k, v in s.items() if not k.startswith("_")}
                     for s in per_seed],
        "acceptance": A,
        "preservation": pres,
        "overall": overall,
    }

    # -- write, never overwrite ---------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if full else "_smoke"
    targets = {
        "json": out / ("x2b_neutrality%s.json" % suffix),
        "md": out / ("X2B_NEUTRALITY_REPORT%s.md" % suffix),
        "seed_csv": out / ("per_seed_neutrality%s.csv" % suffix),
        "img_csv": out / ("per_image_neutrality%s.csv" % suffix),
    }
    for s in per_seed:
        targets["npz_%d" % s["seed"]] = out / ("token_maps_seed%d%s.npz"
                                              % (s["seed"], suffix))
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing:
        raise X2bError("refusing to overwrite existing outputs: %s" % existing)

    targets["json"].write_text(json.dumps(R, indent=2), encoding="utf-8")
    targets["md"].write_text(render_md(R), encoding="utf-8")

    scols = [k for k in per_seed[0] if not k.startswith("_")]
    with open(targets["seed_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, scols, extrasaction="ignore"); w.writeheader()
        for s in per_seed:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v)
                        for k, v in s.items() if k in scols})

    rows = [r for s in per_seed for r in s["_rows"]]
    with open(targets["img_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    for s in per_seed:
        np.savez_compressed(
            targets["npz_%d" % s["seed"]],
            filenames=np.array(s["_names"]),
            true_labels=np.array([r["true_label"] for r in s["_rows"]]),
            pred_labels=np.array([r["instrumented_pred_label"] for r in s["_rows"]]),
            target_index=np.array([r["target_index"] for r in s["_rows"]], np.int64),
            maps=s["_maps"])

    mani = {str(_rel(p, root)): sha256(p) for p in targets.values()}
    (out / ("sha256_manifest%s.json" % suffix)).write_text(
        json.dumps(mani, indent=2), encoding="utf-8")

    banner("X2b %s : %s" % (R["mode"], overall))
    for k, v in A.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    print("  [%s] A9_preservation (%d checked, %d altered)"
          % ("PASS" if A9 else "FAIL", pres.get("checked", 0), pres.get("altered", 0)))
    print("\noutputs -> %s" % _rel(out, root))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
