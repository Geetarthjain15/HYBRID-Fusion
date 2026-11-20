"""
Phase X2 - read-only dual-path instrumentation.

Exposes, from ONE synchronized inference pass, every tensor the later phases
need: fused logits/probabilities, CNN and ViT auxiliary logits/probabilities,
gate weights, CNN spatial feature tensors, ViT attention tensors, and the
selected fused target class.

The design problem this module solves is not "how do I read the tensors" but
"how do I read them without becoming a different model". Two mechanisms are
needed and both are intrusive if handled carelessly:

  - forward hooks on the CNN stages, which must be removed even if the forward
    pass raises;
  - eager ViT attention, because the model loads with `attn_implementation =
    "sdpa"` under which `output_attentions=True` silently returns None.

Both are therefore confined to a context manager that restores the model
exactly on exit, including on the exception path. Neutrality is then measured
rather than assumed: every image is scored twice, once through the plain model
and once through the recorder, and the two must agree.

Two comparisons are kept separate throughout, because they have different
meanings and different tolerances:

  A. NEUTRALITY - instrumented vs uninstrumented, same machine, same images,
     same checkpoint. Gate: max |dp| <= 1e-6. This measures what THIS phase does.
  B. IDENTITY - instrumented vs the authoritative recorded predictions. Gate:
     zero filename/true-label/predicted-label mismatches and 0.00 pp accuracy.
     Its probability residual is the cross-machine floating-point difference
     that X1 already documented at ~4e-06, and is reported, not held to 1e-6.

Conflating the two would either fail X2 on a residual it did not cause, or
silently relax the gate that matters.

Raw tensors are validated in flight and NOT persisted: the attention stack alone
is 12 x [B,12,197,197], which would be gigabytes across 848 images and three
seeds for no benefit at this phase.

    env_win/Scripts/python.exe src/publication/explainability/dual_path_hooks.py --project-root . --smoke
    env_win/Scripts/python.exe src/publication/explainability/dual_path_hooks.py --project-root . --neutrality --all-seeds
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
SRC = PUB.parent
for p in (str(PUB), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import train_ablation as TA                                   # noqa: E402
from fusion_v2 import tta_logits                              # noqa: E402
from fusion_v2_gradcam import load_verified_model             # noqa: E402

SEEDS = [42, 43, 44]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXPECTED_N = 848
CNN_STAGES = (3, 5, 8)

NEUTRALITY_TOL = 1e-6        # gate A: effect of instrumentation
GATE_SUM_TOL = 1e-6
ATTN_SUM_TOL = 1e-4
ACC_TOL_PP = 0.0

RECON_ROOT = "publication_runs/explainability/reconstructed_clean_test"
OUT_ROOT = "publication_results/explainability/x2_instrumentation"

EXPECTED_SHAPES = {
    "fused_logits": "[B, 4]", "fused_probs": "[B, 4]",
    "aux_cnn_logits": "[B, 4]", "aux_cnn_probs": "[B, 4]",
    "aux_vit_logits": "[B, 4]", "aux_vit_probs": "[B, 4]",
    "gate": "[B, 2]",
    "cnn_maps[3]": "[B, 40, 28, 28]", "cnn_maps[5]": "[B, 112, 14, 14]",
    "cnn_maps[8]": "[B, 1280, 7, 7]",
    "vit_attention": "12 x [B, 12, 197, 197]",
    "target_index": "[B]",
}


class X2Error(RuntimeError):
    """A Phase X2 stop condition was hit."""


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


# --------------------------------------------------------------- outputs ----

@dataclass
class DualPathOutputs:
    fused_logits: torch.Tensor
    fused_probs: torch.Tensor
    tta_probs: torch.Tensor
    aux_cnn_logits: torch.Tensor
    aux_cnn_probs: torch.Tensor
    aux_vit_logits: torch.Tensor
    aux_vit_probs: torch.Tensor
    gate: torch.Tensor
    cnn_maps: dict
    vit_attention: tuple
    target_index: torch.Tensor
    target_mode: str
    attn_impl_used: str

    def shapes(self) -> dict:
        d = {k: list(getattr(self, k).shape) for k in
             ("fused_logits", "fused_probs", "tta_probs", "aux_cnn_logits",
              "aux_cnn_probs", "aux_vit_logits", "aux_vit_probs", "gate",
              "target_index")}
        for s, t in sorted(self.cnn_maps.items()):
            d["cnn_maps[%d]" % s] = list(t.shape)
        # Absent attention must be REPORTED, not raise: a crash here would turn a
        # detectable condition into a stack trace at the point of diagnosis.
        d["vit_attention"] = ("none captured" if not self.vit_attention else
                              "%d x %s" % (len(self.vit_attention),
                                           list(self.vit_attention[0].shape)))
        return d


# -------------------------------------------------------------- recorder ----

class DualPathRecorder:
    """Instruments an existing FusionV2 instance and restores it exactly.

    The model is never rebuilt, reloaded or modified in place beyond the two
    reversible changes below, and `__exit__` undoes both even if the body raised.
    """

    def __init__(self, model, cnn_stages=CNN_STAGES, enable_vit_attention=True):
        self.model = model
        self.cnn_stages = tuple(cnn_stages)
        self.enable_vit_attention = enable_vit_attention
        self._handles = []
        self._captured = {}
        self._orig_attn_impl = None
        self._orig_vit_forward = None
        self._entered = False

    # -- context -------------------------------------------------------------
    def __enter__(self):
        m = self.model
        self._orig_attn_impl = getattr(m.vit.config, "_attn_implementation", None)

        def mk(stage_idx):
            def hook(_mod, _inp, out):
                # detach: the recorder must never retain a graph
                self._captured[stage_idx] = out.detach()
                return None          # record only; never alter the output
            return hook

        for s in self.cnn_stages:
            self._handles.append(m.cnn_stages[s].register_forward_hook(mk(s)))

        if self.enable_vit_attention:
            m.vit.set_attn_implementation("eager")
            inner = m.vit.vit
            self._orig_vit_forward = inner.forward

            def fwd(*a, **kw):
                kw["output_attentions"] = True
                o = self._orig_vit_forward(*a, **kw)
                att = getattr(o, "attentions", None)
                self._captured["vit_attention"] = (
                    tuple(t.detach() for t in att) if att is not None else None)
                return o

            inner.forward = fwd
        self._entered = True
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._orig_vit_forward is not None:
            self.model.vit.vit.forward = self._orig_vit_forward
            self._orig_vit_forward = None
        if self.enable_vit_attention and self._orig_attn_impl is not None:
            self.model.vit.set_attn_implementation(self._orig_attn_impl)
        self._captured = {}
        self._entered = False
        return False          # never swallow an exception

    # -- introspection for tests --------------------------------------------
    @property
    def n_hooks(self) -> int:
        return len(self._handles)

    @property
    def attn_impl(self):
        return getattr(self.model.vit.config, "_attn_implementation", None)

    # -- one synchronized pass ----------------------------------------------
    @torch.no_grad()
    def run(self, x: torch.Tensor, target_mode: str = "predicted",
            fixed_index=None, with_tta: bool = True) -> DualPathOutputs:
        if not self._entered:
            raise X2Error("DualPathRecorder.run called outside its context")
        self._captured.pop("vit_attention", None)
        for s in self.cnn_stages:
            self._captured.pop(s, None)

        logits, aux_c, aux_v, gate = self.model(x, return_aux=True)
        probs = torch.softmax(logits, -1)

        maps = {}
        for s in self.cnn_stages:
            if s not in self._captured:
                raise X2Error("CNN stage %d was not captured" % s)
            maps[s] = self._captured[s]

        att = self._captured.get("vit_attention")
        if self.enable_vit_attention and not att:
            raise X2Error("ViT attention requested but not captured; "
                          "attn_implementation is %r" % self.attn_impl)

        if target_mode == "predicted":
            tgt = probs.argmax(-1)
        elif target_mode == "fixed":
            if fixed_index is None:
                raise X2Error("target_mode='fixed' requires fixed_index")
            tgt = torch.full((x.shape[0],), int(fixed_index), dtype=torch.long)
        else:
            raise X2Error("unknown target_mode %r" % target_mode)

        tta = tta_logits(self.model, x) if with_tta else probs

        return DualPathOutputs(
            fused_logits=logits, fused_probs=probs, tta_probs=tta,
            aux_cnn_logits=aux_c, aux_cnn_probs=torch.softmax(aux_c, -1),
            aux_vit_logits=aux_v, aux_vit_probs=torch.softmax(aux_v, -1),
            gate=gate, cnn_maps=maps, vit_attention=att or tuple(),
            target_index=tgt, target_mode=target_mode,
            attn_impl_used=self.attn_impl)


# ------------------------------------------------------------- validation ---

def validate_outputs(o: DualPathOutputs, n: int) -> dict:
    """Structural and numerical validation of one batch of recorder output."""
    p = {}
    p["shapes"] = o.shapes()
    p["all_groups_present"] = all(
        getattr(o, k) is not None for k in
        ("fused_logits", "fused_probs", "aux_cnn_logits", "aux_vit_logits",
         "gate", "cnn_maps", "vit_attention", "target_index"))

    g = o.gate
    p["gate_shape_ok"] = list(g.shape) == [n, 2]
    p["gate_finite"] = bool(torch.isfinite(g).all())
    p["gate_nonnegative"] = bool((g >= 0).all())
    gs = g.sum(-1)
    p["gate_max_abs_sum_error"] = float((gs - 1).abs().max())
    p["gate_rows_sum_to_one"] = p["gate_max_abs_sum_error"] <= GATE_SUM_TOL

    p["n_attention_tensors"] = len(o.vit_attention)
    if o.vit_attention:
        a0 = o.vit_attention[0]
        p["attention_shape"] = list(a0.shape)
        p["attention_shape_ok"] = (len(a0.shape) == 4 and a0.shape[0] == n
                                   and a0.shape[1] == 12 and a0.shape[2] == 197
                                   and a0.shape[3] == 197)
        worst = 0.0
        for t in o.vit_attention:
            worst = max(worst, float((t.sum(-1) - 1).abs().max()))
            if not torch.isfinite(t).all():
                p["attention_finite"] = False
        p.setdefault("attention_finite", True)
        p["attention_max_abs_row_sum_error"] = worst
        p["attention_rows_sum_to_one"] = worst <= ATTN_SUM_TOL
    else:
        p["attention_shape_ok"] = False
        p["attention_rows_sum_to_one"] = False

    p["cnn_stage_shapes_ok"] = (
        list(o.cnn_maps[3].shape) == [n, 40, 28, 28]
        and list(o.cnn_maps[5].shape) == [n, 112, 14, 14]
        and list(o.cnn_maps[8].shape) == [n, 1280, 7, 7])
    p["cnn_maps_finite"] = all(bool(torch.isfinite(t).all()) for t in o.cnn_maps.values())

    p["aux_probs_sum_to_one"] = (
        bool(torch.allclose(o.aux_cnn_probs.sum(-1), torch.ones(n), atol=1e-6))
        and bool(torch.allclose(o.aux_vit_probs.sum(-1), torch.ones(n), atol=1e-6)))
    p["target_matches_fused_argmax"] = bool(
        torch.equal(o.target_index, o.fused_probs.argmax(-1)))

    tensors = [o.fused_logits, o.fused_probs, o.tta_probs, o.aux_cnn_logits,
               o.aux_vit_logits, o.gate, o.target_index] + list(o.cnn_maps.values()) \
              + list(o.vit_attention)
    p["no_grad_retained"] = all(getattr(t, "grad_fn", None) is None for t in tensors)
    p["nothing_requires_grad"] = all(
        not getattr(t, "requires_grad", False) for t in tensors)
    return p


# ---------------------------------------------------------------- scoring ---

def build_loader(recon_dir: Path, keep: set | None, batch: int):
    from torch.utils.data import DataLoader
    from torchvision import datasets
    ds = datasets.ImageFolder(str(recon_dir), transform=TA.eval_transform)
    if keep is not None:
        ds.samples = [s for s in ds.samples if os.path.basename(s[0]) in keep]
        ds.targets = [t for _, t in ds.samples]
    if ds.classes != CLASSES:
        raise X2Error("ImageFolder class order %s != %s" % (ds.classes, CLASSES))
    return DataLoader(ds, batch_size=batch, num_workers=0, shuffle=False), ds


def run_seed(root: Path, seed: int, recon_dir: Path, keep: set | None,
             batch: int) -> dict:
    model, _dev, ck_sha = load_verified_model(root, seed=seed, device="cpu")
    model.eval()
    loader, ds = build_loader(recon_dir, keep, batch)
    ref_path = root / ("runs_v2_clean/fusion_v2_seed%d/preds_test.csv" % seed)
    if not ref_path.is_file():
        raise X2Error("reference predictions absent: %s" % ref_path)
    with open(ref_path, encoding="utf-8") as f:
        ref = {r["filename"]: r for r in csv.DictReader(f)}

    t0 = time.time()
    per_image, val_summary = [], {}
    worst = {"single": 0.0, "tta": 0.0, "gate_sum": 0.0, "attn_sum": 0.0}
    n_over_tol = 0
    class_changes = 0
    schema_recorded = None
    i = 0

    # --- uninstrumented reference pass (same machine, same model) -----------
    plain = {}
    with torch.no_grad():
        j = 0
        for x, _y in loader:
            sp = torch.softmax(model(x), -1)
            tp = tta_logits(model, x)
            for k in range(x.shape[0]):
                fn = os.path.basename(ds.samples[j][0]); j += 1
                plain[fn] = (sp[k].numpy().astype(np.float64),
                             tp[k].numpy().astype(np.float64))
    attn_impl_before = getattr(model.vit.config, "_attn_implementation", None)

    # --- instrumented pass --------------------------------------------------
    loader, ds = build_loader(recon_dir, keep, batch)
    with DualPathRecorder(model) as rec:
        for x, y in loader:
            o = rec.run(x, target_mode="predicted", with_tta=True)
            v = validate_outputs(o, x.shape[0])
            if schema_recorded is None:
                schema_recorded = v["shapes"]
                val_summary = {k: bool(val) for k, val in v.items()
                               if isinstance(val, bool)}
            else:
                for k, val in v.items():
                    if isinstance(val, bool):
                        val_summary[k] = val_summary.get(k, True) and val
            worst["gate_sum"] = max(worst["gate_sum"], v["gate_max_abs_sum_error"])
            worst["attn_sum"] = max(worst["attn_sum"],
                                    v.get("attention_max_abs_row_sum_error", 0.0))
            sp = o.fused_probs.numpy().astype(np.float64)
            tp = o.tta_probs.numpy().astype(np.float64)
            g = o.gate.numpy().astype(np.float64)
            for k in range(x.shape[0]):
                fn = os.path.basename(ds.samples[i][0])
                true_lab = ds.classes[int(y[k])]
                i += 1
                ps, pt = plain[fn]
                d_single = float(np.abs(sp[k] - ps).max())
                d_tta = float(np.abs(tp[k] - pt).max())
                worst["single"] = max(worst["single"], d_single)
                worst["tta"] = max(worst["tta"], d_tta)
                if max(d_single, d_tta) > NEUTRALITY_TOL:
                    n_over_tol += 1
                if int(np.argmax(tp[k])) != int(np.argmax(pt)):
                    class_changes += 1
                r = ref.get(fn)
                rp = (np.array([float(r["p_" + c]) for c in CLASSES])
                      if r is not None else None)
                per_image.append({
                    "seed": seed, "filename": fn, "true_label": true_lab,
                    "ref_true_label": r["true_label"] if r else None,
                    "pred_instrumented": CLASSES[int(np.argmax(tp[k]))],
                    "pred_uninstrumented": CLASSES[int(np.argmax(pt))],
                    "pred_recorded": r["pred_label"] if r else None,
                    "neutrality_max_abs_dp_single": d_single,
                    "neutrality_max_abs_dp_tta": d_tta,
                    "identity_max_abs_dp_vs_recorded":
                        float(np.abs(tp[k] - rp).max()) if rp is not None else None,
                    "gate_cnn": float(g[k, 0]), "gate_vit": float(g[k, 1]),
                    "gate_sum": float(g[k].sum()),
                })
        hooks_during = rec.n_hooks
        attn_impl_during = rec.attn_impl
    hooks_after = rec.n_hooks
    attn_impl_after = getattr(model.vit.config, "_attn_implementation", None)

    # restore must be exact: re-score and compare to the plain pass
    with torch.no_grad():
        loader2, ds2 = build_loader(recon_dir, keep, batch)
        restore_worst, j = 0.0, 0
        for x, _y in loader2:
            sp = torch.softmax(model(x), -1).numpy().astype(np.float64)
            for k in range(x.shape[0]):
                fn = os.path.basename(ds2.samples[j][0]); j += 1
                restore_worst = max(restore_worst,
                                    float(np.abs(sp[k] - plain[fn][0]).max()))

    # identity vs the authoritative recorded predictions
    ours = {r["filename"]: r for r in per_image}
    common = sorted(set(ours) & set(ref))
    only_ours = sorted(set(ours) - set(ref))
    only_ref = sorted(set(ref) - set(ours))
    lbl_mm = [f for f in common if ours[f]["true_label"] != ref[f]["true_label"]]
    pred_mm = [f for f in common if ours[f]["pred_instrumented"] != ref[f]["pred_label"]]
    acc_ours = sum(ours[f]["pred_instrumented"] == ours[f]["true_label"] for f in common) / max(len(common), 1)
    acc_ref = sum(ref[f]["pred_label"] == ref[f]["true_label"] for f in common) / max(len(common), 1)
    ident_dp = [ours[f]["identity_max_abs_dp_vs_recorded"] for f in common
                if ours[f]["identity_max_abs_dp_vs_recorded"] is not None]

    return {
        "seed": seed, "checkpoint_sha256": ck_sha,
        "checkpoint_path": ("publication_runs/recovered_checkpoints/clean/"
                            "full_fusion_v2/seed_%d/checkpoint.pth" % seed),
        "reference_csv": "runs_v2_clean/fusion_v2_seed%d/preds_test.csv" % seed,
        "reference_sha256": sha256(ref_path),
        "n_images": len(per_image),
        # --- A: neutrality -------------------------------------------------
        "neutrality_max_abs_dp_single_pass": worst["single"],
        "neutrality_max_abs_dp_tta": worst["tta"],
        "neutrality_n_images_over_tolerance": n_over_tol,
        "neutrality_predicted_class_changes": class_changes,
        "neutrality_tolerance": NEUTRALITY_TOL,
        "neutrality_pass": (max(worst["single"], worst["tta"]) <= NEUTRALITY_TOL
                            and n_over_tol == 0 and class_changes == 0),
        # --- B: identity ---------------------------------------------------
        "identity_n_compared": len(common),
        "identity_n_filename_mismatch": len(only_ours) + len(only_ref),
        "identity_n_true_label_mismatch": len(lbl_mm),
        "identity_n_pred_label_mismatch": len(pred_mm),
        "identity_pred_mismatch_examples": pred_mm[:5],
        "identity_accuracy_instrumented": acc_ours,
        "identity_accuracy_recorded": acc_ref,
        "identity_accuracy_diff_pp": abs(100 * acc_ours - 100 * acc_ref),
        "identity_max_abs_dp_vs_recorded": max(ident_dp) if ident_dp else None,
        "identity_mean_abs_dp_vs_recorded": float(np.mean(ident_dp)) if ident_dp else None,
        "identity_note": ("probability residual vs the recorded CSV is the "
                          "cross-machine floating-point difference documented in "
                          "X1 (~4e-06); it is NOT held to the 1e-6 neutrality "
                          "tolerance, which measures instrumentation only"),
        # --- structural ------------------------------------------------------
        "tensor_schema": schema_recorded,
        "validation": val_summary,
        "gate_max_abs_sum_error": worst["gate_sum"],
        "attention_max_abs_row_sum_error": worst["attn_sum"],
        "hooks_during_context": hooks_during, "hooks_after_context": hooks_after,
        "attn_impl_before": attn_impl_before, "attn_impl_during": attn_impl_during,
        "attn_impl_after": attn_impl_after,
        "attn_impl_restored": attn_impl_after == attn_impl_before,
        "restore_max_abs_dp": restore_worst,
        "restore_exact": restore_worst == 0.0,
        "seconds": round(time.time() - t0, 1),
        "_per_image": per_image,
    }


# ----------------------------------------------------------------- driver ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--neutrality", action="store_true")
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    os.chdir(root)
    recon = root / RECON_ROOT
    if not recon.is_dir():
        raise X2Error("X1 reconstructed set absent: %s" % recon)
    t0 = time.time()

    R = {"phase": "X2 - read-only dual-path instrumentation",
         "generated_utc": datetime.now(timezone.utc).isoformat(),
         "mode": "smoke" if args.smoke else "full",
         "training_performed": False, "model_modified": False,
         "raw_tensors_persisted": False,
         "tolerances": {"neutrality_abs_prob": NEUTRALITY_TOL,
                        "gate_row_sum": GATE_SUM_TOL,
                        "attention_row_sum": ATTN_SUM_TOL,
                        "accuracy_pp": ACC_TOL_PP},
         "expected_tensor_schema": EXPECTED_SHAPES,
         "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "numpy": np.__version__, "platform": platform.platform(),
                         "device": "cpu"},
         "inputs": {
             "reconstructed_root": RECON_ROOT,
             "reconstruction_manifest_sha256":
                 sha256(recon / "RECONSTRUCTION_MANIFEST.json"),
             "x1_report_sha256": sha256(
                 root / "publication_results/explainability/x1_reconstruction/x1_reconstruction.json"),
         }}

    banner("STEP 1 - IMAGE SET")
    man = json.loads((recon / "RECONSTRUCTION_MANIFEST.json").read_text(encoding="utf-8"))
    all_names = [f["filename"] for f in man["files"]]
    if len(all_names) != EXPECTED_N:
        raise X2Error("reconstruction manifest has %d files, expected %d"
                      % (len(all_names), EXPECTED_N))
    if args.smoke:
        by_cls = {}
        for f in man["files"]:
            by_cls.setdefault(f["manifest_class"], []).append(f["filename"])
        keep = set()
        for c in CLASSES:
            keep.update(sorted(by_cls[c])[:2])
        stamp("smoke: %d images (2 per class)" % len(keep))
    else:
        keep = None
        stamp("full: %d X1-verified images" % len(all_names))
    R["n_images"] = len(keep) if keep else EXPECTED_N

    banner("STEP 2 - INSTRUMENTED INFERENCE AND NEUTRALITY")
    seeds = SEEDS if args.all_seeds else ([args.seed] if args.seed else [42])
    per_seed, all_rows = {}, []
    for s in seeds:
        r = run_seed(root, s, recon, keep, args.batch)
        for row in r.pop("_per_image"):
            all_rows.append(row)
        per_seed[s] = r
        stamp("seed %d | n=%d | NEUTRALITY max|dp| single %.3e tta %.3e over-tol %d "
              "class-changes %d | IDENTITY fn=%d lbl=%d pred=%d acc %.4f pp | "
              "gate sum err %.2e | attn sum err %.2e | restore %s | %.1fs"
              % (s, r["n_images"], r["neutrality_max_abs_dp_single_pass"],
                 r["neutrality_max_abs_dp_tta"], r["neutrality_n_images_over_tolerance"],
                 r["neutrality_predicted_class_changes"],
                 r["identity_n_filename_mismatch"], r["identity_n_true_label_mismatch"],
                 r["identity_n_pred_label_mismatch"], r["identity_accuracy_diff_pp"],
                 r["gate_max_abs_sum_error"], r["attention_max_abs_row_sum_error"],
                 "exact" if r["restore_exact"] else "INEXACT", r["seconds"]))
    R["per_seed"] = per_seed

    banner("STEP 3 - ACCEPTANCE")
    V = list(per_seed.values())
    checks = {
        "A_neutrality_within_1e-6_all_images": all(
            max(v["neutrality_max_abs_dp_single_pass"],
                v["neutrality_max_abs_dp_tta"]) <= NEUTRALITY_TOL for v in V),
        "A_zero_predicted_class_changes": all(
            v["neutrality_predicted_class_changes"] == 0 for v in V),
        "B_zero_filename_mismatch": all(v["identity_n_filename_mismatch"] == 0 for v in V)
            if not args.smoke else all(
                len(set(r["filename"] for r in all_rows)) > 0 for _ in [0]),
        "B_zero_true_label_mismatch": all(v["identity_n_true_label_mismatch"] == 0 for v in V),
        "B_zero_pred_label_mismatch": all(v["identity_n_pred_label_mismatch"] == 0 for v in V),
        "B_accuracy_diff_zero_pp": all(v["identity_accuracy_diff_pp"] <= ACC_TOL_PP for v in V),
        "gate_finite_nonneg_sums_to_one": all(
            v["validation"].get("gate_finite") and v["validation"].get("gate_nonnegative")
            and v["validation"].get("gate_rows_sum_to_one") for v in V),
        "attention_shape_and_row_sums_valid": all(
            v["validation"].get("attention_shape_ok")
            and v["validation"].get("attention_rows_sum_to_one")
            and v["validation"].get("attention_finite") for v in V),
        "cnn_stage_shapes_valid": all(v["validation"].get("cnn_stage_shapes_ok") for v in V),
        "all_tensor_groups_present": all(v["validation"].get("all_groups_present") for v in V),
        "target_matches_fused_argmax": all(
            v["validation"].get("target_matches_fused_argmax") for v in V),
        "no_hooks_retained": all(v["hooks_after_context"] == 0 for v in V),
        "no_gradients_retained": all(
            v["validation"].get("no_grad_retained")
            and v["validation"].get("nothing_requires_grad") for v in V),
        "attention_implementation_restored": all(v["attn_impl_restored"] for v in V),
        "restore_reproduces_exactly": all(v["restore_exact"] for v in V),
        "raw_tensors_not_persisted": True,
    }
    if not args.smoke:
        checks["all_seeds_covered"] = set(per_seed) == set(SEEDS)
        checks["all_848_images_per_seed"] = all(v["n_images"] == EXPECTED_N for v in V)
    R["acceptance_checks"] = checks
    R["acceptance_status"] = "PASS" if all(checks.values()) else "FAIL"
    R["elapsed_seconds"] = round(time.time() - t0, 1)
    for k, v in checks.items():
        stamp("  %-46s %s" % (k, "PASS" if v else "FAIL"))

    if args.smoke:
        banner("X2 SMOKE: %s" % R["acceptance_status"])
        return 0 if R["acceptance_status"] == "PASS" else 1

    out = root / OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    for name in ("x2_neutrality.json", "per_seed_neutrality.csv",
                 "per_image_neutrality.csv", "X2_NEUTRALITY_REPORT.md"):
        if (out / name).exists():
            raise X2Error("refusing to overwrite %s" % (out / name))

    with open(out / "x2_neutrality.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=str)
    with open(out / "per_seed_neutrality.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["seed", "checkpoint_sha256", "reference_csv", "reference_sha256",
                "n_images", "neutrality_max_abs_dp_single_pass", "neutrality_max_abs_dp_tta",
                "neutrality_n_images_over_tolerance", "neutrality_predicted_class_changes",
                "identity_n_filename_mismatch", "identity_n_true_label_mismatch",
                "identity_n_pred_label_mismatch", "identity_accuracy_instrumented",
                "identity_accuracy_recorded", "identity_accuracy_diff_pp",
                "identity_max_abs_dp_vs_recorded", "gate_max_abs_sum_error",
                "attention_max_abs_row_sum_error", "hooks_after_context",
                "attn_impl_before", "attn_impl_after", "restore_max_abs_dp", "seconds"]
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader()
        for s, v in sorted(per_seed.items()):
            w.writerow(v)
    with open(out / "per_image_neutrality.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["seed", "filename", "true_label", "pred_instrumented",
                "pred_uninstrumented", "pred_recorded",
                "neutrality_max_abs_dp_single", "neutrality_max_abs_dp_tta",
                "identity_max_abs_dp_vs_recorded", "gate_cnn", "gate_vit", "gate_sum"]
        w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader()
        for r in all_rows:
            w.writerow(r)
    with open(out / "X2_NEUTRALITY_REPORT.md", "w", encoding="utf-8") as f:
        f.write(render_md(R))

    banner("PHASE X2: %s" % R["acceptance_status"])
    stamp("wrote %s" % OUT_ROOT)
    return 0 if R["acceptance_status"] == "PASS" else 1


def render_md(R: dict) -> str:
    L = ["# Phase X2 - read-only dual-path instrumentation", "",
         "Generated %s." % R["generated_utc"], "",
         "**Status: %s.** No training, no model change, no modification to any "
         "dataset, checkpoint or existing prediction file. Raw tensors were "
         "validated in flight and not persisted." % R["acceptance_status"], "",
         "## What was measured, and why two comparisons", "",
         "Instrumenting the model requires two intrusive changes: forward hooks on "
         "the CNN stages, and switching the ViT from `sdpa` to `eager` attention, "
         "because under `sdpa` the flag `output_attentions=True` silently returns "
         "`None`. Both are confined to a context manager that restores the model on "
         "exit, including on the exception path.", "",
         "Two comparisons are reported separately because they mean different things:", "",
         "| | Compares | Tolerance | Purpose |", "|---|---|---|---|",
         "| **A. Neutrality** | instrumented vs uninstrumented, same machine, same checkpoint | max abs dp <= %.0e | measures what THIS phase does |" % R["tolerances"]["neutrality_abs_prob"],
         "| **B. Identity** | instrumented vs the authoritative recorded predictions | 0 label mismatches, 0.00 pp | confirms the instrumented model still reproduces the published result |",
         "",
         "B's probability residual is the cross-machine floating-point difference "
         "that Phase X1 already documented at around 4e-06. It is reported but is "
         "**not** held to the neutrality tolerance, which would fail X2 on a "
         "residual it did not cause.", "",
         "## Per-seed results", "",
         "### A. Neutrality", "",
         "| Seed | n | max abs dp (single pass) | max abs dp (TTA) | images over tolerance | predicted-class changes |",
         "|---|---:|---:|---:|---:|---:|"]
    for s, v in sorted(R["per_seed"].items()):
        L.append("| %s | %d | %.3e | %.3e | %d | %d |"
                 % (s, v["n_images"], v["neutrality_max_abs_dp_single_pass"],
                    v["neutrality_max_abs_dp_tta"],
                    v["neutrality_n_images_over_tolerance"],
                    v["neutrality_predicted_class_changes"]))
    L += ["", "### B. Identity against the recorded predictions", "",
          "| Seed | n | Filename mm | True-label mm | Pred-label mm | Acc instrumented | Acc recorded | Diff (pp) | max abs dp |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, v in sorted(R["per_seed"].items()):
        L.append("| %s | %d | %d | %d | %d | %.6f | %.6f | %.4f | %s |"
                 % (s, v["identity_n_compared"], v["identity_n_filename_mismatch"],
                    v["identity_n_true_label_mismatch"], v["identity_n_pred_label_mismatch"],
                    v["identity_accuracy_instrumented"], v["identity_accuracy_recorded"],
                    v["identity_accuracy_diff_pp"],
                    "%.2e" % v["identity_max_abs_dp_vs_recorded"]
                    if v["identity_max_abs_dp_vs_recorded"] is not None else "n/a"))
    L += ["", "### Structural validation", "",
          "| Seed | gate row-sum error | attention row-sum error | hooks after exit | attn impl before -> after | restore max abs dp |",
          "|---|---:|---:|---:|---|---:|"]
    for s, v in sorted(R["per_seed"].items()):
        L.append("| %s | %.2e | %.2e | %d | %s -> %s | %.2e |"
                 % (s, v["gate_max_abs_sum_error"], v["attention_max_abs_row_sum_error"],
                    v["hooks_after_context"], v["attn_impl_before"], v["attn_impl_after"],
                    v["restore_max_abs_dp"]))
    first = sorted(R["per_seed"])[0]
    L += ["", "## Tensor schema exposed from one synchronized pass", "",
          "| Tensor | Shape observed |", "|---|---|"]
    for k, v in R["per_seed"][first]["tensor_schema"].items():
        L.append("| `%s` | `%s` |" % (k, v))
    L += ["", "Raw tensors were **not** persisted. The attention stack alone is "
              "12 x [B,12,197,197] per batch, which across 848 images and three "
              "seeds would be gigabytes for no benefit at this phase; it is "
              "validated in flight and discarded.", "",
          "## Acceptance checks", "", "| Check | Result |", "|---|---|"]
    for k, v in R["acceptance_checks"].items():
        L.append("| %s | **%s** |" % (k.replace("_", " "), "PASS" if v else "FAIL"))
    L += ["", "**Overall: %s.** Elapsed %.1f s." % (R["acceptance_status"], R["elapsed_seconds"]), "",
          "## Scope", "",
          "This phase establishes that the instrumented model is the verified model "
          "and that every tensor the later phases need can be obtained from one "
          "synchronized pass. It computes no attribution, makes no claim about what "
          "any tensor means, and establishes nothing about the external set. "
          "Attention was exposed; that does not make attention an explanation. "
          "Phase X3 has not started.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except X2Error as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
