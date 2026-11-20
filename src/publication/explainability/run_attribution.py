"""Phase X3 (amended) - synchronized CNN Grad-CAM++ and ViT Gradient x Input.

One grad-enabled forward per batch captures BOTH the EfficientNet stage-8
activation and the ViT patch-embedding output; one `torch.autograd.grad` call
on the selected fused logit returns the gradient at both sites. Both maps
therefore describe the same fused decision: same image, same checkpoint, same
target index, same forward. SDPA is never changed. No composite is formed.

Out of scope by design (later phases): faithfulness, stability, CNN-ViT
agreement, cross-source / external results.

Usage
-----
    env_win/Scripts/python.exe src/publication/explainability/run_attribution.py --project-root . --smoke
    env_win/Scripts/python.exe src/publication/explainability/run_attribution.py --project-root . --dry-run
    env_win/Scripts/python.exe src/publication/explainability/run_attribution.py --project-root . --all-seeds
"""

from __future__ import annotations

import argparse
import ast
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
for _p in (str(HERE), str(PUB), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_ablation as TA                                   # noqa: E402
from fusion_v2 import tta_logits                              # noqa: E402
from fusion_v2_gradcam import load_verified_model, gradcam_plus_plus  # noqa: E402
import cnn_attribution as CA                                  # noqa: E402
import vit_attribution as VA                                  # noqa: E402

SEEDS = [42, 43, 44]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXPECTED_N = 848
EXPECTED_COUNTS = {"glioma": 243, "meningioma": 228, "notumor": 116,
                   "pituitary": 261}
SMOKE_PER_CLASS = 2
NEUTRALITY_TOL = 1e-6
EQUIV_TOL = 1e-5
NPZ_KEYS = {"cnn_map", "vit_map", "target_index", "target_mode",
            "target_class", "filename", "true_label"}

RECON_ROOT = "publication_runs/explainability/reconstructed_clean_test"
OUT_ROOT = "publication_results/explainability/x3_attribution"
MAP_ROOT = "publication_runs/explainability/attribution"
BASELINE_MANIFEST = "publication_results/audit/baseline_manifest.json"
MODULES = ("cnn_attribution.py", "vit_attribution.py", "run_attribution.py")


class X3Error(RuntimeError):
    """A Phase X3 stop condition was hit."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_or_absent(path) -> str:
    """Digest a project-host document that a detached compute host may lack.

    The amendment proposal lives under publication_results/ and is provenance,
    not a compute input, so it is absent on e.g. Kaggle. Recording that plainly
    is correct; inventing a digest or crashing the run is not. The true digest is
    re-verified on the project host when results are brought back.
    """
    try:
        return sha256(path)
    except FileNotFoundError:
        return "not available on this host"


def stamp(m):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m), flush=True)


def _rel(p, root) -> str:
    try:
        return os.path.relpath(str(p), str(root)).replace("\\", "/")
    except ValueError:
        return os.path.abspath(str(p)).replace("\\", "/")


def source_gate(path) -> dict:
    """AST check: no call to set_attn_implementation, no output_attentions kwarg."""
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


def state_dict_digest(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------ attributor ----

class SynchronizedAttributor:
    """Hooks CNN stage 8 and the ViT patch embedding; one forward, one grad."""

    def __init__(self, model):
        self.model = model
        self._handles, self._cap = [], {}
        self._entered = False

    def __enter__(self):
        def mk(key):
            def hook(_m, _i, out):
                self._cap[key] = out[0] if isinstance(out, tuple) else out
                return out
            return hook
        m = self.model
        self._handles.append(m.cnn_stages[-1].register_forward_hook(mk("cnn")))
        self._handles.append(m.vit.vit.embeddings.register_forward_hook(mk("vit")))
        self._entered = True
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles, self._entered = [], False
        self._cap.clear()
        return False

    @property
    def n_hooks(self):
        return len(self._handles)

    def attn_impl(self):
        return getattr(self.model.vit.config, "_attn_implementation", None)

    def attribute(self, x, target_index=None):
        if not self._entered:
            raise X3Error("attribute() called outside context")
        self._cap.clear()
        logits = self.model(x)                       # autograd enabled
        a_cnn, h_vit = self._cap.get("cnn"), self._cap.get("vit")
        if a_cnn is None or h_vit is None:
            raise X3Error("a hook captured nothing")
        probs = logits.softmax(-1)
        tgt = probs.argmax(-1) if target_index is None else target_index.long()
        sel = logits.gather(1, tgt.view(-1, 1)).sum()
        g_cnn, g_vit = torch.autograd.grad(sel, [a_cnn, h_vit])
        cnn = CA.gradcam_pp_native(a_cnn.detach().cpu().numpy(),
                                   g_cnn.detach().cpu().numpy())
        vit = VA.patch_relevance(g_vit, h_vit).cpu().numpy().astype(np.float32)
        return {"cnn_maps": cnn, "vit_maps": vit,
                "probs": probs.detach(), "target_index": tgt.detach(),
                "cnn_act_shape": tuple(a_cnn.shape),
                "vit_patch_grad_abs_sum": g_vit[:, 1:].abs().sum((1, 2)).detach()}


# --------------------------------------------------------------- running ----

def build_loader(recon_dir, keep, batch):
    from torch.utils.data import DataLoader
    from torchvision import datasets
    ds = datasets.ImageFolder(str(recon_dir), transform=TA.eval_transform)
    if keep is not None:
        ds.samples = [s for s in ds.samples if os.path.basename(s[0]) in keep]
        ds.targets = [t for _, t in ds.samples]
    if ds.classes != CLASSES:
        raise X3Error("class order %s != %s" % (ds.classes, CLASSES))
    return DataLoader(ds, batch_size=batch, num_workers=0, shuffle=False), ds


def run_seed(root, seed, recon, keep, batch, map_dir, equivalence, device="cpu"):
    model, dev, ck = load_verified_model(root, seed=seed, device=device)
    model.eval()
    sd0 = state_dict_digest(model)
    impl0 = model.vit.config._attn_implementation
    if impl0 != "sdpa":
        raise X3Error("attention implementation %r != 'sdpa'" % impl0)
    loader, ds = build_loader(recon, keep, batch)
    order = [os.path.basename(p) for p, _ in ds.samples]
    true_of = {os.path.basename(p): CLASSES[t] for p, t in ds.samples}
    ref_path = root / ("runs_v2_clean/fusion_v2_seed%d/preds_test.csv" % seed)
    with open(ref_path, encoding="utf-8") as f:
        ref = {r["filename"]: r for r in csv.DictReader(f)}

    plain = {}
    with torch.no_grad():
        j = 0
        for x, _ in loader:
            x = x.to(dev)
            sp, tp = model(x).softmax(-1), tta_logits(model, x)
            sp, tp = sp.cpu(), tp.cpu()
            for k in range(x.shape[0]):
                plain[order[j]] = (sp[k].numpy().astype(np.float64),
                                   tp[k].numpy().astype(np.float64)); j += 1

    rows, eq_rows, impls = [], [], {impl0}
    t_attr = 0.0
    map_dir.mkdir(parents=True, exist_ok=True)
    with SynchronizedAttributor(model) as att:
        j = 0
        for x, _ in loader:
            x = x.to(dev)
            n = x.shape[0]; fns = order[j:j + n]; j += n
            tgt = torch.tensor([CLASSES.index(ref[f]["pred_label"]) for f in fns],
                               device=dev)
            t0 = time.time()
            out = att.attribute(x, target_index=tgt)
            t_attr += time.time() - t0
            impls.add(att.attn_impl())
            with torch.no_grad():
                tta = tta_logits(model, x).cpu()
            probs_cpu = out["probs"].cpu()
            for k, fn in enumerate(fns):
                ps = probs_cpu[k].numpy().astype(np.float64)
                pt = tta[k].numpy().astype(np.float64)
                cm, vm = out["cnn_maps"][k], out["vit_maps"][k]
                ti = int(out["target_index"][k])
                valid_c = (cm.shape == (CA.CNN_GRID, CA.CNN_GRID)
                           and bool(np.isfinite(cm).all()) and cm.max() > cm.min())
                valid_v = (vm.shape == (VA.VIT_GRID, VA.VIT_GRID)
                           and bool(np.isfinite(vm).all()) and vm.max() > vm.min()
                           and float(out["vit_patch_grad_abs_sum"][k]) > 0)
                p = map_dir / (os.path.splitext(fn)[0] + ".npz")
                if p.exists():
                    raise X3Error("refusing to overwrite %s" % p)
                np.savez_compressed(p, cnn_map=cm, vit_map=vm,
                                    target_index=np.int64(ti),
                                    target_mode=np.str_("predicted"),
                                    target_class=np.str_(CLASSES[ti]),
                                    filename=np.str_(fn),
                                    true_label=np.str_(true_of[fn]))
                rows.append({
                    "filename": fn, "seed": seed, "true_label": true_of[fn],
                    "recorded_pred_label": ref[fn]["pred_label"],
                    "instrumented_pred_label": CLASSES[int(pt.argmax())],
                    "plain_pred_label": CLASSES[int(plain[fn][1].argmax())],
                    "target_index": ti, "target_class": CLASSES[ti],
                    "cnn_target_index": ti, "vit_target_index": ti,
                    "max_abs_prob_diff_single": float(np.abs(ps - plain[fn][0]).max()),
                    "max_abs_prob_diff_tta": float(np.abs(pt - plain[fn][1]).max()),
                    "cnn_map_valid": bool(valid_c), "vit_map_valid": bool(valid_v),
                    "cnn_map_shape": "x".join(map(str, cm.shape)),
                    "vit_map_shape": "x".join(map(str, vm.shape)),
                    "npz": _rel(p, root), "npz_sha256": sha256(p),
                })
        hooks_inside = att.n_hooks

    if equivalence:
        # Amendment §3: native map rescaled like the library vs the verified
        # helper. The helper calls .backward() and fills parameter .grad, so
        # gradients are cleared afterwards and A9 is checked after that.
        for r in rows:
            fn = r["filename"]
            x1 = torch.stack([ds[order.index(fn)][0]]).to(dev)
            helper = gradcam_plus_plus(model, x1, dev, r["target_index"])
            native = np.load(root / r["npz"])["cnn_map"]
            mine = CA.rescale_like_library(native)
            d = float(np.abs(mine - helper).max())
            eq_rows.append({"filename": fn, "seed": seed,
                            "target_index": r["target_index"],
                            "max_abs_diff_vs_helper": d,
                            "within_tol": d <= EQUIV_TOL})
        model.zero_grad(set_to_none=True)

    acc_i = 100.0 * sum(r["instrumented_pred_label"] == r["true_label"] for r in rows) / len(rows)
    acc_r = 100.0 * sum(r["recorded_pred_label"] == r["true_label"] for r in rows) / len(rows)
    return {
        "seed": seed, "checkpoint_sha256": ck, "n_images": len(rows),
        "batch_size": batch,
        "attn_impls_observed": sorted(str(i) for i in impls | {model.vit.config._attn_implementation}),
        "hooks_inside": hooks_inside, "hooks_after_exit": att.n_hooks,
        "param_grads_all_none": all(p.grad is None for p in model.parameters()),
        "model_still_eval": not model.training,
        "state_dict_unchanged": state_dict_digest(model) == sd0,
        "max_abs_prob_diff_single": max(r["max_abs_prob_diff_single"] for r in rows),
        "max_abs_prob_diff_tta": max(r["max_abs_prob_diff_tta"] for r in rows),
        "pred_label_changes_vs_plain": sum(r["instrumented_pred_label"] != r["plain_pred_label"] for r in rows),
        "pred_label_changes_vs_recorded": sum(r["instrumented_pred_label"] != r["recorded_pred_label"] for r in rows),
        "accuracy_instrumented_pct": acc_i, "accuracy_recorded_pct": acc_r,
        "accuracy_diff_pp": acc_i - acc_r,
        "cnn_maps_valid": sum(r["cnn_map_valid"] for r in rows),
        "vit_maps_valid": sum(r["vit_map_valid"] for r in rows),
        "targets_synchronized": all(r["cnn_target_index"] == r["vit_target_index"] for r in rows),
        "filename_mismatches": sum(r["filename"] not in ref for r in rows),
        "true_label_mismatches": sum(ref[r["filename"]]["true_label"] != r["true_label"] for r in rows),
        "gradcam_equivalence_max_abs_diff": max((e["max_abs_diff_vs_helper"] for e in eq_rows), default=None),
        "gradcam_equivalence_n": len(eq_rows),
        "attribution_seconds": t_attr,
        "attribution_s_per_image": t_attr / len(rows),
        "_rows": rows, "_eq": eq_rows,
    }


def fmt_equiv(v) -> str:
    """Format the Grad-CAM++ equivalence value, which is None by design.

    Per amendment §8 the equivalence check runs on seed 42 only, so seeds 43
    and 44 legitimately have no value. `%.3e` on None raises TypeError, which
    is how this crashed mid-grid; the smoke test never caught it because smoke
    is seed 42 only, where the value is always present.
    """
    return "n/a" if v is None else "%.3e" % v


def verdict(S: dict, skip_preservation: bool):
    """Return (gates_passed, prose). Exit codes must use the BOOLEAN.

    A check whose value is None was not decidable on this host (e.g. the
    preservation baseline is absent on a detached compute host). Such a check is
    excluded from the verdict rather than counted as a pass, and the prose says
    so. Regression guard: the prose carries a caveat suffix, so comparing it to
    "PASS" would report failure on a fully passing run.
    """
    decided = {k: v for k, v in S.items() if v is not None}
    gates_passed = all(decided.values())
    prose = "PASS" if gates_passed else "FAIL"
    if skip_preservation:
        prose += " (PRESERVATION NOT VERIFIED ON THIS HOST)"
    return gates_passed, prose


def preservation(root):
    man = json.loads((root / BASELINE_MANIFEST).read_text(encoding="utf-8"))
    inv = man["inventory"]
    ent = inv["prediction_csvs"] + inv["metric_files"]
    altered = [e["path"] for e in ent if sha256(root / e["path"]) != e["sha256"]]
    return {"checked": len(ent), "altered": len(altered), "altered_paths": altered}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda (cuda changes float32 reduction order)")
    ap.add_argument("--skip-preservation", action="store_true",
                    help="detached host only (e.g. Kaggle); re-verify on project host")
    a = ap.parse_args()
    if a.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise X3Error("--device cuda requested but CUDA is not available")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    root = Path(a.project_root).resolve()
    recon = root / RECON_ROOT
    man = json.loads((recon / "RECONSTRUCTION_MANIFEST.json").read_text(encoding="utf-8"))
    if man["n_files"] != EXPECTED_N or man["per_class_counts"] != EXPECTED_COUNTS:
        raise X3Error("X1 manifest mismatch")

    if a.dry_run:
        print("matrix: %d images x %d seeds x 2 methods x 1 target mode = %d maps"
              % (EXPECTED_N, len(SEEDS), EXPECTED_N * len(SEEDS) * 2))
        return 0
    if not (a.smoke or a.all_seeds):
        ap.error("choose --smoke, --dry-run or --all-seeds")

    if a.smoke:
        by = {c: [] for c in CLASSES}
        for e in sorted(man["files"], key=lambda e: e["filename"]):
            if len(by[e["manifest_class"]]) < SMOKE_PER_CLASS:
                by[e["manifest_class"]].append(e["filename"])
        keep, seeds, sfx = {f for v in by.values() for f in v}, [42], "_smoke"
        map_base = root / (MAP_ROOT + "_smoke")
    else:
        keep, seeds, sfx, map_base = None, SEEDS, "", root / MAP_ROOT

    out = root / OUT_ROOT
    targets = [out / ("x3_attribution%s.json" % sfx),
               out / ("X3_ATTRIBUTION_REPORT%s.md" % sfx),
               out / ("gradcam_equivalence%s.csv" % sfx),
               out / ("per_image_attribution%s.csv" % sfx),
               out / ("sha256_manifest%s.json" % sfx)]
    clash = [str(p) for p in targets if p.exists()] + \
            [str(map_base / str(s)) for s in seeds if (map_base / str(s)).exists()]
    if clash:
        raise X3Error("refusing to overwrite: %s" % clash)

    gates = {m: source_gate(HERE / m) for m in MODULES}
    stamp("source gates: %s" % gates)
    per = []
    for s in seeds:
        stamp("seed %d ..." % s)
        r = run_seed(root, s, recon, keep, a.batch, map_base / str(s),
                     equivalence=(s == 42),   # amendment §8: seed 42 only
                     device=a.device)
        stamp("  dp single=%.3e tta=%.3e | changes %d/%d | cnn %d/%d vit %d/%d | "
              "gradcam equiv max=%s | %.3f s/img"
              % (r["max_abs_prob_diff_single"], r["max_abs_prob_diff_tta"],
                 r["pred_label_changes_vs_plain"], r["pred_label_changes_vs_recorded"],
                 r["cnn_maps_valid"], r["n_images"], r["vit_maps_valid"], r["n_images"],
                 fmt_equiv(r["gradcam_equivalence_max_abs_diff"]),
                 r["attribution_s_per_image"]))
        per.append(r)

    n_exp = SMOKE_PER_CLASS * 4 if a.smoke else EXPECTED_N
    S = {
        "S1_sdpa_throughout": all(r["attn_impls_observed"] == ["sdpa"] for r in per),
        "S2_eager_route_unreachable": all(all(v.values()) for v in gates.values()),
        "S3_cnn_maps_valid_7x7": all(r["cnn_maps_valid"] == r["n_images"] for r in per),
        "S4_vit_maps_valid_14x14": all(r["vit_maps_valid"] == r["n_images"] for r in per),
        "S5_gradcam_equivalence_le_1e-5": any(r["seed"] == 42 for r in per) and all(
            (r["gradcam_equivalence_max_abs_diff"] is not None
             and r["gradcam_equivalence_max_abs_diff"] <= EQUIV_TOL
             and r["gradcam_equivalence_n"] == r["n_images"])
            if r["seed"] == 42 else r["gradcam_equivalence_n"] == 0 for r in per),
        "S6_neutrality": all(r["max_abs_prob_diff_single"] == 0.0
                             and r["max_abs_prob_diff_tta"] <= NEUTRALITY_TOL for r in per),
        "S7_zero_label_changes": all(r["pred_label_changes_vs_plain"] == 0
                                     and r["pred_label_changes_vs_recorded"] == 0
                                     and r["accuracy_diff_pp"] == 0.0 for r in per),
        "S8_targets_synchronized": all(r["targets_synchronized"] for r in per),
        "S9_identity": all(r["filename_mismatches"] == 0 and r["true_label_mismatches"] == 0
                           and r["n_images"] == n_exp for r in per),
        "no_retained_state": all(r["hooks_after_exit"] == 0 and r["param_grads_all_none"]
                                 and r["model_still_eval"] and r["state_dict_unchanged"]
                                 for r in per),
    }
    if a.skip_preservation:
        pres = {"checked": 0, "altered": 0, "altered_paths": [],
                "skipped": True,
                "note": "not verifiable on this host; must be re-verified on the "
                        "project host before the run is accepted"}
        S["preservation"] = None
    else:
        pres = preservation(root)
        pres["skipped"] = False
        S["preservation"] = pres["altered"] == 0
    gates_passed, overall = verdict(S, a.skip_preservation)

    spi = float(np.mean([r["attribution_s_per_image"] for r in per]))
    R = {
        "phase": "X3", "mode": "smoke" if a.smoke else "full",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "amendment": "publication_results/explainability/X3_BRANCH_ATTRIBUTION_AMENDMENT_PROPOSAL.md",
        "amendment_sha256": sha256_or_absent(
            root / "publication_results/explainability/"
                   "X3_BRANCH_ATTRIBUTION_AMENDMENT_PROPOSAL.md"),
        "attn_implementation": "sdpa",
        "cnn_method": "gradcam_plus_plus_native_7x7",
        "vit_method": "gradient_x_input_patch_embedding_14x14",
        "target_mode": "predicted (recorded inference)",
        "composite_formed": False,
        "seeds": seeds, "batch_size": a.batch,
        "class_order": CLASSES,
        "device": a.device,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(),
                        "cuda_available": torch.cuda.is_available(),
                        "cuda_device": (torch.cuda.get_device_name(0)
                                        if torch.cuda.is_available() else None),
                        "transformers": __import__("transformers").__version__},
        "source_gates": gates,
        "per_seed": [{k: v for k, v in r.items() if not k.startswith("_")} for r in per],
        "checks": S, "preservation": pres, "overall": overall,
        "gates_passed": gates_passed,
        "preservation_verified_on_this_host": not a.skip_preservation,
        "measured_attribution_s_per_image": spi,
        "full_grid_extrapolation_min": {
            "attribution_only": spi * EXPECTED_N * 3 / 60,
            "note": "plus plain reference pass, TTA check and equivalence check; "
                    "see report for the measured wall-clock breakdown"},
    }

    out.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(R, indent=2), encoding="utf-8")
    with open(targets[2], "w", newline="", encoding="utf-8") as f:
        eq = [e for r in per for e in r["_eq"]]
        w = csv.DictWriter(f, list(eq[0].keys())); w.writeheader(); w.writerows(eq)
    with open(targets[3], "w", newline="", encoding="utf-8") as f:
        rows = [x for r in per for x in r["_rows"]]
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L = ["# Phase X3 (amended) - %s" % R["mode"], "", "**Overall: %s**" % overall, "",
         "| check | result |", "|---|---|"]
    L += ["| %s | %s |" % (k, "PASS" if v else "FAIL") for k, v in S.items()]
    L += ["", "| seed | dp single | dp TTA | label changes | CNN valid | ViT valid | "
          "Grad-CAM++ equiv max | s/img |", "|---|---|---|---|---|---|---|---|"]
    for r in per:
        L.append("| %d | %.3e | %.3e | %d/%d | %d/%d | %d/%d | %s | %.3f |"
                 % (r["seed"], r["max_abs_prob_diff_single"], r["max_abs_prob_diff_tta"],
                    r["pred_label_changes_vs_plain"], r["pred_label_changes_vs_recorded"],
                    r["cnn_maps_valid"], r["n_images"], r["vit_maps_valid"], r["n_images"],
                    fmt_equiv(r["gradcam_equivalence_max_abs_diff"]),
                    r["attribution_s_per_image"]))
    L += ["", "Separate CNN (7x7) and ViT (14x14) maps only. No composite. No attention",
          "matrix read. Maps are model sensitivity, not localisation; no region may be",
          "described as tumour. Faithfulness, stability, CNN-ViT agreement and external",
          "results are not computed in X3.", ""]
    targets[1].write_text("\n".join(L), encoding="utf-8")
    mani = {_rel(p, root): sha256(p) for p in targets[:4]}
    targets[4].write_text(json.dumps(mani, indent=2), encoding="utf-8")

    print("\nX3 %s: %s" % (R["mode"], overall))
    for k, v in S.items():
        print("  [%s] %s" % ("SKIP" if v is None else ("PASS" if v else "FAIL"), k))
    print("measured attribution: %.3f s/image" % spi)
    return 0 if gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
