"""Phase X5 - quantitative faithfulness and stability evaluation.

Executes the protocol locked in `x5_protocol_lock.json`. The lock's SHA-256 is
verified on every run; if it does not match, nothing is computed. No parameter
in this module may differ from the lock.

Scope, fixed by the lock:
  * deletion and insertion curves + AUC, per method
  * confidence drop and prediction-change rate at top-k
  * stability under six label-preserving transformations
  * CNN-ViT map agreement (descriptive)
  * image-level class-stratified bootstrap CIs

Out of scope: external / cross-source evaluation (Phase X6), and any anatomical,
tumour or clinical claim. SDPA is never changed; no attention matrix is read.

Usage
-----
    python faithfulness.py --project-root . --verify-protocol-lock
    python faithfulness.py --project-root . --smoke
    python faithfulness.py --project-root . --all --device cuda
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
SRC = PUB.parent
for _p in (str(HERE), str(PUB), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_ablation as TA                                   # noqa: E402
from fusion_v2_gradcam import load_verified_model             # noqa: E402
from run_attribution import SynchronizedAttributor            # noqa: E402
import composite as CO                                        # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEEDS = [42, 43, 44]
N_FULL = 848
GRID = 14
NCELL = GRID * GRID

PROTOCOL_LOCK_SHA256 = "a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5"
RECON_ROOT = "publication_runs/explainability/reconstructed_clean_test"
PROTO_ROOT = "publication_results/explainability/x5_protocol"
OUT_ROOT = "publication_results/explainability/x5_faithfulness"
BASELINE_MANIFEST = "publication_results/audit/baseline_manifest.json"

METHODS = ["cnn_gradcampp", "vit_grad_x_input", "composite_equal",
           "composite_gate_weighted", "baseline_random", "baseline_centre_prior"]


class X5Error(RuntimeError):
    """A Phase X5 stop condition was hit."""


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


def verify_lock(root: Path) -> dict:
    p = root / PROTO_ROOT / "x5_protocol_lock.json"
    if not p.is_file():
        raise X5Error("protocol lock absent: %s" % p)
    h = sha256(p)
    if h != PROTOCOL_LOCK_SHA256:
        raise X5Error("protocol lock hash mismatch\n  expected %s\n  got      %s"
                      % (PROTOCOL_LOCK_SHA256, h))
    proto = json.loads(p.read_text(encoding="utf-8"))
    ids = [m["id"] for m in proto["methods"]]
    if ids != METHODS:
        raise X5Error("method set differs from the lock: %s" % ids)
    amd = root / PROTO_ROOT / "x5_scale_decision.json"
    amd_h = sha256(amd) if amd.is_file() else None
    return {"protocol_lock_sha256": h, "scale_amendment_sha256": amd_h,
            "protocol": proto}


# ---------------------------------------------------------------- masking ---

def build_masked(x1: torch.Tensor, order: torch.Tensor, ks, start_masked: bool):
    """One batch of len(ks) variants. Nearest 16x16 blocks, mask value 0.0."""
    n = len(ks)
    grid = torch.zeros(n, NCELL, device=x1.device)
    for i, k in enumerate(ks):
        if k > 0:
            grid[i, order[:k]] = 1.0
    up = F.interpolate(grid.reshape(n, 1, GRID, GRID), size=(224, 224), mode="nearest")
    xb = x1.expand(n, -1, -1, -1)
    return xb * up if start_masked else xb * (1.0 - up)


def curve_auc(y: np.ndarray) -> float:
    """Trapezoidal AUC over equally spaced points, x scaled to [0,1]."""
    return float(np.trapezoid(y, dx=1.0 / (len(y) - 1)))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.reshape(-1).double().cpu().numpy()
    y = b.reshape(-1).double().cpu().numpy()
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------ transforms ----

def apply_transform(x: torch.Tensor, tid: str, gen: torch.Generator) -> torch.Tensor:
    if tid == "hflip":
        return torch.flip(x, dims=[3])
    if tid in ("rot_plus10", "rot_minus10"):
        deg = 10.0 if tid == "rot_plus10" else -10.0
        th = torch.tensor(deg * np.pi / 180.0, dtype=torch.float32)
        m = torch.tensor([[torch.cos(th), -torch.sin(th), 0.0],
                          [torch.sin(th), torch.cos(th), 0.0]], dtype=torch.float32)
        g = F.affine_grid(m[None].to(x.device).expand(x.shape[0], -1, -1),
                          list(x.shape), align_corners=False)
        return F.grid_sample(x, g, mode="bilinear", padding_mode="zeros",
                             align_corners=False)
    if tid in ("scale_plus5", "scale_minus5"):
        s = 1.0 / 1.05 if tid == "scale_plus5" else 1.0 / 0.95
        m = torch.tensor([[s, 0.0, 0.0], [0.0, s, 0.0]], dtype=torch.float32)
        g = F.affine_grid(m[None].to(x.device).expand(x.shape[0], -1, -1),
                          list(x.shape), align_corners=False)
        return F.grid_sample(x, g, mode="bilinear", padding_mode="zeros",
                             align_corners=False)
    if tid == "noise":
        return x + torch.normal(0.0, 0.01, size=x.shape, generator=gen,
                                device=x.device, dtype=x.dtype)
    raise X5Error("unknown transformation %r" % tid)


def invert_map_transform(m: torch.Tensor, tid: str) -> torch.Tensor:
    """Undo the geometric part on the 14x14 map so maps are comparable."""
    if tid == "hflip":
        return torch.flip(m, dims=[1])
    if tid in ("rot_plus10", "rot_minus10"):
        deg = -10.0 if tid == "rot_plus10" else 10.0
        th = torch.tensor(deg * np.pi / 180.0, dtype=torch.float32)
        a = torch.tensor([[torch.cos(th), -torch.sin(th), 0.0],
                          [torch.sin(th), torch.cos(th), 0.0]], dtype=torch.float32)
        g = F.affine_grid(a[None].to(m.device), [1, 1, GRID, GRID], align_corners=False)
        return F.grid_sample(m[None, None], g, mode="bilinear",
                             padding_mode="zeros", align_corners=False)[0, 0]
    if tid in ("scale_plus5", "scale_minus5"):
        s = 1.05 if tid == "scale_plus5" else 0.95
        a = torch.tensor([[s, 0.0, 0.0], [0.0, s, 0.0]], dtype=torch.float32)
        g = F.affine_grid(a[None].to(m.device), [1, 1, GRID, GRID], align_corners=False)
        return F.grid_sample(m[None, None], g, mode="bilinear",
                             padding_mode="zeros", align_corners=False)[0, 0]
    return m            # noise is not geometric


# ------------------------------------------------------------- evaluation ---

def method_maps(cnn7, vit14, gate_row, rnd_gen):
    c14 = CO.to_common(torch.as_tensor(cnn7))
    v14 = CO.to_common(torch.as_tensor(vit14))
    return {
        "cnn_gradcampp": c14,
        "vit_grad_x_input": v14,
        "composite_equal": CO.equal_weight(c14, v14),
        "composite_gate_weighted": CO.gate_weighted(c14, v14, float(gate_row[0]),
                                                    float(gate_row[1])),
        "baseline_random": CO.random_map(rnd_gen),
        "baseline_centre_prior": CO.centre_prior(),
    }


def evaluate_image(model, x1, maps, tgt, proto, dev):
    steps = proto["deletion"]["steps"]
    topk = proto["top_k_percent"]
    del_ks = [int(round(NCELL * s / steps)) for s in range(steps + 1)]
    ins_ks = list(del_ks)
    tk_ks = [int(round(NCELL * p / 100.0)) for p in topk]
    out = {}
    with torch.no_grad():
        base = float(model(x1).softmax(-1)[0, tgt])
        for mid, m in maps.items():
            order = torch.argsort(m.reshape(-1).to(dev), descending=True, stable=True)
            xb = torch.cat([build_masked(x1, order, del_ks, False),
                            build_masked(x1, order, ins_ks, True),
                            build_masked(x1, order, tk_ks, False)], 0)
            p = model(xb).softmax(-1)[:, tgt].float().cpu().numpy()
            nd, ni = len(del_ks), len(ins_ks)
            dele, ins, tk = p[:nd], p[nd:nd + ni], p[nd + ni:]
            pred_after = model(build_masked(x1, order, tk_ks, False)).argmax(-1).cpu().numpy()
            out[mid] = {
                "deletion_curve": dele, "insertion_curve": ins,
                "deletion_auc": curve_auc(dele), "insertion_auc": curve_auc(ins),
                "topk_conf": tk, "topk_conf_drop": base - tk,
                "topk_pred_changed": (pred_after != int(tgt)).astype(int),
                "baseline_conf": base,
            }
    return out


def stability_for_image(model, x1, tgt, proto, dev, gen):
    """Recompute maps under each transformation and compare on the common grid."""
    res = {}
    with SynchronizedAttributor(model) as att:
        o0 = att.attribute(x1, target_index=torch.tensor([tgt], device=dev))
        c0 = CO.to_common(torch.as_tensor(o0["cnn_maps"][0]))
        v0 = CO.to_common(torch.as_tensor(o0["vit_maps"][0]))
        for t in proto["stability"]["transformations"]:
            tid = t["id"]
            xt = apply_transform(x1, tid, gen)
            ot = att.attribute(xt, target_index=torch.tensor([tgt], device=dev))
            ct = invert_map_transform(CO.to_common(torch.as_tensor(ot["cnn_maps"][0])), tid)
            vt = invert_map_transform(CO.to_common(torch.as_tensor(ot["vit_maps"][0])), tid)
            with torch.no_grad():
                changed = int(model(xt).argmax(-1).item() != tgt)
            res[tid] = {"cnn_spearman": spearman(c0, ct),
                        "vit_spearman": spearman(v0, vt),
                        "prediction_changed": changed}
    return res, spearman(c0, v0)


def bootstrap(rows, proto, key="deletion_auc"):
    """Image-level, class-stratified, paired across methods. 95% percentile CIs."""
    it = proto["bootstrap"]["iterations"]
    rng = np.random.default_rng(proto["bootstrap"]["seed"])
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["true_label"], []).append(r)
    idx = {c: np.arange(len(v)) for c, v in by_cls.items()}
    vals = {m: np.array([[r["%s__%s" % (m, key)] for r in by_cls[c]] for c in by_cls],
                        dtype=object) for m in METHODS}
    out = {}
    means = {m: [] for m in METHODS}
    for _ in range(it):
        picks = {c: rng.integers(0, len(idx[c]), len(idx[c])) for c in by_cls}
        for m in METHODS:
            acc = []
            for ci, c in enumerate(by_cls):
                arr = np.asarray(vals[m][ci], dtype=np.float64)
                acc.append(arr[picks[c]])
            means[m].append(float(np.concatenate(acc).mean()))
    for m in METHODS:
        a = np.array(means[m])
        out[m] = {"mean": float(a.mean()),
                  "ci_low": float(np.percentile(a, 2.5)),
                  "ci_high": float(np.percentile(a, 97.5))}
    return out


def preservation(root):
    man = json.loads((root / BASELINE_MANIFEST).read_text(encoding="utf-8"))
    inv = man["inventory"]
    ent = inv["prediction_csvs"] + inv["metric_files"]
    altered = [e["path"] for e in ent if sha256(root / e["path"]) != e["sha256"]]
    return {"checked": len(ent), "altered": len(altered), "altered_paths": altered}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase X5 faithfulness evaluation")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--verify-protocol-lock", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-preservation", action="store_true")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    root = Path(a.project_root).resolve()

    V = verify_lock(root)
    print("protocol lock verified: %s" % V["protocol_lock_sha256"])
    if V["scale_amendment_sha256"]:
        print("scale amendment:        %s" % V["scale_amendment_sha256"])
    if a.verify_protocol_lock:
        return 0
    if not (a.smoke or a.all):
        ap.error("choose --verify-protocol-lock, --smoke or --all")

    gates = {m: source_gate(HERE / m) for m in ("faithfulness.py", "composite.py")}
    if not all(all(v.values()) for v in gates.values()):
        raise X5Error("eager-attention route reachable")

    if a.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise X5Error("--device cuda requested but CUDA is unavailable")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dev = torch.device(a.device)
    proto = V["protocol"]

    man = json.loads((root / RECON_ROOT / "RECONSTRUCTION_MANIFEST.json")
                     .read_text(encoding="utf-8"))
    entries = sorted(man["files"], key=lambda e: e["filename"])
    if a.smoke:
        by = {c: [] for c in CLASSES}
        for e in entries:
            if len(by[e["manifest_class"]]) < 2:
                by[e["manifest_class"]].append(e)
        entries = [e for v in by.values() for e in v]
        seeds = [42]
    else:
        seeds = SEEDS
    sfx = a.out_suffix or ("_smoke" if a.smoke else "")

    out = root / OUT_ROOT
    targets = {"json": out / ("x5_faithfulness%s.json" % sfx),
               "md": out / ("X5_FAITHFULNESS_REPORT%s.md" % sfx),
               "curves": out / ("per_image_curves%s.csv" % sfx),
               "boot": out / ("bootstrap_effects%s.csv" % sfx),
               "stab": out / ("stability%s.csv" % sfx)}
    clash = [str(p) for p in targets.values() if p.exists()]
    if clash:
        raise X5Error("refusing to overwrite: %s" % clash)

    from PIL import Image
    all_rows, all_stab = [], []
    t_start = time.time()
    for seed in seeds:
        model, _d, ck = load_verified_model(root, seed=seed, device=a.device)
        model.eval()
        if model.vit.config._attn_implementation != "sdpa":
            raise X5Error("attention implementation is not sdpa")
        rnd_gen = torch.Generator().manual_seed(proto["seeds"]["random_baseline"])
        noise_gen = torch.Generator(device=dev).manual_seed(proto["seeds"]["noise_transform"])
        stamp("seed %d: %d images" % (seed, len(entries)))
        for i, e in enumerate(entries, 1):
            x1 = TA.eval_transform(Image.open(root / e["destination_path"])
                                   .convert("RGB")).unsqueeze(0).to(dev)
            with torch.no_grad():
                p = model(x1).softmax(-1)
                _, _, _, gate = model(x1, return_aux=True)
            tgt = int(p.argmax(-1))
            with SynchronizedAttributor(model) as att:
                o = att.attribute(x1, target_index=torch.tensor([tgt], device=dev))
            maps = method_maps(o["cnn_maps"][0], o["vit_maps"][0], gate[0].cpu(), rnd_gen)
            ev = evaluate_image(model, x1, maps, tgt, proto, dev)
            st, agree = stability_for_image(model, x1, tgt, proto, dev, noise_gen)

            row = {"filename": e["filename"], "seed": seed,
                   "true_label": e["manifest_class"], "pred_class": CLASSES[tgt],
                   "correct": int(CLASSES[tgt] == e["manifest_class"]),
                   "baseline_conf": ev[METHODS[0]]["baseline_conf"],
                   "cnn_vit_agreement_spearman": agree}
            for mid in METHODS:
                r = ev[mid]
                row["%s__deletion_auc" % mid] = r["deletion_auc"]
                row["%s__insertion_auc" % mid] = r["insertion_auc"]
                for j, pct in enumerate(proto["top_k_percent"]):
                    row["%s__conf_drop_top%d" % (mid, pct)] = float(r["topk_conf_drop"][j])
                    row["%s__pred_changed_top%d" % (mid, pct)] = int(r["topk_pred_changed"][j])
                row["%s__deletion_curve" % mid] = ";".join("%.6f" % v for v in r["deletion_curve"])
                row["%s__insertion_curve" % mid] = ";".join("%.6f" % v for v in r["insertion_curve"])
            all_rows.append(row)
            for tid, s in st.items():
                all_stab.append({"filename": e["filename"], "seed": seed,
                                 "transformation": tid, **s})
            if i % 50 == 0:
                stamp("  %d/%d  (%.1f min elapsed)" % (i, len(entries),
                                                       (time.time() - t_start) / 60))
        del model
    runtime = time.time() - t_start

    boot = {k: bootstrap(all_rows, proto, k) for k in ("deletion_auc", "insertion_auc")}
    pres = ({"checked": 0, "altered": 0, "skipped": True} if a.skip_preservation
            else {**preservation(root), "skipped": False})

    out.mkdir(parents=True, exist_ok=True)
    with open(targets["curves"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    with open(targets["stab"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(all_stab[0].keys())); w.writeheader(); w.writerows(all_stab)
    with open(targets["boot"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["quantity", "method", "mean", "ci_low", "ci_high"])
        for q, d in boot.items():
            for m, v in d.items():
                w.writerow([q, m, v["mean"], v["ci_low"], v["ci_high"]])

    R = {"phase": "X5", "mode": "smoke" if a.smoke else "full",
         "created_utc": datetime.now(timezone.utc).isoformat(),
         "protocol_lock_sha256": V["protocol_lock_sha256"],
         "scale_amendment_sha256": V["scale_amendment_sha256"],
         "seeds": seeds, "n_images": len(entries), "device": a.device,
         "runtime_seconds": runtime, "methods": METHODS,
         "bootstrap": boot, "preservation": pres,
         "environment": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": __import__("transformers").__version__,
                         "cuda_device": (torch.cuda.get_device_name(0)
                                         if torch.cuda.is_available() else None)},
         "source_gates": gates,
         "scope_exclusions": ["external_or_cross_source (Phase X6)",
                              "anatomical or clinical claims"]}
    targets["json"].write_text(json.dumps(R, indent=2), encoding="utf-8")

    L = ["# Phase X5 - Faithfulness and Stability", "",
         "Protocol lock `%s`." % V["protocol_lock_sha256"], "",
         "Mode `%s`, %d images x %d seeds, device `%s`, %.1f min."
         % (R["mode"], len(entries), len(seeds), a.device, runtime / 60), "",
         "## Deletion AUC (lower indicates the ranking removes evidence faster)", "",
         "| method | mean | 95% CI |", "|---|---|---|"]
    for m in METHODS:
        b = boot["deletion_auc"][m]
        L.append("| `%s` | %.4f | [%.4f, %.4f] |" % (m, b["mean"], b["ci_low"], b["ci_high"]))
    L += ["", "## Insertion AUC (higher indicates the ranking restores evidence faster)", "",
          "| method | mean | 95% CI |", "|---|---|---|"]
    for m in METHODS:
        b = boot["insertion_auc"][m]
        L.append("| `%s` | %.4f | [%.4f, %.4f] |" % (m, b["mean"], b["ci_low"], b["ci_high"]))
    L += ["", "Bootstrap: image-level, stratified by true class, %d iterations, seed %s."
          % (proto["bootstrap"]["iterations"], proto["bootstrap"]["seed"]), "",
          "## Interpretation limits", "",
          "These are measures of how well each ranking predicts this model's own",
          "behaviour under masking. They are not anatomical localisation, not tumour",
          "detection, and not clinical validity. CNN-ViT agreement is descriptive and",
          "is not evidence that either map is correct. Gate mass is branch",
          "utilisation, not causal importance. No attention matrix was read.", ""]
    targets["md"].write_text("\n".join(L), encoding="utf-8")
    (out / ("sha256_manifest%s.json" % sfx)).write_text(
        json.dumps({_rel(p, root): sha256(p) for p in targets.values()}, indent=2),
        encoding="utf-8")

    print("\nX5 %s complete in %.1f min" % (R["mode"], runtime / 60))
    print("  rows: %d, stability rows: %d" % (len(all_rows), len(all_stab)))
    print("  preservation: %s" % pres)
    return 0


if __name__ == "__main__":
    sys.exit(main())
