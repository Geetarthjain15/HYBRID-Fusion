"""Phase X5 - protocol lock and dry-run planning ONLY.

Two strictly ordered jobs, and nothing else:

1. `--write-protocol-lock` writes the complete X5 protocol and records its
   SHA-256. Every parameter that could steer a faithfulness result is fixed
   here, in writing, before any such result exists.

2. `--dry-run` measures wall-clock cost on a fixed 16-image stratified sample
   and projects the full run. It executes the real deletion/insertion loops so
   the timing is realistic, but **every scored value is discarded**: no AUC, no
   curve, no confidence drop, no aggregate of any kind is retained, written or
   printed. Only timings and counts leave this module.

That separation is the point. Choosing the evaluated subset after seeing
faithfulness values would make the subset a function of the answer. The scale
decision is therefore taken from runtime and storage alone.

This module does not compute X5 results. It does not modify FusionV2, any
checkpoint, dataset, prediction, or any X3/X4 artifact. SDPA is never changed
and attention matrices are never read.

Usage
-----
    env_win/Scripts/python.exe src/publication/explainability/x5_protocol_dryrun.py --project-root . --write-protocol-lock
    env_win/Scripts/python.exe src/publication/explainability/x5_protocol_dryrun.py --project-root . --dry-run
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import random
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
import cnn_attribution as CA                                  # noqa: E402
import vit_attribution as VA                                  # noqa: E402
from run_attribution import SynchronizedAttributor            # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEEDS = [42, 43, 44]
N_IMAGES_FULL = 848
DRYRUN_PER_CLASS = 4          # 16-image stratified sample
DRYRUN_SAMPLE_SEED = 42

RECON_ROOT = "publication_runs/explainability/reconstructed_clean_test"
OUT_ROOT = "publication_results/explainability/x5_protocol"

# --------------------------------------------------------- THE LOCKED PROTOCOL
# Every value below is fixed before any faithfulness number exists. Changing any
# of them after results are seen invalidates the lock, and the lock hash is what
# makes that detectable.
PROTOCOL = {
    "phase": "X5",
    "protocol_version": "1.0",
    "purpose": ("Quantitative faithfulness and stability evaluation of the X3 "
                "attribution maps. Locked before any faithfulness value is "
                "computed."),

    "methods": [
        {"id": "cnn_gradcampp", "native_grid": [7, 7],
         "source": "X3 CNN Grad-CAM++ at native stage-8 resolution"},
        {"id": "vit_grad_x_input", "native_grid": [14, 14],
         "source": "X3 ViT patch-embedding Gradient x Input (the X2b method)"},
        {"id": "composite_equal", "native_grid": [14, 14],
         "source": "0.5 * norm(cnn) + 0.5 * norm(vit), both on the common grid",
         "note": "formed in X5 only; X3 deliberately formed no composite"},
        {"id": "composite_gate_weighted", "native_grid": [14, 14],
         "source": "gate[:,0] * norm(cnn) + gate[:,1] * norm(vit)",
         "gate_source": "model(x, return_aux=True) gate from the same forward "
                        "pass that produced the maps and the target class",
         "note": "gate mass measures branch utilisation, not causal importance"},
        {"id": "baseline_random", "native_grid": [14, 14],
         "source": "deterministic uniform random map, null baseline"},
        {"id": "baseline_centre_prior", "native_grid": [14, 14],
         "source": "deterministic isotropic Gaussian centred on the grid, "
                   "sigma = grid/4, null baseline"},
    ],
    "method_set_divergence_from_adopted_plan": (
        "The adopted plan's X5 section lists ViT attention rollout and "
        "gradient-weighted ViT attention. Both were removed by the approved X3 "
        "amendment because they require eager attention, which failed the X2 "
        "neutrality gate. The ViT method evaluated here is the SDPA-safe "
        "Gradient x Input method verified in X2b and X3."),

    "common_grid": [14, 14],
    "resampling": {
        "cnn_7_to_14": "bilinear, align_corners=False",
        "implementation": "torch.nn.functional.interpolate",
        "order": "resample FIRST, normalise SECOND",
    },
    "normalisation": {
        "rule": "per-map min-max to [0,1] after resampling",
        "constant_map_policy": "a map with max == min becomes all zeros and is "
                               "recorded as degenerate; X4 established no real "
                               "map is constant",
    },

    "masking": {
        "mask_value_normalised_tensor": 0.0,
        "mask_value_raw_pixel": 0.5,
        "explanation": ("eval_transform normalises with mean 0.5 and std 0.5, so "
                        "a raw mid-grey pixel of 0.5 maps to exactly 0.0 in "
                        "tensor space; masking therefore replaces a region with "
                        "the dataset channel mean"),
        "mask_upsample_14_to_224": "nearest-neighbour, exact 16x16 pixel blocks",
        "rationale": ("nearest keeps each grid cell a hard-edged block so the "
                      "masked fraction is exact; bilinear would blur the "
                      "boundary and make the fraction approximate"),
    },
    "deletion": {
        "start": "unmodified image",
        "order": "descending map value; ties broken by ascending flat grid index",
        "steps": 20,
        "fractions": "equal steps of 5% from 0% to 100% of the 196 grid cells",
        "score": "fused softmax probability of the locked target class",
    },
    "insertion": {
        "start": "fully masked image (all cells at the mask value)",
        "order": "descending map value; ties broken by ascending flat grid index",
        "steps": 20,
        "fractions": "equal steps of 5% from 0% to 100% of the 196 grid cells",
        "score": "fused softmax probability of the locked target class",
    },
    "auc": {"rule": "trapezoidal over the 21 scored points including both endpoints",
            "normalisation": "x axis scaled to [0,1]"},
    "top_k_percent": [1, 5, 10, 20, 50],
    "top_k_metrics": ["confidence_drop", "prediction_change_rate"],

    "target_class": {
        "rule": "predicted class from the recorded inference",
        "identical_to": "X3 target_mode 'predicted (recorded inference)'",
        "note": "the same target index is used for every method on a given "
                "image, so methods are compared on one decision",
    },

    "stability": {
        "transformations": [
            {"id": "hflip", "spec": "horizontal flip"},
            {"id": "rot_plus10", "spec": "+10 degrees, bilinear, zero fill"},
            {"id": "rot_minus10", "spec": "-10 degrees, bilinear, zero fill"},
            {"id": "scale_plus5", "spec": "+5% scale about centre, bilinear"},
            {"id": "scale_minus5", "spec": "-5% scale about centre, bilinear"},
            {"id": "noise", "spec": "additive Gaussian, sigma=0.01 in normalised "
                                    "tensor space, fixed seed 20260927"},
        ],
        "map_comparison": "Spearman rank correlation on the common grid, after "
                          "inverting the geometric transform on the map",
        "label_preserving": True,
        "prediction_change_recorded": True,
    },

    "agreement": {
        "metric": "Spearman rank correlation on the common grid",
        "scope": "descriptive; CNN-ViT agreement is NOT evidence that either map "
                 "is correct",
    },

    "subset_decision": {
        "status": "OPEN at lock time - to be fixed by the dry run",
        "decision_rule": ("taken from measured runtime and storage ALONE. "
                          "Faithfulness values are not computed in the dry run, "
                          "so the subset cannot be a function of the answer."),
        "if_subset_required": {
            "rule": "stratified by true class, proportional allocation",
            "sampling_seed": 20260927,
            "selection_time": "before any faithfulness value is computed",
            "recorded": "every selected filename written into the amendment",
        },
        "amendment_procedure": ("the decision is appended as a separate, "
                                "separately hashed amendment; the original lock "
                                "hash is never rewritten"),
    },

    "bootstrap": {
        "unit": "image",
        "stratification": "by true class",
        "iterations": 10000,
        "seed": 20260927,
        "quantities": "every headline quantity, reported with 95% percentile CIs",
        "paired": "methods compared on the same resampled image sets",
    },

    "seeds": {"model_seeds": SEEDS, "random_baseline": 20260927,
              "noise_transform": 20260927, "bootstrap": 20260927,
              "subset_sampling_if_needed": 20260927},

    "determinism": {"torch_deterministic": True, "cudnn_benchmark": False,
                    "batch_size_recorded": True,
                    "note": "batch size changes float32 reduction order; the "
                            "value actually used is recorded with the results"},

    "prohibited_in_x5": [
        "any anatomical, tumour, lesion or clinical claim",
        "any attention-based claim (no attention matrix is read)",
        "treating CNN-ViT agreement as evidence of correctness",
        "treating gate mass as causal importance",
        "altering any locked parameter after observing a result",
        "selecting the evaluated subset using faithfulness values",
        "external or cross-source evaluation (that is Phase X6)",
    ],
}


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


# ----------------------------------------------------------- protocol lock ---

def render_lock(P: dict) -> str:
    L = ["# Phase X5 Protocol Lock", "",
         "**Written before any faithfulness value was computed.**", "",
         "Version `%s`. Every parameter below is fixed. Changing any of them "
         "after a result is observed invalidates this lock, and the SHA-256 "
         "recorded alongside it is what makes that detectable." % P["protocol_version"],
         "", "## Methods", "", "| id | native grid | source |", "|---|---|---|"]
    for m in P["methods"]:
        L.append("| `%s` | %dx%d | %s |" % (m["id"], m["native_grid"][0],
                                            m["native_grid"][1], m["source"]))
    L += ["", "> %s" % P["method_set_divergence_from_adopted_plan"], "",
          "## Grid, resampling, normalisation", "", "| parameter | value |", "|---|---|",
          "| common grid | %dx%d |" % tuple(P["common_grid"]),
          "| CNN 7 -> 14 | %s |" % P["resampling"]["cnn_7_to_14"],
          "| order | %s |" % P["resampling"]["order"],
          "| normalisation | %s |" % P["normalisation"]["rule"], "",
          "## Masking", "", "| parameter | value |", "|---|---|",
          "| mask value (tensor space) | %s |" % P["masking"]["mask_value_normalised_tensor"],
          "| mask value (raw pixel) | %s |" % P["masking"]["mask_value_raw_pixel"],
          "| mask upsample 14 -> 224 | %s |" % P["masking"]["mask_upsample_14_to_224"],
          "", "%s" % P["masking"]["explanation"].capitalize() + ".", "",
          "## Deletion and insertion", "", "| parameter | deletion | insertion |",
          "|---|---|---|",
          "| start | %s | %s |" % (P["deletion"]["start"], P["insertion"]["start"]),
          "| order | %s | %s |" % (P["deletion"]["order"], P["insertion"]["order"]),
          "| steps | %d | %d |" % (P["deletion"]["steps"], P["insertion"]["steps"]),
          "| score | %s | %s |" % (P["deletion"]["score"], P["insertion"]["score"]),
          "", "AUC: %s, %s." % (P["auc"]["rule"], P["auc"]["normalisation"]),
          "", "Top-k percentages: %s. Metrics: %s."
          % (P["top_k_percent"], ", ".join(P["top_k_metrics"])), "",
          "## Stability", "", "| transformation | specification |", "|---|---|"]
    for t in P["stability"]["transformations"]:
        L.append("| `%s` | %s |" % (t["id"], t["spec"]))
    L += ["", "Map comparison: %s." % P["stability"]["map_comparison"], "",
          "## Bootstrap", "", "| parameter | value |", "|---|---|"]
    for k in ("unit", "stratification", "iterations", "seed", "quantities", "paired"):
        L.append("| %s | %s |" % (k, P["bootstrap"][k]))
    L += ["", "## Subset decision", "",
          "**Status at lock time: %s**" % P["subset_decision"]["status"], "",
          P["subset_decision"]["decision_rule"], "",
          "If a subset is required: %s, sampling seed %s, selected %s."
          % (P["subset_decision"]["if_subset_required"]["rule"],
             P["subset_decision"]["if_subset_required"]["sampling_seed"],
             P["subset_decision"]["if_subset_required"]["selection_time"]), "",
          P["subset_decision"]["amendment_procedure"].capitalize() + ".", "",
          "## Prohibited in X5", ""]
    for p in P["prohibited_in_x5"]:
        L.append("- %s" % p)
    L += ["", "## Seeds", "", "| purpose | seed |", "|---|---|"]
    for k, v in P["seeds"].items():
        L.append("| %s | %s |" % (k, v))
    L.append("")
    return "\n".join(L)


# ----------------------------------------------------------------- dry run ---

def stratified_sample(manifest, per_class, seed):
    by = {c: [] for c in CLASSES}
    for e in sorted(manifest["files"], key=lambda e: e["filename"]):
        by[e["manifest_class"]].append(e)
    rng = random.Random(seed)
    out = []
    for c in CLASSES:
        out.extend(sorted(rng.sample(by[c], per_class), key=lambda e: e["filename"]))
    return out


def to_common(m: np.ndarray) -> torch.Tensor:
    """Resample to 14x14 (bilinear, align_corners=False) then min-max to [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(m)).float()[None, None]
    if t.shape[-2:] != (14, 14):
        t = F.interpolate(t, size=(14, 14), mode="bilinear", align_corners=False)
    t = t[0, 0]
    lo, hi = float(t.min()), float(t.max())
    return (t - lo) / (hi - lo) if hi > lo else torch.zeros_like(t)


def masked_batch(x: torch.Tensor, order: torch.Tensor, k: int, start_masked: bool):
    """Build one 224x224 image with k grid cells revealed/masked (nearest blocks)."""
    grid = torch.zeros(196, dtype=torch.bool)
    if k > 0:
        grid[order[:k]] = True
    g = grid.reshape(1, 1, 14, 14).float()
    up = F.interpolate(g, size=(224, 224), mode="nearest")
    if start_masked:                      # insertion: revealed cells keep image
        return x * up + 0.0 * (1 - up)
    return x * (1 - up) + 0.0 * up        # deletion: masked cells -> 0.0


def dry_run(root: Path, protocol: dict) -> dict:
    model, dev, ck = load_verified_model(root, seed=42, device="cpu")
    model.eval()
    if model.vit.config._attn_implementation != "sdpa":
        raise X5Error("attention implementation is not sdpa")

    man = json.loads((root / RECON_ROOT / "RECONSTRUCTION_MANIFEST.json")
                     .read_text(encoding="utf-8"))
    entries = stratified_sample(man, DRYRUN_PER_CLASS, DRYRUN_SAMPLE_SEED)
    from PIL import Image
    xs = torch.stack([TA.eval_transform(
        Image.open(root / e["destination_path"]).convert("RGB")) for e in entries])
    n = len(entries)
    stamp("dry-run sample: %d images (%d per class, seed %d)"
          % (n, DRYRUN_PER_CLASS, DRYRUN_SAMPLE_SEED))

    steps = protocol["deletion"]["steps"]
    topk = protocol["top_k_percent"]
    rng = np.random.default_rng(protocol["seeds"]["random_baseline"])

    # ---- map production (timed) -------------------------------------------
    t0 = time.time()
    with torch.no_grad():
        probs = model(xs).softmax(-1)
        _, _, _, gate = model(xs, return_aux=True)
    tgt = probs.argmax(-1)
    with SynchronizedAttributor(model) as att:
        o = att.attribute(xs, target_index=tgt)
    t_maps = time.time() - t0

    yy, xx = np.mgrid[0:14, 0:14].astype(np.float32)
    cp = np.exp(-(((yy - 6.5) ** 2 + (xx - 6.5) ** 2) / (2 * (14 / 4.0) ** 2)))

    timings = []
    per_method_total = {}
    for k in range(n):
        c14 = to_common(o["cnn_maps"][k])
        v14 = to_common(o["vit_maps"][k])
        g0, g1 = float(gate[k, 0]), float(gate[k, 1])
        maps = {
            "cnn_gradcampp": c14,
            "vit_grad_x_input": v14,
            "composite_equal": to_common((0.5 * c14 + 0.5 * v14).numpy()),
            "composite_gate_weighted": to_common((g0 * c14 + g1 * v14).numpy()),
            "baseline_random": to_common(rng.random((14, 14), dtype=np.float32)),
            "baseline_centre_prior": to_common(cp.astype(np.float32)),
        }
        x1 = xs[k:k + 1]
        for mid, m in maps.items():
            t1 = time.time()
            order = torch.argsort(m.reshape(-1), descending=True, stable=True)
            n_fwd = 0
            with torch.no_grad():
                for s in range(steps + 1):
                    kk = int(round(196 * s / steps))
                    model(masked_batch(x1, order, kk, False)).softmax(-1)
                    n_fwd += 1
                for s in range(steps + 1):
                    kk = int(round(196 * s / steps))
                    model(masked_batch(x1, order, kk, True)).softmax(-1)
                    n_fwd += 1
                for pct in topk:
                    kk = int(round(196 * pct / 100.0))
                    model(masked_batch(x1, order, kk, False)).softmax(-1)
                    n_fwd += 1
            # Every scored value above is DISCARDED on purpose. Only the cost of
            # producing it is retained.
            dt = time.time() - t1
            timings.append({"filename": entries[k]["filename"], "method": mid,
                            "seconds": dt, "forward_passes": n_fwd})
            per_method_total[mid] = per_method_total.get(mid, 0.0) + dt

    return {
        "n_images": n, "checkpoint_sha256": ck,
        "map_production_seconds_for_batch": t_maps,
        "map_production_s_per_image": t_maps / n,
        "per_method_s_per_image": {k: v / n for k, v in per_method_total.items()},
        "forward_passes_per_image_per_method": timings[0]["forward_passes"],
        "sample": [e["filename"] for e in entries],
        "_timings": timings,
    }


def project(dr: dict, protocol: dict) -> dict:
    n_methods = len(protocol["methods"])
    per_img_all_methods = sum(dr["per_method_s_per_image"].values())
    per_img = per_img_all_methods + dr["map_production_s_per_image"]
    n_transforms = len(protocol["stability"]["transformations"])
    # stability re-derives maps and re-scores under each transformation
    stability_per_img = (dr["map_production_s_per_image"] * n_transforms)
    total_per_img = per_img + stability_per_img

    full_images = N_IMAGES_FULL * len(SEEDS)
    cpu_hours = total_per_img * full_images / 3600.0
    # GPU projection uses the ratio MEASURED in X3 on this exact code path:
    # CPU 0.404 s/img vs Tesla T4 0.023 s/img.
    gpu_ratio = 0.404 / 0.023
    gpu_hours = cpu_hours / gpu_ratio

    fwd_total = (dr["forward_passes_per_image_per_method"] * n_methods * full_images)
    curve_pts = (protocol["deletion"]["steps"] + 1) * 2 + len(protocol["top_k_percent"])
    storage_mb = (full_images * n_methods * curve_pts * 8 * 3) / 1e6

    return {
        "n_methods": n_methods,
        "seconds_per_image_all_methods": per_img_all_methods,
        "seconds_per_image_map_production": dr["map_production_s_per_image"],
        "seconds_per_image_stability": stability_per_img,
        "seconds_per_image_total": total_per_img,
        "full_run_images": full_images,
        "forward_passes_total": int(fwd_total),
        "cpu_hours": cpu_hours,
        "gpu_hours_projected": gpu_hours,
        "gpu_ratio_source": ("measured in X3 on this code path: CPU 0.404 s/img "
                             "vs Tesla T4 0.023 s/img = %.1fx" % gpu_ratio),
        "storage_mb_curves": storage_mb,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase X5 protocol lock / dry run")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--write-protocol-lock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (a.write_protocol_lock or a.dry_run):
        ap.error("choose --write-protocol-lock or --dry-run")
    root = Path(a.project_root).resolve()
    out = root / OUT_ROOT
    lock_md, lock_json = out / "X5_PROTOCOL_LOCK.md", out / "x5_protocol_lock.json"

    if a.write_protocol_lock:
        if lock_md.exists() or lock_json.exists():
            raise X5Error("protocol lock already exists; it is never rewritten")
        out.mkdir(parents=True, exist_ok=True)
        payload = dict(PROTOCOL)
        payload["locked_utc"] = datetime.now(timezone.utc).isoformat()
        payload["environment"] = {"python": platform.python_version(),
                                  "torch": torch.__version__,
                                  "transformers": __import__("transformers").__version__}
        payload["inputs"] = {
            "x3_attribution_json_sha256":
                sha256(root / "publication_results/explainability/x3_attribution/x3_attribution.json"),
            "x4_sanity_json_sha256":
                sha256(root / "publication_results/explainability/x4_sanity/x4_sanity.json"),
            "x3_amendment_sha256":
                sha256(root / "publication_results/explainability/X3_BRANCH_ATTRIBUTION_AMENDMENT_PROPOSAL.md"),
            "reconstruction_manifest_sha256":
                sha256(root / RECON_ROOT / "RECONSTRUCTION_MANIFEST.json"),
        }
        lock_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        lock_md.write_text(render_lock(payload), encoding="utf-8")
        h_json, h_md = sha256(lock_json), sha256(lock_md)
        (out / "PROTOCOL_LOCK_SHA256.txt").write_text(
            "x5_protocol_lock.json  %s\nX5_PROTOCOL_LOCK.md    %s\n" % (h_json, h_md),
            encoding="utf-8")
        print("PROTOCOL LOCKED")
        print("  x5_protocol_lock.json  %s" % h_json)
        print("  X5_PROTOCOL_LOCK.md    %s" % h_md)
        return 0

    # ---- dry run: requires the lock to already exist -----------------------
    if not lock_json.is_file():
        raise X5Error("no protocol lock; write it before any timing run")
    protocol = json.loads(lock_json.read_text(encoding="utf-8"))
    lock_hash = sha256(lock_json)

    targets = {"json": out / "x5_dryrun.json", "md": out / "X5_DRYRUN_REPORT.md",
               "csv": out / "dryrun_timings.csv"}
    clash = [str(p) for p in targets.values() if p.exists()]
    if clash:
        raise X5Error("refusing to overwrite: %s" % clash)

    gates = {m: source_gate(HERE / m) for m in
             ("x5_protocol_dryrun.py", "cnn_attribution.py", "vit_attribution.py")}
    if not all(all(v.values()) for v in gates.values()):
        raise X5Error("eager-attention route reachable")
    stamp("source gates ok; protocol lock %s" % lock_hash[:16])

    dr = dry_run(root, protocol)
    pr = project(dr, protocol)

    # scale decision, from runtime and storage ONLY
    if pr["cpu_hours"] <= 2.0:
        rec, why = "CPU (local)", "projected CPU cost is small"
    elif pr["gpu_hours_projected"] <= 3.0:
        rec, why = ("Kaggle GPU, full 848 x 3",
                    "infeasible on this CPU-only host but comfortable on a T4; "
                    "no subset needed, so no selection decision arises")
    else:
        rec, why = ("Kaggle GPU with a prespecified stratified subset",
                    "full set exceeds a single Kaggle session even on GPU")

    R = {
        "phase": "X5", "stage": "protocol lock + dry run only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_lock_sha256": lock_hash,
        "protocol_lock_md_sha256": sha256(lock_md),
        "faithfulness_values_computed": False,
        "faithfulness_values_retained": False,
        "note": ("the dry run executed the real deletion/insertion loops for "
                 "realistic timing and discarded every scored value; no AUC, "
                 "curve or aggregate exists at this stage"),
        "dry_run": {k: v for k, v in dr.items() if not k.startswith("_")},
        "projection": pr,
        "scale_recommendation": {"recommendation": rec, "reason": why,
                                 "decided_from": "runtime and storage only",
                                 "status": "PROPOSED - requires approval before "
                                           "the full X5 run"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": __import__("transformers").__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "device_used": "cpu"},
        "source_gates": gates,
    }

    targets["json"].write_text(json.dumps(R, indent=2), encoding="utf-8")
    with open(targets["csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(dr["_timings"][0].keys()))
        w.writeheader(); w.writerows(dr["_timings"])

    L = ["# Phase X5 Dry Run", "",
         "**No faithfulness value was computed or retained.** The deletion and",
         "insertion loops were executed for realistic timing and every scored",
         "value was discarded, so the scale decision below cannot be a function",
         "of the answer.", "",
         "Protocol lock SHA-256: `%s`" % lock_hash, "",
         "## Measured on %d images (%d per class, seed %d), CPU"
         % (dr["n_images"], DRYRUN_PER_CLASS, DRYRUN_SAMPLE_SEED), "",
         "| quantity | value |", "|---|---|",
         "| map production | %.3f s/image |" % dr["map_production_s_per_image"],
         "| forward passes per image per method | %d |" % dr["forward_passes_per_image_per_method"],
         "", "| method | s/image |", "|---|---|"]
    for k, v in sorted(dr["per_method_s_per_image"].items(), key=lambda kv: -kv[1]):
        L.append("| `%s` | %.3f |" % (k, v))
    L += ["", "## Projected full run (848 images x 3 seeds x %d methods)" % pr["n_methods"],
          "", "| quantity | value |", "|---|---|",
          "| seconds per image, all methods | %.2f |" % pr["seconds_per_image_all_methods"],
          "| seconds per image, stability | %.2f |" % pr["seconds_per_image_stability"],
          "| seconds per image, total | %.2f |" % pr["seconds_per_image_total"],
          "| total forward passes | %s |" % "{:,}".format(pr["forward_passes_total"]),
          "| **CPU** | **%.1f hours** |" % pr["cpu_hours"],
          "| **Kaggle T4 (projected)** | **%.1f hours** |" % pr["gpu_hours_projected"],
          "| curve storage | %.1f MB |" % pr["storage_mb_curves"],
          "", "GPU projection basis: %s." % pr["gpu_ratio_source"], "",
          "## Scale recommendation", "",
          "**%s**" % rec, "", "%s." % why.capitalize(), "",
          "Decided from runtime and storage only. Requires approval before the",
          "full X5 run begins.", ""]
    targets["md"].write_text("\n".join(L), encoding="utf-8")
    mani = {_rel(p, root): sha256(p) for p in
            list(targets.values()) + [lock_md, lock_json]}
    (out / "sha256_manifest.json").write_text(json.dumps(mani, indent=2), encoding="utf-8")

    print("\nDRY RUN COMPLETE - no faithfulness value computed")
    print("  map production        %.3f s/image" % dr["map_production_s_per_image"])
    for k, v in sorted(dr["per_method_s_per_image"].items(), key=lambda kv: -kv[1]):
        print("    %-26s %.3f s/image" % (k, v))
    print("  total per image       %.2f s" % pr["seconds_per_image_total"])
    print("  forward passes        %s" % "{:,}".format(pr["forward_passes_total"]))
    print("  CPU                   %.1f h" % pr["cpu_hours"])
    print("  Kaggle T4 (projected) %.1f h" % pr["gpu_hours_projected"])
    print("  storage               %.1f MB" % pr["storage_mb_curves"])
    print("  recommendation        %s" % rec)
    print("  protocol lock         %s" % lock_hash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
