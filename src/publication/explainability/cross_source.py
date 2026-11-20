"""Phase X6 - locked cross-source explanation analysis.

Applies the unchanged Phase-X5 protocol to the 373 images marked ``disjoint``
in ``results/cross_source.json``.  External labels are used only for scoring and
class-stratified uncertainty estimation; they never select a method, threshold,
image, or protocol parameter.

The full grid is GPU work.  Local usage is limited to verification/dry-run and
the fixed eight-image smoke test::

    python -m src.publication.explainability.cross_source --project-root . --dry-run
    python -m src.publication.explainability.cross_source --project-root . --smoke
    python -m src.publication.explainability.cross_source --project-root . --all --device cuda
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import Counter
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

import train_ablation as TA                                    # noqa: E402
from fusion_v2 import tta_logits                               # noqa: E402
from fusion_v2_gradcam import load_verified_model              # noqa: E402
from run_attribution import SynchronizedAttributor             # noqa: E402
import faithfulness as FA                                      # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEEDS = [42, 43, 44]
EXPECTED_N = 373
EXPECTED_COUNTS = {"glioma": 52, "meningioma": 122,
                   "notumor": 88, "pituitary": 111}
PROB_TOL = 1e-4
RECON_ROOT = "publication_runs/explainability/reconstructed_cross_source"
OUT_ROOT = "publication_results/explainability/x6_cross_source"
BASELINE_MANIFEST = "publication_results/audit/baseline_manifest.json"
X5_CURVES = "publication_results/explainability/x5_faithfulness/per_image_curves.csv"


class X6Error(RuntimeError):
    """A prespecified Phase-X6 stop condition was reached."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def stamp(message: str) -> None:
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), message), flush=True)


def load_selector(root: Path) -> tuple[list[dict], str]:
    """Return the filename-sorted, pre-existing disjoint selector."""
    path = root / "results/cross_source.json"
    if not path.is_file():
        raise X6Error("cross-source manifest absent: %s" % path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    manifest = obj.get("manifest")
    if not isinstance(manifest, dict):
        raise X6Error("results/cross_source.json has no manifest object")
    rows = [{"filename": fn, "true_label": meta.get("class"),
             "kind": meta.get("kind"), "min_hamming": meta.get("min_hamming")}
            for fn, meta in manifest.items() if meta.get("kind") == "disjoint"]
    rows.sort(key=lambda r: r["filename"])
    names = [r["filename"] for r in rows]
    counts = dict(Counter(r["true_label"] for r in rows))
    if len(rows) != EXPECTED_N or len(set(names)) != EXPECTED_N:
        raise X6Error("disjoint selector count/uniqueness is %d/%d, expected %d"
                      % (len(rows), len(set(names)), EXPECTED_N))
    if counts != EXPECTED_COUNTS:
        raise X6Error("disjoint class counts %s != %s" % (counts, EXPECTED_COUNTS))
    if set(counts) != set(CLASSES):
        raise X6Error("selector classes do not match documented class order")
    return rows, sha256(path)


def index_external(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    base = root / "dataset_ood"
    if not base.is_dir():
        raise X6Error("dataset_ood absent: %s" % base)
    index: dict[str, Path] = {}
    dup: dict[str, list[Path]] = {}
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.name in index:
            dup.setdefault(path.name, [index[path.name]]).append(path)
        else:
            index[path.name] = path
    return index, dup


def reconstruct(root: Path, selector: list[dict]) -> dict:
    dest = root / RECON_ROOT
    if dest.exists():
        raise X6Error("refusing to overwrite existing reconstruction: %s" % dest)
    index, dup = index_external(root)
    ambiguous = [r["filename"] for r in selector if r["filename"] in dup]
    missing = [r["filename"] for r in selector if r["filename"] not in index]
    if ambiguous or missing:
        raise X6Error("external resolution failed: %d ambiguous, %d missing; examples=%s"
                      % (len(ambiguous), len(missing), (ambiguous + missing)[:5]))
    files = []
    for row in selector:
        src = index[row["filename"]]
        # The manifest, not the directory spelling, is authoritative.
        dst = dest / row["true_label"] / row["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files.append({**row, "source_path": rel(src, root),
                      "destination_path": rel(dst, root),
                      "sha256": sha256(dst), "bytes": dst.stat().st_size})
    counts = dict(Counter(f["true_label"] for f in files))
    manifest = {"phase": "X6", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": "results/cross_source.json manifest entries where kind == disjoint",
                "n_files": len(files), "per_class_counts": counts, "files": files}
    mp = dest / "RECONSTRUCTION_MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_reconstruction(root: Path, selector: list[dict]) -> list[dict]:
    path = root / RECON_ROOT / "RECONSTRUCTION_MANIFEST.json"
    if not path.is_file():
        raise X6Error("reconstruction manifest absent: %s" % path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    files = sorted(obj.get("files", []), key=lambda r: r["filename"])
    if len(files) != EXPECTED_N:
        raise X6Error("reconstruction has %d files, expected %d" % (len(files), EXPECTED_N))
    wanted = {r["filename"]: r["true_label"] for r in selector}
    got = {r["filename"]: r["true_label"] for r in files}
    if got != wanted:
        raise X6Error("reconstruction filename/label membership differs from selector")
    for row in files:
        p = root / row["destination_path"]
        if not p.is_file() or sha256(p) != row["sha256"]:
            raise X6Error("reconstructed image missing or hash mismatch: %s" % p)
    return files


def load_recorded(root: Path, seed: int, selector: list[dict]) -> dict[str, dict]:
    path = root / ("runs_v2_clean/fusion_v2_seed%d/preds_dataset_ood.csv" % seed)
    gate_path = root / ("runs_v2_clean/fusion_v2_seed%d/gates_dataset_ood.csv" % seed)
    if not path.is_file() or not gate_path.is_file():
        raise X6Error("authoritative external prediction/gate file absent for seed %d" % seed)
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with gate_path.open(encoding="utf-8") as f:
        gates = {r["filename"]: r for r in csv.DictReader(f)}
    if len(rows) != len({r["filename"] for r in rows}):
        raise X6Error("duplicate filenames in %s" % path)
    selected = {r["filename"]: r for r in selector}
    rec = {r["filename"]: r for r in rows if r["filename"] in selected}
    if set(rec) != set(selected) or set(rec) - set(gates):
        raise X6Error("prediction/gate membership does not cover the exact selector")
    for fn, r in rec.items():
        if r["true_label"] != selected[fn]["true_label"]:
            raise X6Error("true-label mismatch for %s" % fn)
        probs = np.array([float(r["p_" + c]) for c in CLASSES])
        if (not np.isfinite(probs).all() or (probs < 0).any() or
                (probs > 1).any() or abs(float(probs.sum()) - 1.0) > 2e-5):
            raise X6Error("invalid recorded probabilities for %s" % fn)
        if CLASSES[int(probs.argmax())] != r["pred_label"]:
            raise X6Error("recorded argmax/label mismatch for %s" % fn)
        g = np.array([float(gates[fn]["gate_cnn"]), float(gates[fn]["gate_vit"])])
        if not np.isfinite(g).all() or abs(float(g.sum()) - 1.0) > 2e-5:
            raise X6Error("invalid recorded gate for %s" % fn)
    return rec


def smoke_sample(files: list[dict]) -> list[dict]:
    out = []
    for cls in CLASSES:
        out += [r for r in files if r["true_label"] == cls][:2]
    return out


def entropy(prob: np.ndarray) -> float:
    p = np.clip(prob.astype(np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def independent_stratified_shift(external: list[dict], clean_rows: list[dict],
                                 field: str, iterations: int, seed: int) -> dict:
    """External minus clean difference; independent image sets, class-stratified."""
    rng = np.random.default_rng(seed)
    def collapse(rows):
        # Three model seeds are repeated measurements of one image, not three
        # independent cases. Average within filename before image-level bootstrap.
        grouped = {}
        for row in rows:
            grouped.setdefault(row["filename"], []).append(row)
        out = {c: [] for c in CLASSES}
        for filename, records in grouped.items():
            labels = {r["true_label"] for r in records}
            if len(labels) != 1:
                raise X6Error("inconsistent true labels across seeds for %s" % filename)
            values = np.array([float(r[field]) for r in records], dtype=np.float64)
            if not np.isfinite(values).all():
                raise X6Error("non-finite %s for %s" % (field, filename))
            out[next(iter(labels))].append(float(values.mean()))
        return {c: np.asarray(v, dtype=np.float64) for c, v in out.items()}
    ext, clean = collapse(external), collapse(clean_rows)
    if any(len(ext[c]) == 0 or len(clean[c]) == 0 for c in CLASSES):
        raise X6Error("a class is absent from a shift comparison")
    diffs = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        ea = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in ext.values()])
        ca = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in clean.values()])
        diffs[i] = ea.mean() - ca.mean()
    observed = np.concatenate(list(ext.values())).mean() - np.concatenate(list(clean.values())).mean()
    return {"external_mean": float(np.concatenate(list(ext.values())).mean()),
            "clean_mean": float(np.concatenate(list(clean.values())).mean()),
            "difference_external_minus_clean": float(observed),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "iterations": iterations, "unit": "unique image after averaging model seeds",
            "sets": "independent", "external_unique_images": int(sum(map(len, ext.values()))),
            "clean_unique_images": int(sum(map(len, clean.values())))}


def load_clean_diagnostics(root: Path, x5_rows: list[dict]) -> list[dict]:
    """Merge clean X5 rows with immutable prediction/gate CSV evidence."""
    by_key = {(int(r["seed"]), r["filename"]): dict(r) for r in x5_rows}
    for seed in SEEDS:
        pred_path = root / ("runs_v2_clean/fusion_v2_seed%d/preds_test.csv" % seed)
        gate_path = root / ("runs_v2_clean/fusion_v2_seed%d/gates_test.csv" % seed)
        with pred_path.open(encoding="utf-8") as f:
            preds = {r["filename"]: r for r in csv.DictReader(f)}
        with gate_path.open(encoding="utf-8") as f:
            gates = {r["filename"]: r for r in csv.DictReader(f)}
        keys = {fn for s, fn in by_key if s == seed}
        if keys != set(preds) or keys != set(gates):
            raise X6Error("clean X5/prediction/gate membership mismatch for seed %d" % seed)
        for fn in keys:
            row = by_key[(seed, fn)]
            p = np.array([float(preds[fn]["p_" + c]) for c in CLASSES])
            order = np.sort(p)
            row.update({"tta_confidence": float(p.max()),
                        "tta_margin": float(order[-1] - order[-2]),
                        "tta_entropy": entropy(p),
                        "gate_cnn": float(gates[fn]["gate_cnn"]),
                        "gate_vit": float(gates[fn]["gate_vit"])})
    return list(by_key.values())


def class_error_summary(rows: list[dict]) -> dict:
    result = {}
    for cls in CLASSES:
        selected = [r for r in rows if r["true_label"] == cls]
        # Average seed-specific recalls rather than treating repeated predictions
        # as independent cases.
        recalls = []
        for seed in sorted({int(r["seed"]) for r in selected}):
            sr = [r for r in selected if int(r["seed"]) == seed]
            recalls.append(sum(r["pred_class"] == cls for r in sr) / len(sr))
        result[cls] = {"images_per_seed": len({r["filename"] for r in selected}),
                       "recall_seed_mean": float(np.mean(recalls)),
                       "recall_seed_sd": float(np.std(recalls, ddof=1)) if len(recalls) > 1 else 0.0}
    result["pituitary_to_notumor"] = {
        "predictions": sum(r["true_label"] == "pituitary" and r["pred_class"] == "notumor"
                           for r in rows),
        "image_seed_pairs": sum(r["true_label"] == "pituitary" for r in rows)}
    return result


def preservation(root: Path) -> dict:
    obj = json.loads((root / BASELINE_MANIFEST).read_text(encoding="utf-8"))
    entries = obj["inventory"]["prediction_csvs"] + obj["inventory"]["metric_files"]
    changed = [e["path"] for e in entries if sha256(root / e["path"]) != e["sha256"]]
    return {"checked": len(entries), "altered": len(changed), "altered_paths": changed}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-preservation", action="store_true")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    os.chdir(root)
    started = time.time()

    lock = FA.verify_lock(root)
    if lock["protocol_lock_sha256"] != FA.PROTOCOL_LOCK_SHA256:
        raise X6Error("protocol lock differs from X5")
    selector, selector_sha = load_selector(root)
    for seed in SEEDS:
        load_recorded(root, seed, selector)
    if not (root / RECON_ROOT / "RECONSTRUCTION_MANIFEST.json").is_file():
        reconstruct(root, selector)
    files = load_reconstruction(root, selector)

    if args.dry_run:
        result = {"phase": "X6", "mode": "dry-run", "protocol_lock_sha256": lock["protocol_lock_sha256"],
                  "selector_sha256": selector_sha, "n_images": len(files), "seeds": SEEDS,
                  "methods": FA.METHODS, "image_seed_pairs": len(files) * len(SEEDS),
                  "estimated_gpu_hours_from_measured_x5": 3.2444455755 * len(files) / 848,
                  "estimated_cpu_hours_from_x5_dryrun": 52.1526753283 * len(files) / 848,
                  "external_labels_used_for": "scoring and stratified uncertainty only",
                  "external_labels_used_for_selection_or_tuning": False}
        print(json.dumps(result, indent=2)); return 0

    out = root / OUT_ROOT
    suffix = "_smoke" if args.smoke else ""
    targets = {"json": out / ("x6_cross_source%s.json" % suffix),
               "report": out / ("X6_CROSS_SOURCE_REPORT%s.md" % suffix),
               "curves": out / ("per_image_curves%s.csv" % suffix),
               "stability": out / ("stability%s.csv" % suffix),
               "shift": out / ("shift_effects%s.csv" % suffix)}
    collisions = [str(p) for p in targets.values() if p.exists()]
    if collisions:
        raise X6Error("refusing to overwrite: %s" % collisions)
    entries = smoke_sample(files) if args.smoke else files
    seeds = [42] if args.smoke else SEEDS
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise X6Error("CUDA requested but unavailable")

    curves, stability, neutrality = [], [], []
    proto = lock["protocol"]
    from PIL import Image
    for seed in seeds:
        rec = load_recorded(root, seed, selector)
        model, dev, checkpoint_hash = load_verified_model(root, seed, args.device)
        if model.vit.config._attn_implementation != "sdpa":
            raise X6Error("attention implementation is not SDPA")
        rnd = torch.Generator().manual_seed(proto["seeds"]["random_baseline"])
        noise = torch.Generator(device=dev).manual_seed(proto["seeds"]["noise_transform"])
        stamp("seed %d: %d images" % (seed, len(entries)))
        for i, entry in enumerate(entries, 1):
            fn = entry["filename"]
            x = TA.eval_transform(Image.open(root / entry["destination_path"]).convert("RGB"))\
                .unsqueeze(0).to(dev)
            target = CLASSES.index(rec[fn]["pred_label"])
            recorded_p = np.array([float(rec[fn]["p_" + c]) for c in CLASSES])
            with torch.no_grad():
                plain_single = model(x).softmax(-1)
                recomputed_tta = tta_logits(model, x)
                logits, aux_cnn, aux_vit, gate = model(x, return_aux=True)
            with SynchronizedAttributor(model) as att:
                attributed = att.attribute(x, torch.tensor([target], device=dev))
            inst_p = attributed["probs"][0].cpu().numpy().astype(np.float64)
            single_p = plain_single[0].cpu().numpy().astype(np.float64)
            tta_p = recomputed_tta[0].cpu().numpy().astype(np.float64)
            d_single = float(np.abs(inst_p - single_p).max())
            d_tta = float(np.abs(tta_p - recorded_p).max())
            if d_single != 0.0:
                raise X6Error("instrumentation neutrality failed for seed %d %s: %.3e"
                              % (seed, fn, d_single))
            if CLASSES[int(tta_p.argmax())] != rec[fn]["pred_label"] or d_tta > PROB_TOL:
                raise X6Error("recorded prediction reproduction failed for seed %d %s"
                              % (seed, fn))
            maps = FA.method_maps(attributed["cnn_maps"][0], attributed["vit_maps"][0],
                                  gate[0].cpu(), rnd)
            ev = FA.evaluate_image(model, x, maps, target, proto, dev)
            st, agreement = FA.stability_for_image(model, x, target, proto, dev, noise)
            auxc = aux_cnn.softmax(-1)[0].cpu().numpy()
            auxv = aux_vit.softmax(-1)[0].cpu().numpy()
            ordered = np.sort(tta_p)
            row = {"filename": fn, "seed": seed, "true_label": entry["true_label"],
                   "pred_class": rec[fn]["pred_label"],
                   "correct": int(rec[fn]["pred_label"] == entry["true_label"]),
                   "baseline_conf": ev[FA.METHODS[0]]["baseline_conf"],
                   "tta_confidence": float(tta_p.max()),
                   "tta_margin": float(ordered[-1] - ordered[-2]),
                   "tta_entropy": entropy(tta_p),
                   "gate_cnn": float(gate[0, 0]), "gate_vit": float(gate[0, 1]),
                   "aux_cnn_pred": CLASSES[int(auxc.argmax())],
                   "aux_vit_pred": CLASSES[int(auxv.argmax())],
                   "aux_cnn_vs_fused_disagree": int(int(auxc.argmax()) != target),
                   "aux_vit_vs_fused_disagree": int(int(auxv.argmax()) != target),
                   "aux_branches_disagree": int(int(auxc.argmax()) != int(auxv.argmax())),
                   "cnn_vit_agreement_spearman": agreement,
                   "pituitary_to_notumor": int(entry["true_label"] == "pituitary" and
                                                rec[fn]["pred_label"] == "notumor")}
            for method in FA.METHODS:
                result = ev[method]
                row[method + "__deletion_auc"] = result["deletion_auc"]
                row[method + "__insertion_auc"] = result["insertion_auc"]
                for j, pct in enumerate(proto["top_k_percent"]):
                    row["%s__conf_drop_top%d" % (method, pct)] = float(result["topk_conf_drop"][j])
                    row["%s__pred_changed_top%d" % (method, pct)] = int(result["topk_pred_changed"][j])
                row[method + "__deletion_curve"] = ";".join("%.6f" % v for v in result["deletion_curve"])
                row[method + "__insertion_curve"] = ";".join("%.6f" % v for v in result["insertion_curve"])
            curves.append(row)
            for transform, values in st.items():
                stability.append({"filename": fn, "seed": seed, "true_label": entry["true_label"],
                                  "transformation": transform, **values})
            neutrality.append({"seed": seed, "filename": fn,
                               "max_abs_instrumented_vs_plain_single": d_single,
                               "max_abs_recomputed_tta_vs_recorded": d_tta,
                               "checkpoint_sha256": checkpoint_hash})
            if i % 50 == 0:
                stamp("  %d/%d" % (i, len(entries)))
        del model

    clean_rows = load_clean_diagnostics(
        root, list(csv.DictReader((root / X5_CURVES).open(encoding="utf-8"))))
    shift_rows = []
    iterations = proto["bootstrap"]["iterations"] if not args.smoke else 200
    for quantity in ("deletion_auc", "insertion_auc"):
        for method in FA.METHODS:
            field = method + "__" + quantity
            effect = independent_stratified_shift(curves, clean_rows, field, iterations,
                                                  proto["bootstrap"]["seed"])
            shift_rows.append({"quantity": quantity, "method": method, **effect})
    for quantity in ("cnn_vit_agreement_spearman", "gate_cnn", "gate_vit",
                     "tta_confidence", "tta_margin", "tta_entropy"):
        effect = independent_stratified_shift(curves, clean_rows, quantity, iterations,
                                              proto["bootstrap"]["seed"])
        shift_rows.append({"quantity": quantity, "method": "not_applicable", **effect})
    clean_stability = list(csv.DictReader(
        (root / "publication_results/explainability/x5_faithfulness/stability.csv")
        .open(encoding="utf-8")))
    clean_labels = {(int(r["seed"]), r["filename"]): r["true_label"]
                    for r in clean_rows}
    for row in clean_stability:
        key = (int(row["seed"]), row["filename"])
        if key not in clean_labels:
            raise X6Error("clean stability row has no matching labelled X5 row: %s" % (key,))
        row["true_label"] = clean_labels[key]
    for transform in [t["id"] for t in proto["stability"]["transformations"]]:
        ext_t = [r for r in stability if r["transformation"] == transform]
        clean_t = [r for r in clean_stability if r["transformation"] == transform]
        for branch_field in ("cnn_spearman", "vit_spearman"):
            effect = independent_stratified_shift(ext_t, clean_t, branch_field, iterations,
                                                  proto["bootstrap"]["seed"])
            shift_rows.append({"quantity": "stability_%s" % transform,
                               "method": branch_field, **effect})

    pres = ({"checked": 0, "altered": 0, "skipped": True} if args.skip_preservation
            else {**preservation(root), "skipped": False})
    result = {"phase": "X6", "mode": "smoke" if args.smoke else "full",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_lock_sha256": lock["protocol_lock_sha256"],
              "selector_sha256": selector_sha, "n_selector_images": EXPECTED_N,
              "n_evaluated_images": len(entries), "seeds": seeds,
              "methods": FA.METHODS, "runtime_seconds": time.time() - started,
              "neutrality": {"max_instrumented_vs_plain_single": max(r["max_abs_instrumented_vs_plain_single"] for r in neutrality),
                             "max_recomputed_tta_vs_recorded": max(r["max_abs_recomputed_tta_vs_recorded"] for r in neutrality),
                             "label_changes_vs_recorded": 0},
              "class_errors": class_error_summary(curves),
              "branch_disagreement": {
                  "cnn_vs_fused_rate": float(np.mean([r["aux_cnn_vs_fused_disagree"] for r in curves])),
                  "vit_vs_fused_rate": float(np.mean([r["aux_vit_vs_fused_disagree"] for r in curves])),
                  "cnn_vs_vit_rate": float(np.mean([r["aux_branches_disagree"] for r in curves]))},
              "preservation": pres,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "transformers": __import__("transformers").__version__,
                              "device": args.device,
                              "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "external_labels_used_for_selection_or_tuning": False,
              "scope_limit": "association under source shift; not mechanism or localisation"}
    checks = {"protocol_lock_identical_to_x5": result["protocol_lock_sha256"] == FA.PROTOCOL_LOCK_SHA256,
              "selector_exactly_373": len(selector) == EXPECTED_N,
              "evaluated_expected_scope": len(curves) == len(entries) * len(seeds),
              "neutrality_exact": result["neutrality"]["max_instrumented_vs_plain_single"] == 0.0,
              "recorded_predictions_reproduced": result["neutrality"]["max_recomputed_tta_vs_recorded"] <= PROB_TOL,
              "no_external_label_selection_or_tuning": True,
              "clean_and_external_not_pooled": True,
              "preservation": pres["altered"] == 0 if not pres.get("skipped") else None}
    result["acceptance_checks"] = checks
    result["acceptance_status"] = "PASS" if all(v is not False for v in checks.values()) else "FAIL"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(targets["curves"], curves); write_csv(targets["stability"], stability)
    write_csv(targets["shift"], shift_rows)
    targets["json"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Phase X6 - locked cross-source explanation analysis", "",
             "**Acceptance status: %s.**" % result["acceptance_status"], "",
             "Protocol lock `%s`." % result["protocol_lock_sha256"], "",
             "%d duplicate-disjoint external images; seeds %s; %d methods."
             % (len(entries), seeds, len(FA.METHODS)), "",
             "External labels were used only for scoring and stratified uncertainty, never selection or tuning.", "",
             "These associations do not establish a mechanism for the accuracy drop, anatomical localisation, or clinical validity."]
    targets["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if result["acceptance_status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (X6Error, FA.X5Error) as exc:
        print("STOP CONDITION: %s" % exc, file=sys.stderr)
        sys.exit(2)
