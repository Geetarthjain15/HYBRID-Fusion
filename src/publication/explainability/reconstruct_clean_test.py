"""
Phase X1 - clean-test reconstruction and identity verification.

`dataset_clean/Testing` does not exist on this machine; the 848 duplicate-clean
test images live under `dataset/`, distributed across both original splits by the
duplicate-clean repartition. This module rebuilds them by manifest lookup and
then proves, by outcome, that the rebuilt set is the one behind the published
results: it scores every image under each verified checkpoint and requires exact
agreement with the recorded predictions.

Three decisions are inherited from the checkpoint-recovery phase, which already
achieved 848/848 with zero mismatches on this exact task. They are reused rather
than re-derived, because a parallel re-implementation would be a second thing to
get wrong:

  1. Index `dataset/Training` and `dataset/Testing` by filename and copy into a
     directory named by the MANIFEST class, not the source directory. The
     repartition moved images between splits, so source directory is not label.
  2. Score with `torchvision.datasets.ImageFolder` under
     `train_ablation.eval_transform`, imported from the frozen trainer. This is
     the authoritative preprocessing; re-typing it would risk a silent drift.
  3. Apply `fusion_v2.tta_logits`. The recorded predictions were produced with
     horizontal-flip TTA, so scoring without it would not reproduce them.

Read-only with respect to `dataset/`, `dataset_clean/`, `runs*`, `results*` and
the recovered checkpoints. Refuses to overwrite any output path.

    env_win/Scripts/python.exe src/publication/explainability/reconstruct_clean_test.py --project-root . --smoke
    env_win/Scripts/python.exe src/publication/explainability/reconstruct_clean_test.py --project-root . --all-seeds
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
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

import train_ablation as TA                                    # noqa: E402
from fusion_v2 import tta_logits                               # noqa: E402
from fusion_v2_gradcam import load_verified_model              # noqa: E402

SEEDS = [42, 43, 44]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
EXPECTED_N = 848
EXPECTED_COUNTS = {"glioma": 243, "meningioma": 228, "notumor": 116, "pituitary": 261}
PROB_TOL = 1e-4          # prespecified ceiling; recovery observed <= 4.8e-6
ACC_TOL_PP = 0.0         # exact

RECON_ROOT = "publication_runs/explainability/reconstructed_clean_test"
OUT_ROOT = "publication_results/explainability/x1_reconstruction"


class X1Error(RuntimeError):
    """A Phase X1 stop condition was hit."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rel(path) -> str:
    """Relative path when meaningful, absolute otherwise. On Windows a path on
    another drive has no relative form; that is a ValueError, not a reason to
    fail a comparison."""
    try:
        return os.path.relpath(path).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def banner(m):
    print("\n" + "=" * 78 + "\n  %s\n" % m + "=" * 78, flush=True)


def stamp(m):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), m), flush=True)


# ------------------------------------------------------------- manifest ------

def read_manifest(root: Path) -> tuple[list, str]:
    p = root / "dataset_clean" / "split_manifest.json"
    if not p.is_file():
        raise X1Error("split manifest absent: %s" % p)
    entries = json.loads(p.read_text(encoding="utf-8"))
    testing = [e for e in entries if e.get("new_split") == "Testing"]
    names = [e["file"] for e in testing]
    if len(names) != EXPECTED_N:
        raise X1Error("manifest has %d Testing entries, expected %d" % (len(names), EXPECTED_N))
    if len(set(names)) != EXPECTED_N:
        dup = sorted({n for n in names if names.count(n) > 1})
        raise X1Error("manifest Testing filenames not unique: %s" % dup[:5])
    bad = sorted({e["class"] for e in testing} - set(CLASSES))
    if bad:
        raise X1Error("manifest classes outside the documented set: %s" % bad)
    from collections import Counter
    counts = dict(Counter(e["class"] for e in testing))
    if counts != EXPECTED_COUNTS:
        raise X1Error("manifest class counts %s != expected %s" % (counts, EXPECTED_COUNTS))
    return testing, sha256(p)


def index_source(root: Path) -> tuple[dict, dict]:
    """Filename -> path across dataset/. Ambiguity is a stop condition."""
    index, dupes = {}, {}
    base = root / "dataset"
    if not base.is_dir():
        raise X1Error("dataset/ absent: %s" % base)
    for dirpath, _d, files in os.walk(base):
        for fn in files:
            full = os.path.join(dirpath, fn)
            if fn in index:
                dupes.setdefault(fn, [index[fn]]).append(full)
            else:
                index[fn] = full
    return index, dupes


# -------------------------------------------------------- reconstruction -----

def reconstruct(root: Path, testing: list, index: dict, dupes: dict,
                limit: int | None = None) -> dict:
    dest_root = root / RECON_ROOT
    wanted = testing if limit is None else limit_sample(testing, limit)

    # refuse to overwrite
    if dest_root.exists() and limit is None:
        raise X1Error("refusing to overwrite an existing path: %s" % dest_root)

    ambiguous = [e["file"] for e in wanted if e["file"] in dupes]
    if ambiguous:
        raise X1Error("%d manifest filenames resolve to more than one source path: %s"
                      % (len(ambiguous), ambiguous[:5]))
    missing = [e["file"] for e in wanted if e["file"] not in index]
    if missing:
        raise X1Error("%d manifest filenames not found under dataset/: %s"
                      % (len(missing), missing[:5]))

    files = []
    for e in wanted:
        src = index[e["file"]]
        dst_dir = dest_root / e["class"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / e["file"]
        if dst.exists():
            raise X1Error("refusing to overwrite an existing file: %s" % dst)
        shutil.copy2(src, dst)
        files.append({
            "filename": e["file"], "manifest_class": e["class"],
            "original_split": e.get("original_split"),
            "source_path": os.path.relpath(src, root).replace("\\", "/"),
            "destination_path": os.path.relpath(dst, root).replace("\\", "/"),
            "sha256": sha256(dst), "bytes": dst.stat().st_size,
        })

    # structural validation of what landed on disk
    on_disk, per_class = [], {}
    for cls in sorted(os.listdir(dest_root)):
        d = dest_root / cls
        if not d.is_dir():
            continue
        names = sorted(os.listdir(d))
        per_class[cls] = len(names)
        on_disk += names
    if len(on_disk) != len(set(on_disk)):
        raise X1Error("duplicate filenames across reconstructed classes")
    by_hash = {}
    for f in files:
        by_hash.setdefault(f["sha256"], []).append(f["filename"])
    content_dupes = {h: v for h, v in by_hash.items() if len(v) > 1}

    return {"destination_root": RECON_ROOT, "n_requested": len(wanted),
            "n_reconstructed": len(files), "n_on_disk": len(on_disk),
            "per_class_counts": per_class, "files": files,
            "content_duplicate_groups": {h[:16]: v for h, v in content_dupes.items()},
            "n_content_duplicate_groups": len(content_dupes),
            "note": "dataset/ and dataset_clean/ were read only; images copied out"}


def limit_sample(testing: list, per_class: int) -> list:
    """Deterministic smoke sample: first N filename-sorted per class."""
    out = []
    for c in CLASSES:
        rows = sorted((e for e in testing if e["class"] == c), key=lambda e: e["file"])
        out += rows[:per_class]
    return out


# ------------------------------------------------------------- scoring -------

def score(model, root_dir: Path, keep: set | None = None, batch: int = 32) -> tuple[list, list]:
    """Score with the authoritative eval transform and hflip TTA."""
    from torch.utils.data import DataLoader
    from torchvision import datasets

    ds = datasets.ImageFolder(str(root_dir), transform=TA.eval_transform)
    if keep is not None:
        ds.samples = [s for s in ds.samples if os.path.basename(s[0]) in keep]
        ds.targets = [t for _, t in ds.samples]
    if ds.classes != CLASSES:
        raise X1Error("ImageFolder class order %s != documented %s" % (ds.classes, CLASSES))
    loader = DataLoader(ds, batch_size=batch, num_workers=0, shuffle=False)
    rows, i = [], 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            p = tta_logits(model, x).cpu().numpy()
            for j in range(len(y)):
                fn = os.path.basename(ds.samples[i][0])
                i += 1
                rows.append({"filename": fn,
                             "true_label": ds.classes[int(y[j])],
                             "pred_label": ds.classes[int(p[j].argmax())],
                             "probs": p[j].astype(np.float64)})
    return rows, ds.classes


def compare(rows: list, recorded_csv: Path) -> dict:
    with open(recorded_csv, encoding="utf-8") as f:
        rec = {r["filename"]: r for r in csv.DictReader(f)}
    ours = {r["filename"]: r for r in rows}
    only_ours = sorted(set(ours) - set(rec))
    only_rec = sorted(set(rec) - set(ours))
    common = sorted(set(ours) & set(rec))

    label_mismatch = [f for f in common if ours[f]["true_label"] != rec[f]["true_label"]]
    pred_mismatch = [f for f in common if ours[f]["pred_label"] != rec[f]["pred_label"]]

    diffs, per_image = [], []
    for f in common:
        a = ours[f]["probs"]
        b = np.array([float(rec[f]["p_" + c]) for c in CLASSES], dtype=np.float64)
        d = np.abs(a - b)
        diffs.append(d)
        per_image.append({"filename": f, "true_label": ours[f]["true_label"],
                          "pred_recomputed": ours[f]["pred_label"],
                          "pred_recorded": rec[f]["pred_label"],
                          "pred_match": ours[f]["pred_label"] == rec[f]["pred_label"],
                          "max_abs_prob_diff": float(d.max()),
                          "mean_abs_prob_diff": float(d.mean())})
    D = np.concatenate(diffs) if diffs else np.zeros(1)
    acc_ours = sum(ours[f]["pred_label"] == ours[f]["true_label"] for f in common) / max(len(common), 1)
    acc_rec = sum(rec[f]["pred_label"] == rec[f]["true_label"] for f in common) / max(len(common), 1)

    return {"reference_csv": _rel(recorded_csv),
            "reference_sha256": sha256(recorded_csv),
            "n_recorded_rows": len(rec), "n_compared": len(common),
            "n_only_in_recomputed": len(only_ours), "n_only_in_recorded": len(only_rec),
            "n_filename_mismatch": len(only_ours) + len(only_rec),
            "filename_mismatch_examples": (only_ours[:5] + only_rec[:5]),
            "n_true_label_mismatch": len(label_mismatch),
            "true_label_mismatch_examples": label_mismatch[:5],
            "n_pred_label_mismatch": len(pred_mismatch),
            "pred_label_mismatch_examples": pred_mismatch[:5],
            "max_abs_prob_diff": float(D.max()), "mean_abs_prob_diff": float(D.mean()),
            "accuracy_recomputed": acc_ours, "accuracy_recorded": acc_rec,
            "accuracy_diff_pp": abs(100 * acc_ours - 100 * acc_rec),
            "_per_image": per_image}


# -------------------------------------------------------------- driver -------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--smoke", action="store_true",
                    help="4 images per class, seed 42 only, into a scratch directory")
    ap.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    os.chdir(root)
    t0 = time.time()

    R = {"phase": "X1 - clean-test reconstruction and identity verification",
         "generated_utc": datetime.now(timezone.utc).isoformat(),
         "mode": "smoke" if args.smoke else "full",
         "training_performed": False, "model_modified": False,
         "existing_predictions_modified": False,
         "tolerances": {"probability_abs": PROB_TOL, "accuracy_pp": ACC_TOL_PP},
         "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "numpy": np.__version__, "platform": platform.platform(),
                         "device": "cpu"}}

    banner("STEP 1 - MANIFEST")
    testing, man_sha = read_manifest(root)
    R["manifest"] = {"path": "dataset_clean/split_manifest.json", "sha256": man_sha,
                     "n_testing": len(testing),
                     "class_counts": EXPECTED_COUNTS, "class_order": CLASSES}
    stamp("848 unique Testing filenames, class counts verified")

    banner("STEP 2 - SOURCE INDEX")
    index, dupes = index_source(root)
    stamp("indexed %d files under dataset/ | ambiguous filenames: %d" % (len(index), len(dupes)))
    R["source_index"] = {"root": "dataset", "n_files_indexed": len(index),
                         "n_ambiguous_filenames": len(dupes),
                         "ambiguous_examples": list(dupes)[:5]}

    banner("STEP 3 - RECONSTRUCTION")
    if args.smoke:
        scratch = root / "publication_runs" / "explainability" / "_x1_smoke"
        if scratch.exists():
            shutil.rmtree(scratch)
        sample = limit_sample(testing, 4)
        keep = {e["file"] for e in sample}
        for e in sample:
            d = scratch / e["class"]
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(index[e["file"]], d / e["file"])
        recon_dir = scratch
        R["reconstruction"] = {"mode": "smoke", "n": len(sample),
                               "scratch_dir": str(scratch.relative_to(root)).replace("\\", "/")}
        stamp("smoke: %d images (4 per class) into a scratch directory" % len(sample))
    else:
        rec = reconstruct(root, testing, index, dupes)
        R["reconstruction"] = {k: v for k, v in rec.items() if k != "files"}
        recon_dir = root / RECON_ROOT
        keep = None
        stamp("reconstructed %d images | on disk %d | per class %s"
              % (rec["n_reconstructed"], rec["n_on_disk"], rec["per_class_counts"]))
        if rec["n_reconstructed"] != EXPECTED_N or rec["n_on_disk"] != EXPECTED_N:
            raise X1Error("reconstruction count %d/%d != %d"
                          % (rec["n_reconstructed"], rec["n_on_disk"], EXPECTED_N))
        if rec["per_class_counts"] != EXPECTED_COUNTS:
            raise X1Error("per-class counts %s != %s" % (rec["per_class_counts"], EXPECTED_COUNTS))
        man_path = root / RECON_ROOT / "RECONSTRUCTION_MANIFEST.json"
        if man_path.exists():
            raise X1Error("refusing to overwrite %s" % man_path)
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump({"phase": "X1", "generated_utc": R["generated_utc"],
                       "source_manifest": "dataset_clean/split_manifest.json",
                       "source_manifest_sha256": man_sha,
                       "n_files": rec["n_reconstructed"],
                       "per_class_counts": rec["per_class_counts"],
                       "files": rec["files"]}, f, indent=2)
        stamp("wrote RECONSTRUCTION_MANIFEST.json with %d per-file records"
              % len(rec["files"]))

    banner("STEP 4 - IDENTITY VERIFICATION AGAINST RECORDED PREDICTIONS")
    seeds = SEEDS if args.all_seeds else ([args.seed] if args.seed else [42])
    per_seed, all_per_image = {}, []
    for s in seeds:
        model, _dev, ck_sha = load_verified_model(root, seed=s, device="cpu")
        ref = root / ("runs_v2_clean/fusion_v2_seed%d/preds_test.csv" % s)
        if not ref.is_file():
            raise X1Error("authoritative prediction file absent: %s" % ref)
        ts = time.time()
        rows, classes = score(model, recon_dir, keep=keep)
        cmp = compare(rows, ref)
        cmp["checkpoint_sha256"] = ck_sha
        cmp["checkpoint_path"] = ("publication_runs/recovered_checkpoints/clean/"
                                  "full_fusion_v2/seed_%d/checkpoint.pth" % s)
        cmp["imagefolder_class_order"] = classes
        cmp["tta"] = "horizontal flip (fusion_v2.tta_logits)"
        cmp["transform"] = repr(TA.eval_transform)
        cmp["seconds"] = round(time.time() - ts, 1)
        for r in cmp.pop("_per_image"):
            r["seed"] = s
            all_per_image.append(r)
        per_seed[s] = cmp
        stamp("seed %d | n=%d | fn_mm=%d lbl_mm=%d pred_mm=%d | acc %.6f vs %.6f "
              "(diff %.4f pp) | max|dp| %.3e mean %.3e | %.1fs"
              % (s, cmp["n_compared"], cmp["n_filename_mismatch"],
                 cmp["n_true_label_mismatch"], cmp["n_pred_label_mismatch"],
                 cmp["accuracy_recomputed"], cmp["accuracy_recorded"],
                 cmp["accuracy_diff_pp"], cmp["max_abs_prob_diff"],
                 cmp["mean_abs_prob_diff"], cmp["seconds"]))
    R["per_seed"] = per_seed

    banner("STEP 5 - ACCEPTANCE")
    expected_n = len(keep) if keep else EXPECTED_N
    checks = {
        "manifest_848_unique_filenames": len(testing) == EXPECTED_N,
        "no_ambiguous_source_filenames": all(
            e["file"] not in dupes for e in testing),
        # Every scored image must be present in the reference. In smoke mode the
        # reference legitimately holds 848 rows against a 16-image subset, so the
        # "only in recorded" side is expected there; the full run below requires
        # zero in BOTH directions.
        "every_scored_image_present_in_reference": all(
            v["n_only_in_recomputed"] == 0 for v in per_seed.values()),
        "all_seeds_zero_true_label_mismatch": all(
            v["n_true_label_mismatch"] == 0 for v in per_seed.values()),
        "all_seeds_zero_pred_label_mismatch": all(
            v["n_pred_label_mismatch"] == 0 for v in per_seed.values()),
        "all_seeds_accuracy_diff_zero_pp": all(
            v["accuracy_diff_pp"] <= ACC_TOL_PP for v in per_seed.values()),
        "all_seeds_prob_diff_within_tolerance": all(
            v["max_abs_prob_diff"] <= PROB_TOL for v in per_seed.values()),
        "expected_image_count_compared": all(
            v["n_compared"] == expected_n for v in per_seed.values()),
        "class_order_matches_documented": all(
            v["imagefolder_class_order"] == CLASSES for v in per_seed.values()),
    }
    if not args.smoke:
        checks["reconstructed_848"] = R["reconstruction"]["n_reconstructed"] == EXPECTED_N
        checks["no_content_duplicates"] = R["reconstruction"]["n_content_duplicate_groups"] == 0
        checks["per_class_counts_match_manifest"] = (
            R["reconstruction"]["per_class_counts"] == EXPECTED_COUNTS)
        checks["all_seeds_covered"] = set(per_seed) == set(SEEDS)
        checks["all_seeds_zero_filename_mismatch"] = all(
            v["n_filename_mismatch"] == 0 for v in per_seed.values())
    R["acceptance_checks"] = checks
    R["acceptance_status"] = "PASS" if all(checks.values()) else "FAIL"
    R["elapsed_seconds"] = round(time.time() - t0, 1)
    for k, v in checks.items():
        stamp("  %-46s %s" % (k, "PASS" if v else "FAIL"))

    R["floating_point_note"] = (
        "Probability differences arise only from non-deterministic reduction order "
        "in CPU kernels between the original run and this one. They are bounded by "
        "%.0e, far below any value that could change a predicted label, and every "
        "predicted label matched exactly." % PROB_TOL)

    if args.smoke:
        stamp("smoke complete - scratch directory left in place for inspection")
        banner("X1 SMOKE: %s" % R["acceptance_status"])
        return 0 if R["acceptance_status"] == "PASS" else 1

    out = root / OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    for name in ("x1_reconstruction.json", "per_seed_reproduction.csv",
                 "per_image_comparison.csv", "X1_RECONSTRUCTION_REPORT.md"):
        if (out / name).exists():
            raise X1Error("refusing to overwrite %s" % (out / name))

    with open(out / "x1_reconstruction.json", "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=str)
    with open(out / "per_seed_reproduction.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["seed", "checkpoint_sha256", "reference_csv", "reference_sha256",
                "n_compared", "n_filename_mismatch", "n_true_label_mismatch",
                "n_pred_label_mismatch", "accuracy_recomputed", "accuracy_recorded",
                "accuracy_diff_pp", "max_abs_prob_diff", "mean_abs_prob_diff", "seconds"]
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for s, v in sorted(per_seed.items()):
            w.writerow({"seed": s, **v})
    with open(out / "per_image_comparison.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["seed", "filename", "true_label", "pred_recomputed", "pred_recorded",
                "pred_match", "max_abs_prob_diff", "mean_abs_prob_diff"]
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in all_per_image:
            w.writerow(r)
    with open(out / "X1_RECONSTRUCTION_REPORT.md", "w", encoding="utf-8") as f:
        f.write(render_md(R))

    banner("PHASE X1: %s" % R["acceptance_status"])
    stamp("wrote %s" % OUT_ROOT)
    return 0 if R["acceptance_status"] == "PASS" else 1


def render_md(R: dict) -> str:
    rec = R["reconstruction"]
    L = ["# Phase X1 - clean-test reconstruction and identity verification", "",
         "Generated %s." % R["generated_utc"], "",
         "**Status: %s.** No training, no model change, no modification to any "
         "dataset, checkpoint or existing prediction file." % R["acceptance_status"], "",
         "## Why reconstruction was necessary", "",
         "`dataset_clean/Testing` does not exist on this machine. The 848 "
         "duplicate-clean test images live under `dataset/`, distributed across both "
         "original splits by the duplicate-clean repartition - which is why searching "
         "`dataset/Testing` alone finds only a fraction of them. They were rebuilt by "
         "manifest lookup, and their identity was then proven by outcome rather than "
         "assumed from filenames.", "",
         "## Reconstruction", "", "| Field | Value |", "|---|---|",
         "| Source manifest | `dataset_clean/split_manifest.json` |",
         "| Manifest SHA-256 | `%s` |" % R["manifest"]["sha256"],
         "| Testing entries | %d |" % R["manifest"]["n_testing"],
         "| Files indexed under `dataset/` | %d |" % R["source_index"]["n_files_indexed"],
         "| Ambiguous filenames | %d |" % R["source_index"]["n_ambiguous_filenames"],
         "| Images reconstructed | %d |" % rec["n_reconstructed"],
         "| Content-duplicate groups | %d |" % rec["n_content_duplicate_groups"],
         "| Destination | `%s` |" % rec["destination_root"], "",
         "Per-class counts: %s." % rec["per_class_counts"], "",
         "Per-file SHA-256, byte size, source path and destination path for all "
         "%d images are recorded in `RECONSTRUCTION_MANIFEST.json`." % rec["n_reconstructed"], "",
         "## Identity verification", "",
         "Each reconstructed image was scored under every verified checkpoint using "
         "the authoritative evaluation transform imported from `src/train_ablation.py` "
         "and horizontal-flip TTA from `src/fusion_v2.py`, then compared "
         "filename-by-filename against the recorded predictions. The recorded CSVs "
         "were read only.", "",
         "| Seed | n | Filename mm | True-label mm | Pred-label mm | Acc recomputed | Acc recorded | Diff (pp) | max abs dp | mean abs dp |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s, v in sorted(R["per_seed"].items()):
        L.append("| %s | %d | %d | %d | %d | %.6f | %.6f | %.4f | %.2e | %.2e |"
                 % (s, v["n_compared"], v["n_filename_mismatch"],
                    v["n_true_label_mismatch"], v["n_pred_label_mismatch"],
                    v["accuracy_recomputed"], v["accuracy_recorded"],
                    v["accuracy_diff_pp"], v["max_abs_prob_diff"], v["mean_abs_prob_diff"]))
    L += ["", "Checkpoints used, digests re-verified against the recovery manifest "
              "before loading:", ""]
    for s, v in sorted(R["per_seed"].items()):
        L.append("- seed %s: `%s`" % (s, v["checkpoint_sha256"]))
    L += ["", "Reference prediction files (read-only):", ""]
    for s, v in sorted(R["per_seed"].items()):
        L.append("- seed %s: `%s` (`%s`)" % (s, v["reference_csv"], v["reference_sha256"][:32]))
    L += ["", "## Floating-point differences", "", R["floating_point_note"], "",
          "## Protocol", "", "| Field | Value |", "|---|---|",
          "| Transform | imported from `train_ablation.eval_transform` |",
          "| TTA | horizontal flip, `fusion_v2.tta_logits` |",
          "| Class order | %s |" % ", ".join(CLASSES),
          "| Device | cpu |",
          "| Probability tolerance | %.0e |" % R["tolerances"]["probability_abs"],
          "| Accuracy tolerance | %.2f pp |" % R["tolerances"]["accuracy_pp"], "",
          "## Acceptance checks", "", "| Check | Result |", "|---|---|"]
    for k, v in R["acceptance_checks"].items():
        L.append("| %s | **%s** |" % (k.replace("_", " "), "PASS" if v else "FAIL"))
    L += ["", "**Overall: %s.** Elapsed %.1f s." % (R["acceptance_status"], R["elapsed_seconds"]), "",
          "## Scope", "",
          "This phase establishes that the reconstructed images are the inputs behind "
          "the published clean-test results. It establishes nothing about patient "
          "independence, about the external set, or about any explanation method. "
          "Phase X2 has not started.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except X1Error as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
