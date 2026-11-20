"""
Phase 0 - immutable baseline audit.

Records the exact state of every existing prediction CSV and metric JSON
BEFORE any publication work begins, so that accidental regression later is
detectable by hash rather than by memory. Read-only with respect to every
existing artifact: the only writes are the two report files under --out-root.

What it establishes:
  1. INVENTORY   path, byte size, mtime, SHA-256, row count for every
                 prediction CSV and metric JSON under the immutable roots.
  2. VALIDATION  per-CSV schema, class ordering, filename uniqueness,
                 probability range and sum, and agreement between the stored
                 pred_label and the argmax of the stored probabilities.
  3. RECOMPUTE   accuracy and macro-F1 recomputed from the FusionV2
                 prediction CSVs themselves - never copied from a table.
  4. COMPARE     recomputed three-seed summaries against the verified
                 constants in CLAUDE.md, at 0.01 percentage-point tolerance.

Every check that fails is recorded in the manifest and the Markdown report
rather than silently corrected, and the process exits non-zero so a failing
audit cannot be mistaken for a passing one.

    python src/publication/audit_baseline.py --project-root . \
        --out-root publication_results/audit
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Verified constants. Source: CLAUDE.md "Scientific ground truth". These are
# the values the audit must reproduce; they are NOT recomputed from a table.
# ---------------------------------------------------------------------------
VERIFIED = {
    "original": {"acc": (99.14, 0.10), "macro_f1": (99.12, 0.09)},
    "clean": {"acc": (98.27, 0.07), "macro_f1": (98.28, 0.04)},
    "cross_source_disjoint": {"acc": (69.53, 1.98)},
}
VERIFIED_DISJOINT_N = 373
TOLERANCE_PP = 0.01          # percentage points
PROB_SUM_TOL = 1e-3

IMMUTABLE_DIRS = [
    "runs", "runs_original", "runs_clean", "runs_v2_original", "runs_v2_clean",
    "results", "results_original", "results_clean",
    "results_v2_original", "results_v2_clean",
]

# FusionV2 prediction CSVs, by evaluation setting.
FUSION_V2_SOURCES = {
    "original": ("runs_v2_original", "preds_test.csv"),
    "clean": ("runs_v2_clean", "preds_test.csv"),
    "external_full_original": ("runs_v2_original", "preds_dataset_ood.csv"),
    "external_full_clean": ("runs_v2_clean", "preds_dataset_ood.csv"),
}
SEEDS = [42, 43, 44]


class AuditError(Exception):
    """A stop condition from CLAUDE.md was hit."""


# ---------------------------------------------------------------- inventory ---

def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stat_entry(root: str, path: str) -> dict:
    st = os.stat(path)
    return {
        "path": os.path.relpath(path, root).replace("\\", "/"),
        "bytes": st.st_size,
        "mtime_utc": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def inventory(root: str) -> tuple[list, list]:
    preds, metrics = [], []
    for d in IMMUTABLE_DIRS:
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in sorted(filenames):
                p = os.path.join(dirpath, fn)
                if fn.endswith(".csv"):
                    e = stat_entry(root, p)
                    e["rows"] = sum(1 for _ in open(p, encoding="utf-8")) - 1
                    (preds if fn.startswith("preds_") else metrics).append(e)
                    if fn.startswith("gates_"):
                        e["kind"] = "gates"
                elif fn.endswith(".json"):
                    metrics.append(stat_entry(root, p))
    return preds, metrics


# --------------------------------------------------------------- validation ---

def read_predictions(path: str, label: str | None = None) -> dict:
    """Load and fully validate one prediction CSV. Raises AuditError on any
    stop condition so a malformed file cannot silently enter an analysis."""
    path_label = label or path
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise AuditError("empty prediction file: %s" % path_label)

    header = list(rows[0].keys())
    prob_cols = [c for c in header if c.startswith("p_")]
    classes = [c[2:] for c in prob_cols]
    required = ["filename", "true_label", "pred_label"]
    missing = [c for c in required if c not in header]
    if missing:
        raise AuditError("%s: missing required columns %s" % (path_label, missing))
    if len(classes) < 2:
        raise AuditError("%s: fewer than two probability columns" % path_label)
    if classes != sorted(classes):
        raise AuditError("%s: probability columns not in sorted class order: %s"
                         % (path_label, classes))

    files = [r["filename"] for r in rows]
    if len(set(files)) != len(files):
        dupes = sorted({f for f in files if files.count(f) > 1})[:5]
        raise AuditError("%s: duplicate filenames, e.g. %s" % (path_label, dupes))

    seen_labels = {r["true_label"] for r in rows} | {r["pred_label"] for r in rows}
    unknown = sorted(seen_labels - set(classes))
    if unknown:
        raise AuditError("%s: labels outside the probability columns: %s" % (path_label, unknown))

    P = np.array([[float(r[c]) for c in prob_cols] for r in rows], dtype=np.float64)
    if not np.isfinite(P).all():
        raise AuditError("%s: non-finite probabilities" % path_label)
    if P.min() < 0.0 or P.max() > 1.0:
        raise AuditError("%s: probabilities outside [0,1] (min %.6f, max %.6f)"
                         % (path_label, P.min(), P.max()))
    sums = P.sum(1)
    if np.abs(sums - 1.0).max() > PROB_SUM_TOL:
        raise AuditError("%s: probability rows do not sum to 1 (max deviation %.6f)"
                         % (path_label, np.abs(sums - 1.0).max()))

    y = np.array([classes.index(r["true_label"]) for r in rows])
    stored = np.array([classes.index(r["pred_label"]) for r in rows])
    argmax = P.argmax(1)
    mismatch = int((stored != argmax).sum())
    # A tie in the stored probabilities can legitimately break the other way,
    # so only a mismatch on a strictly-ranked row is a real inconsistency.
    if mismatch:
        strict = 0
        for i in np.nonzero(stored != argmax)[0]:
            top2 = np.sort(P[i])[-2:]
            if top2[1] - top2[0] > 1e-9:
                strict += 1
        if strict:
            raise AuditError("%s: pred_label disagrees with argmax on %d strictly "
                             "ranked rows" % (path_label, strict))

    return {"path": path, "classes": classes, "files": files, "y": y,
            "P": P, "pred": stored, "n": len(rows), "argmax_ties": mismatch}


# --------------------------------------------------------------- recompute ---

def accuracy_macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> tuple:
    acc = float((pred == y).mean())
    f1s = []
    for k in range(n_classes):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return acc, float(np.mean(f1s))


def mean_sd(values: list) -> tuple:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def compare(name: str, metric: str, got: tuple, want: tuple) -> dict:
    """Compare a recomputed (mean, sd) pair against the verified constants."""
    d_mean = abs(100 * got[0] - want[0])
    d_sd = abs(100 * got[1] - want[1])
    ok = d_mean <= TOLERANCE_PP and d_sd <= TOLERANCE_PP
    return {"setting": name, "metric": metric,
            "recomputed_mean_pct": round(100 * got[0], 4),
            "recomputed_sd_pct": round(100 * got[1], 4),
            "verified_mean_pct": want[0], "verified_sd_pct": want[1],
            "abs_diff_mean_pp": round(d_mean, 4), "abs_diff_sd_pp": round(d_sd, 4),
            "tolerance_pp": TOLERANCE_PP, "pass": bool(ok)}


# ------------------------------------------------------------------- driver ---

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".",
                    help="Repository root. All immutable paths resolve against it.")
    ap.add_argument("--out-root", default="publication_results/audit",
                    help="Directory for the two generated reports. Created if absent.")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(root, args.out_root)
    # Guard rule 4 of CLAUDE.md: nothing may be written outside publication_*.
    rel_out = os.path.relpath(out_root, root).replace("\\", "/")
    if not rel_out.startswith("publication_"):
        raise AuditError("refusing to write outside publication_*: %s" % rel_out)
    os.makedirs(out_root, exist_ok=True)

    report: dict = {
        "phase": "0 - immutable baseline audit",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "project_root": root,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "tolerance_pp": TOLERANCE_PP,
        "verified_constants_source": "CLAUDE.md :: Scientific ground truth",
        "inventory": {}, "validation": {}, "recomputed": {},
        "comparisons": [], "checks": {}, "failures": [],
    }

    def fail(msg: str) -> None:
        report["failures"].append(msg)
        print("  [FAIL] %s" % msg, flush=True)

    # --- 1 inventory -------------------------------------------------------
    print("1. inventorying immutable artifacts", flush=True)
    preds_inv, metrics_inv = inventory(root)
    report["inventory"] = {
        "prediction_csvs": preds_inv, "metric_files": metrics_inv,
        "n_prediction_csvs": len(preds_inv), "n_metric_files": len(metrics_inv),
        "directories_present": [d for d in IMMUTABLE_DIRS if os.path.isdir(os.path.join(root, d))],
        "directories_absent": [d for d in IMMUTABLE_DIRS if not os.path.isdir(os.path.join(root, d))],
    }
    print("   %d prediction CSVs, %d metric files across %d directories"
          % (len(preds_inv), len(metrics_inv), len(report["inventory"]["directories_present"])),
          flush=True)
    if not preds_inv:
        fail("no prediction CSVs found under the immutable directories")

    # --- 2 validation ------------------------------------------------------
    print("2. validating prediction schemas and alignment", flush=True)
    loaded, class_orders, val_rows = {}, {}, []
    for entry in preds_inv:
        rel = entry["path"]
        p = os.path.join(root, rel)
        if os.path.basename(rel).startswith("gates_"):
            continue
        try:
            d = read_predictions(p, rel)
            loaded[rel] = d
            class_orders.setdefault(tuple(d["classes"]), []).append(rel)
            val_rows.append({"path": rel, "n_rows": d["n"], "classes": d["classes"],
                             "argmax_ties": d["argmax_ties"], "status": "ok"})
        except AuditError as e:
            val_rows.append({"path": rel, "status": "FAILED", "error": str(e)})
            fail(str(e))
    if len(class_orders) > 1:
        fail("inconsistent class ordering across prediction CSVs: %s"
             % {list(k): len(v) for k, v in class_orders.items()})
    report["validation"] = {
        "files": val_rows,
        "n_ok": sum(1 for r in val_rows if r["status"] == "ok"),
        "n_failed": sum(1 for r in val_rows if r["status"] != "ok"),
        "distinct_class_orders": [list(k) for k in class_orders],
    }
    print("   %d validated, %d failed, %d distinct class order(s)"
          % (report["validation"]["n_ok"], report["validation"]["n_failed"], len(class_orders)),
          flush=True)

    # --- 3 duplicate-disjoint subset --------------------------------------
    print("3. confirming duplicate-disjoint cross-source subset", flush=True)
    cs_path = os.path.join(root, "results", "cross_source.json")
    disjoint: set[str] = set()
    if not os.path.exists(cs_path):
        fail("results/cross_source.json is absent; cannot confirm the disjoint subset")
    else:
        cs = json.load(open(cs_path, encoding="utf-8"))
        disjoint = {k for k, v in cs.get("manifest", {}).items() if v.get("kind") == "disjoint"}
        report["cross_source"] = {
            "path": "results/cross_source.json",
            "sha256": sha256(cs_path),
            "n_total": cs.get("n_total"),
            "n_overlapping": cs.get("n_overlapping"),
            "n_disjoint_reported": cs.get("n_disjoint"),
            "n_disjoint_from_manifest": len(disjoint),
            "hamming_threshold": cs.get("hamming_threshold"),
            "by_class": cs.get("by_class"),
        }
        if len(disjoint) != VERIFIED_DISJOINT_N:
            fail("disjoint subset size is %d, expected %d" % (len(disjoint), VERIFIED_DISJOINT_N))
        if cs.get("n_disjoint") != len(disjoint):
            fail("cross_source.json n_disjoint (%s) disagrees with its own manifest (%d)"
                 % (cs.get("n_disjoint"), len(disjoint)))
        print("   disjoint subset: %d images (expected %d)" % (len(disjoint), VERIFIED_DISJOINT_N),
              flush=True)

    # --- 4 recompute FusionV2 ---------------------------------------------
    print("4. recomputing FusionV2 metrics from prediction CSVs", flush=True)
    for setting, (rundir, fname) in FUSION_V2_SOURCES.items():
        per_seed = {}
        for seed in SEEDS:
            rel = "%s/fusion_v2_seed%d/%s" % (rundir, seed, fname)
            if rel not in loaded:
                fail("missing FusionV2 prediction file: %s" % rel)
                continue
            d = loaded[rel]
            acc, f1 = accuracy_macro_f1(d["y"], d["pred"], len(d["classes"]))
            per_seed[seed] = {"accuracy": acc, "macro_f1": f1, "n_images": d["n"],
                              "source": rel, "sha256": sha256(os.path.join(root, rel))}
        if len(per_seed) == len(SEEDS):
            report["recomputed"][setting] = {
                "per_seed": per_seed,
                "accuracy_mean_sd": mean_sd([v["accuracy"] for v in per_seed.values()]),
                "macro_f1_mean_sd": mean_sd([v["macro_f1"] for v in per_seed.values()]),
                "n_images": per_seed[SEEDS[0]]["n_images"],
            }

    # duplicate-disjoint external subset, restricted by the manifest
    if disjoint:
        for split, rundir in (("clean", "runs_v2_clean"), ("original", "runs_v2_original")):
            per_seed = {}
            for seed in SEEDS:
                rel = "%s/fusion_v2_seed%d/preds_dataset_ood.csv" % (rundir, seed)
                if rel not in loaded:
                    continue
                d = loaded[rel]
                mask = np.array([f in disjoint for f in d["files"]])
                if int(mask.sum()) != VERIFIED_DISJOINT_N:
                    fail("%s: disjoint mask selects %d rows, expected %d"
                         % (rel, int(mask.sum()), VERIFIED_DISJOINT_N))
                acc, f1 = accuracy_macro_f1(d["y"][mask], d["pred"][mask], len(d["classes"]))
                per_seed[seed] = {"accuracy": acc, "macro_f1": f1,
                                  "n_images": int(mask.sum()), "source": rel}
            if len(per_seed) == len(SEEDS):
                report["recomputed"]["cross_source_disjoint_%s" % split] = {
                    "per_seed": per_seed,
                    "accuracy_mean_sd": mean_sd([v["accuracy"] for v in per_seed.values()]),
                    "macro_f1_mean_sd": mean_sd([v["macro_f1"] for v in per_seed.values()]),
                    "n_images": VERIFIED_DISJOINT_N,
                    "subset_definition": "results/cross_source.json :: manifest kind == disjoint",
                }

    # --- 5 compare against verified constants ------------------------------
    print("5. comparing against CLAUDE.md verified constants", flush=True)
    plan = [("original", "accuracy", "acc"), ("original", "macro_f1", "macro_f1"),
            ("clean", "accuracy", "acc"), ("clean", "macro_f1", "macro_f1")]
    for setting, metric, key in plan:
        rec = report["recomputed"].get(setting)
        if not rec:
            fail("cannot compare %s/%s - recomputation unavailable" % (setting, metric))
            continue
        got = rec["accuracy_mean_sd"] if metric == "accuracy" else rec["macro_f1_mean_sd"]
        c = compare(setting, metric, got, VERIFIED[setting][key])
        report["comparisons"].append(c)
        if not c["pass"]:
            fail("%s %s differs by %.4f pp (mean) / %.4f pp (sd), tolerance %.2f pp"
                 % (setting, metric, c["abs_diff_mean_pp"], c["abs_diff_sd_pp"], TOLERANCE_PP))

    rec = report["recomputed"].get("cross_source_disjoint_clean")
    if rec:
        c = compare("cross_source_disjoint (clean split)", "accuracy",
                    rec["accuracy_mean_sd"], VERIFIED["cross_source_disjoint"]["acc"])
        report["comparisons"].append(c)
        if not c["pass"]:
            fail("cross-source disjoint accuracy differs by %.4f pp (mean) / %.4f pp (sd)"
                 % (c["abs_diff_mean_pp"], c["abs_diff_sd_pp"]))
    else:
        fail("cannot compare duplicate-disjoint accuracy - recomputation unavailable")

    for c in report["comparisons"]:
        print("   [%s] %-38s %-9s recomputed %.2f +/- %.2f | verified %.2f +/- %.2f"
              % ("PASS" if c["pass"] else "FAIL", c["setting"], c["metric"],
                 c["recomputed_mean_pct"], c["recomputed_sd_pct"],
                 c["verified_mean_pct"], c["verified_sd_pct"]), flush=True)

    # --- 6 acceptance checks ----------------------------------------------
    report["checks"] = {
        "A2_manifest_complete": bool(preds_inv) and all("sha256" in e for e in preds_inv + metrics_inv),
        "A3_fusion_v2_reproduces": all(c["pass"] for c in report["comparisons"]) and bool(report["comparisons"]),
        "A4_disjoint_count_373": len(disjoint) == VERIFIED_DISJOINT_N,
        "A5_all_csvs_valid": report["validation"]["n_failed"] == 0 and len(class_orders) <= 1,
    }
    report["acceptance_status"] = "PASS" if (not report["failures"]
                                             and all(report["checks"].values())) else "FAIL"

    # --- 7 write reports ---------------------------------------------------
    man_path = os.path.join(out_root, "baseline_manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md_path = os.path.join(out_root, "BASELINE_AUDIT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    print("\n   wrote %s" % os.path.relpath(man_path, root))
    print("   wrote %s" % os.path.relpath(md_path, root))
    print("\nACCEPTANCE STATUS: %s" % report["acceptance_status"])
    if report["failures"]:
        print("failures (%d):" % len(report["failures"]))
        for m in report["failures"]:
            print("  - %s" % m)
    return 0 if report["acceptance_status"] == "PASS" else 1


def render_markdown(r: dict) -> str:
    L = ["# Phase 0 - Immutable Baseline Audit", "",
         "Generated by `src/publication/audit_baseline.py` at %s." % r["generated_utc"],
         "",
         "**Acceptance status: %s**" % r["acceptance_status"], "",
         "This audit is read-only with respect to every existing artifact. It records "
         "SHA-256 hashes so later regression is detectable, validates every prediction "
         "file, and recomputes the FusionV2 summaries directly from the prediction CSVs "
         "rather than copying them from any table.", ""]

    L += ["## Acceptance checks", "", "| Check | Result |", "|---|---|"]
    labels = {"A2_manifest_complete": "A2 SHA-256 manifest complete",
              "A3_fusion_v2_reproduces": "A3 FusionV2 summaries reproduce within %.2f pp" % r["tolerance_pp"],
              "A4_disjoint_count_373": "A4 duplicate-disjoint subset == 373 images",
              "A5_all_csvs_valid": "A5 all prediction CSVs pass schema and alignment"}
    for k, v in r["checks"].items():
        L.append("| %s | %s |" % (labels.get(k, k), "PASS" if v else "**FAIL**"))
    L.append("")

    inv = r["inventory"]
    L += ["## Inventory", "",
          "- prediction CSVs: **%d**" % inv["n_prediction_csvs"],
          "- metric/JSON files: **%d**" % inv["n_metric_files"],
          "- directories present: `%s`" % "`, `".join(inv["directories_present"])]
    if inv["directories_absent"]:
        L.append("- directories absent: `%s`" % "`, `".join(inv["directories_absent"]))
    L += ["", "Full per-file paths, byte sizes, modification times, row counts and "
              "SHA-256 hashes are in `baseline_manifest.json`.", ""]

    v = r["validation"]
    L += ["## Validation", "",
          "%d prediction files validated, %d failed." % (v["n_ok"], v["n_failed"]),
          "", "Each file was checked for: required columns, probability columns in "
              "sorted class order, unique filenames, labels drawn from the declared "
              "classes, finite probabilities within [0,1], rows summing to 1 within "
              "%g, and `pred_label` equal to the argmax of the stored probabilities "
              "(ties excepted)." % PROB_SUM_TOL, ""]
    if v["distinct_class_orders"]:
        L += ["Class order observed: `%s`" % ", ".join(v["distinct_class_orders"][0]), ""]
    if v["n_failed"]:
        L += ["| File | Error |", "|---|---|"]
        L += ["| `%s` | %s |" % (f["path"], f.get("error", "")) for f in v["files"]
              if f["status"] != "ok"]
        L.append("")

    if "cross_source" in r:
        cs = r["cross_source"]
        L += ["## Duplicate-disjoint cross-source subset", "",
              "| Quantity | Value |", "|---|---|",
              "| total external images | %s |" % cs["n_total"],
              "| overlapping (re-encoded) | %s |" % cs["n_overlapping"],
              "| **disjoint (genuinely unseen)** | **%s** |" % cs["n_disjoint_from_manifest"],
              "| dHash Hamming threshold | %s |" % cs["hamming_threshold"], ""]
        if cs.get("by_class"):
            L += ["Per class (overlapping / disjoint):", ""]
            L += ["- %s: %s / %s" % (k, val.get("overlapping"), val.get("disjoint"))
                  for k, val in cs["by_class"].items()]
            L.append("")

    L += ["## Recomputed FusionV2 results", "",
          "Recomputed from the prediction CSVs listed in the manifest. "
          "Mean +/- s.d. over seeds %s; the s.d. is variation across training "
          "seeds, not a bootstrap interval over images." % ", ".join(map(str, SEEDS)), "",
          "| Setting | Images | Accuracy (%) | Macro-F1 (%) |", "|---|---:|---:|---:|"]
    for name, rec in r["recomputed"].items():
        am, asd = rec["accuracy_mean_sd"]
        fm, fsd = rec["macro_f1_mean_sd"]
        L.append("| %s | %d | %.2f +/- %.2f | %.2f +/- %.2f |"
                 % (name, rec["n_images"], 100 * am, 100 * asd, 100 * fm, 100 * fsd))
    L.append("")

    L += ["## Comparison against verified constants", "",
          "Source of the verified values: `CLAUDE.md` section *Scientific ground truth*. "
          "Tolerance %.2f percentage points on both mean and s.d." % r["tolerance_pp"], "",
          "| Setting | Metric | Recomputed | Verified | diff (mean) | diff (s.d.) | Result |",
          "|---|---|---:|---:|---:|---:|---|"]
    for c in r["comparisons"]:
        L.append("| %s | %s | %.2f +/- %.2f | %.2f +/- %.2f | %.4f | %.4f | %s |"
                 % (c["setting"], c["metric"], c["recomputed_mean_pct"], c["recomputed_sd_pct"],
                    c["verified_mean_pct"], c["verified_sd_pct"],
                    c["abs_diff_mean_pp"], c["abs_diff_sd_pp"],
                    "PASS" if c["pass"] else "**FAIL**"))
    L.append("")

    if r["failures"]:
        L += ["## Failures", "",
              "Recorded, not corrected. Each must be resolved before Phase 1.", ""]
        L += ["%d. %s" % (i, m) for i, m in enumerate(r["failures"], 1)]
        L.append("")

    L += ["## Scope and limitations", "",
          "- This audit establishes internal consistency and reproducibility of stored "
          "predictions. It does not establish patient-level independence, freedom from "
          "all leakage, or external validity.",
          "- The s.d. reported here is across three training seeds only. Bootstrap "
          "confidence intervals over images are a separate quantity and are produced in "
          "Phase 1, not here.",
          "- No comparison against BRAIN-META is made: its per-image predictions are not "
          "available in this repository.",
          "- Files were hashed as found. The audit cannot attest to anything that "
          "happened before it first ran.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
