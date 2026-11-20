"""Controlled Phase 3 addendum: ablations on 373 duplicate-disjoint images.

No model is trained or selected here. Existing 499-image prediction CSVs are
filtered by the independently documented cross-source manifest, then compared
within seed by exact image identity. External labels are used only to evaluate
the frozen outputs, never for tuning, weighting, thresholding, or selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_fusion_v2 import compute_metrics
from final_analysis import holm, mcnemar_exact
from session_artifact_contract import CLASSES, EXTERNAL_N, PROB_SUM_TOL

SEEDS = (42, 43, 44)
REFERENCE = "full_fusion_v2"
VARIANTS = ("no_gate", "no_multiscale", "no_auxiliary_loss", "no_ema",
            "cnn_branch_only", "vit_branch_only", "no_tta")
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260925
EXPECTED_DISJOINT = 373


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_manifest(root: Path) -> tuple[dict[str, str], dict]:
    path = root / "results/cross_source.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("n_disjoint") != EXPECTED_DISJOINT:
        raise AnalysisError(f"Manifest says {payload.get('n_disjoint')} disjoint, expected 373")
    mapping = {name: row["class"] for name, row in payload["manifest"].items()
               if row.get("kind") == "disjoint"}
    if len(mapping) != EXPECTED_DISJOINT:
        raise AnalysisError(f"Manifest yields {len(mapping)} disjoint identities, expected 373")
    unknown = sorted(set(mapping.values()) - set(CLASSES))
    if unknown:
        raise AnalysisError(f"Manifest contains unknown labels: {unknown}")
    return mapping, {"path": rel(path, root), "sha256": sha256(path),
                     "n": len(mapping), "class_counts": dict(Counter(mapping.values()))}


def prediction_path(root: Path, variant: str, seed: int) -> Path:
    if variant == REFERENCE:
        return root / f"runs_v2_clean/fusion_v2_seed{seed}/preds_dataset_ood.csv"
    return root / f"publication_runs/ablations/clean/{variant}/seed_{seed}/preds_dataset_ood.csv"


def load_disjoint(path: Path, mapping: dict[str, str], root: Path) -> dict:
    if not path.is_file():
        raise AnalysisError(f"Missing prediction file: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXTERNAL_N:
        raise AnalysisError(f"{path}: {len(rows)} rows, expected {EXTERNAL_N}")
    prob_cols = [c for c in rows[0] if c.startswith("p_")]
    classes = [c[2:] for c in prob_cols]
    if classes != list(CLASSES):
        raise AnalysisError(f"{path}: class order {classes} != {list(CLASSES)}")
    by_name = {}
    for row in rows:
        name = row["filename"]
        if name in by_name:
            raise AnalysisError(f"{path}: duplicate filename {name}")
        by_name[name] = row
    missing = sorted(set(mapping) - set(by_name))
    if missing:
        raise AnalysisError(f"{path}: missing {len(missing)} disjoint identities: {missing[:3]}")
    names = sorted(mapping)
    selected = [by_name[name] for name in names]
    bad_labels = [name for name, row in zip(names, selected)
                  if row["true_label"] != mapping[name]]
    if bad_labels:
        raise AnalysisError(f"{path}: label mismatch for {len(bad_labels)} images")
    P = np.asarray([[float(row[col]) for col in prob_cols] for row in selected], np.float64)
    if not np.isfinite(P).all() or P.min() < 0 or P.max() > 1:
        raise AnalysisError(f"{path}: invalid probabilities")
    max_dev = float(np.abs(P.sum(axis=1) - 1).max())
    if max_dev > PROB_SUM_TOL:
        raise AnalysisError(f"{path}: probability-sum deviation {max_dev:.3e}")
    pred = P.argmax(axis=1)
    stored = np.asarray([classes.index(row["pred_label"]) for row in selected])
    if not np.array_equal(pred, stored):
        raise AnalysisError(f"{path}: pred_label does not equal argmax")
    y = np.asarray([classes.index(mapping[name]) for name in names])
    return {"path": rel(path, root), "sha256": sha256(path), "names": names,
            "y": y, "P": P, "pred": pred, "classes": classes,
            "max_prob_sum_deviation": max_dev, "metrics": compute_metrics(y, P, classes)}


def paired_counts(reference: dict, ablation: dict) -> dict:
    if reference["names"] != ablation["names"] or not np.array_equal(reference["y"], ablation["y"]):
        raise AnalysisError("Pairing identity or labels differ")
    y = reference["y"]
    r_ok, a_ok = reference["pred"] == y, ablation["pred"] == y
    return {"both_correct": int((r_ok & a_ok).sum()),
            "reference_correct_ablation_wrong": int((r_ok & ~a_ok).sum()),
            "reference_wrong_ablation_correct": int((~r_ok & a_ok).sum()),
            "both_wrong": int((~r_ok & ~a_ok).sum())}


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for cls in range(len(CLASSES)):
        tp = int(((pred == cls) & (y == cls)).sum())
        fp = int(((pred == cls) & (y != cls)).sum())
        fn = int(((pred != cls) & (y == cls)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def paired_bootstrap(reference: dict, ablation: dict, seed: int) -> dict:
    y, rp, ap = reference["y"], reference["pred"], ablation["pred"]
    groups = [np.flatnonzero(y == k) for k in range(len(CLASSES))]
    rng = np.random.default_rng(seed)
    d_acc, d_f1 = np.empty(BOOTSTRAP_ITERATIONS), np.empty(BOOTSTRAP_ITERATIONS)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        take = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        yy, rr, aa = y[take], rp[take], ap[take]
        d_acc[iteration] = (aa == yy).mean() - (rr == yy).mean()
        d_f1[iteration] = macro_f1(yy, aa) - macro_f1(yy, rr)
    def summarize(values: np.ndarray, point: float) -> dict:
        lo, hi = np.percentile(values, [2.5, 97.5]) * 100
        return {"point_pp": point * 100, "ci95_lo_pp": float(lo),
                "ci95_hi_pp": float(hi), "excludes_zero": bool(lo * hi > 0)}
    return {"iterations": BOOTSTRAP_ITERATIONS, "seed": seed,
            "unit": "image", "paired": True, "stratified_by": "true class",
            "accuracy": summarize(d_acc, (ap == y).mean() - (rp == y).mean()),
            "macro_f1": summarize(d_f1, macro_f1(y, ap) - macro_f1(y, rp))}


def mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise AnalysisError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def make_figures(out: Path, summaries: list[dict], per_class: list[dict]) -> None:
    labels = [row["variant"] for row in summaries]
    means = [row["accuracy_mean_pct"] for row in summaries]
    sds = [row["accuracy_sd_pct"] for row in summaries]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = np.arange(len(labels))
    ax.barh(positions, means, xerr=sds, color="#2878B5", alpha=.86, capsize=3)
    ax.set_yticks(positions, labels=labels); ax.invert_yaxis()
    ax.set_xlabel("Accuracy on 373 duplicate-disjoint images (%)")
    ax.set_title("Cross-source performance of completed ablations\nerror bars: seed SD, n=3")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"cross_source_ablation_accuracy.{ext}", dpi=220)
    plt.close(fig)

    pituitary = [row for row in per_class if row["class"] == "pituitary"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(np.arange(len(pituitary)), [r["recall_mean_pct"] for r in pituitary],
            xerr=[r["recall_sd_pct"] for r in pituitary], color="#C44E52", alpha=.86, capsize=3)
    ax.set_yticks(np.arange(len(pituitary)), labels=[r["variant"] for r in pituitary]); ax.invert_yaxis()
    ax.set_xlabel("Pituitary recall on duplicate-disjoint images (%)")
    ax.set_title("Class-specific source-shift failure\nerror bars: seed SD, n=3")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pituitary_recall_by_ablation.{ext}", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); root = args.project_root.resolve(); out = args.out_root.resolve()
    final_json = out / "cross_source_ablation_analysis.json"
    if final_json.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {final_json}; use --resume after review")
    out.mkdir(parents=True, exist_ok=True); figures = out / "figures"; figures.mkdir(exist_ok=True)
    mapping, manifest_meta = load_manifest(root)
    variants = (REFERENCE,) + VARIANTS
    runs, input_rows = {}, []
    for variant in variants:
        runs[variant] = {}
        for seed in SEEDS:
            loaded = load_disjoint(prediction_path(root, variant, seed), mapping, root)
            runs[variant][seed] = loaded
            input_rows.append({"variant": variant, "seed": seed, "path": loaded["path"],
                               "sha256": loaded["sha256"], "n": len(loaded["names"]),
                               "max_prob_sum_deviation": loaded["max_prob_sum_deviation"]})
    canonical_names = runs[REFERENCE][42]["names"]
    canonical_y = runs[REFERENCE][42]["y"]
    for variant in variants:
        for seed in SEEDS:
            if runs[variant][seed]["names"] != canonical_names or not np.array_equal(runs[variant][seed]["y"], canonical_y):
                raise AnalysisError(f"Identity/label alignment failed: {variant} seed {seed}")

    metric_rows, per_class_rows, summaries = [], [], []
    for variant in variants:
        for seed in SEEDS:
            metrics = runs[variant][seed]["metrics"]
            metric_rows.append({"variant": variant, "seed": seed, "n_images": EXPECTED_DISJOINT,
                                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                                "balanced_accuracy": metrics["balanced_accuracy"],
                                "nll": metrics["nll"], "brier": metrics["brier"], "ece": metrics["ece"]})
            for row in metrics["per_class"]:
                per_class_rows.append({"variant": variant, "seed": seed, "class": row["class"],
                                       "support": row["support"], "precision": row["precision"],
                                       "recall": row["recall_sensitivity"], "specificity": row["specificity"],
                                       "f1": row["f1"]})
        acc = [runs[variant][s]["metrics"]["accuracy"] * 100 for s in SEEDS]
        f1 = [runs[variant][s]["metrics"]["macro_f1"] * 100 for s in SEEDS]
        acc_m, acc_sd = mean_sd(acc); f1_m, f1_sd = mean_sd(f1)
        summaries.append({"variant": variant, "accuracy_mean_pct": acc_m,
                          "accuracy_sd_pct": acc_sd, "macro_f1_mean_pct": f1_m,
                          "macro_f1_sd_pct": f1_sd})

    tests, p_values = [], []
    for variant_index, variant in enumerate(VARIANTS):
        for seed_index, seed in enumerate(SEEDS):
            ref, abl = runs[REFERENCE][seed], runs[variant][seed]
            counts = paired_counts(ref, abl)
            test = mcnemar_exact(counts["reference_correct_ablation_wrong"],
                                  counts["reference_wrong_ablation_correct"])
            boot = paired_bootstrap(ref, abl, BOOTSTRAP_SEED + variant_index * 10 + seed_index)
            row = {"variant": variant, "seed": seed, **counts,
                   "p_unadjusted": test["p_value"], "bootstrap": boot}
            tests.append(row); p_values.append(test["p_value"])
    if len(tests) != 21:
        raise AnalysisError(f"Expected 21 paired tests, got {len(tests)}")
    adjusted = holm(p_values)
    test_rows = []
    for row, adj in zip(tests, adjusted):
        row["p_holm"] = adj; row["significant_holm_0_05"] = bool(adj < .05)
        test_rows.append({"variant": row["variant"], "seed": row["seed"],
                          "both_correct": row["both_correct"],
                          "ref_correct_ablation_wrong": row["reference_correct_ablation_wrong"],
                          "ref_wrong_ablation_correct": row["reference_wrong_ablation_correct"],
                          "both_wrong": row["both_wrong"], "p_unadjusted": row["p_unadjusted"],
                          "p_holm": row["p_holm"], "significant_holm_0_05": row["significant_holm_0_05"],
                          "delta_accuracy_pp": row["bootstrap"]["accuracy"]["point_pp"],
                          "accuracy_ci95_lo_pp": row["bootstrap"]["accuracy"]["ci95_lo_pp"],
                          "accuracy_ci95_hi_pp": row["bootstrap"]["accuracy"]["ci95_hi_pp"]})

    class_summaries = []
    for variant in variants:
        for class_name in CLASSES:
            selected = [r for r in per_class_rows if r["variant"] == variant and r["class"] == class_name]
            recall_m, recall_sd = mean_sd([r["recall"] * 100 for r in selected])
            f1_m, f1_sd = mean_sd([r["f1"] * 100 for r in selected])
            class_summaries.append({"variant": variant, "class": class_name,
                                    "support": selected[0]["support"],
                                    "recall_mean_pct": recall_m, "recall_sd_pct": recall_sd,
                                    "f1_mean_pct": f1_m, "f1_sd_pct": f1_sd})

    ref_summary = next(r for r in summaries if r["variant"] == REFERENCE)
    reproduces = abs(ref_summary["accuracy_mean_pct"] - 69.53) <= .01 and abs(ref_summary["accuracy_sd_pct"] - 1.98) <= .01
    if not reproduces:
        raise AnalysisError(f"Reference failed reproduction: {ref_summary}")

    write_csv(out / "metrics_by_seed.csv", metric_rows)
    write_csv(out / "paired_tests.csv", test_rows)
    write_csv(out / "per_class_metrics.csv", per_class_rows)
    make_figures(figures, summaries, class_summaries)
    payload = {"phase": "Phase 3 addendum - duplicate-disjoint cross-source ablations",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "manifest": manifest_meta, "inputs": input_rows,
               "summaries": summaries, "per_class_summaries": class_summaries,
               "paired_tests": tests,
               "multiple_comparison": {"family": "7 variants x 3 matching seeds", "n": 21,
                                       "method": "Holm-Bonferroni", "alpha": .05,
                                       "significant": sum(r["significant_holm_0_05"] for r in tests)},
               "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "unit": "image",
                             "paired": True, "stratified_by": "true class"},
               "external_labels_used_for": "evaluation only; no tuning, selection, weighting, or thresholding",
               "reference_reproduces": reproduces, "acceptance_status": "PASS",
               "limitations": ["373 images from one external collection",
                               "image-level, not patient-level, uncertainty",
                               "three training seeds",
                               "analysis does not establish multi-institutional generalization"]}
    final_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Duplicate-disjoint cross-source ablation analysis", "",
             "**Status: PASS.** Existing predictions only; no training, tuning, selection, or thresholding.", "",
             "## Performance on 373 duplicate-disjoint images", "",
             "| Variant | Accuracy mean ± seed SD (%) | Macro-F1 mean ± seed SD (%) |",
             "|---|---:|---:|"]
    for row in summaries:
        lines.append(f"| `{row['variant']}` | {row['accuracy_mean_pct']:.2f} ± {row['accuracy_sd_pct']:.2f} | "
                     f"{row['macro_f1_mean_pct']:.2f} ± {row['macro_f1_sd_pct']:.2f} |")
    lines += ["", "## Statistical scope", "",
              f"Exactly 21 seed-matched McNemar tests were Holm-corrected together; "
              f"{sum(r['significant_holm_0_05'] for r in tests)} remained significant at 0.05. "
              "Paired image-level bootstrap intervals use 10,000 class-stratified resamples per comparison. "
              "Repeated predictions of the same 373 images are never treated as independent cases.", "",
              "## Interpretation boundaries", "",
              "- These results evaluate frozen outputs on one duplicate-disjoint external collection.",
              "- External labels were not used to choose a component or tune any setting.",
              "- Seed SD is not a confidence interval; bootstrap intervals are image-level, not patient-level.",
              "- A nonsignificant difference is not equivalence or proof that a component is unnecessary."]
    (out / "CROSS_SOURCE_ABLATION_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: {final_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

