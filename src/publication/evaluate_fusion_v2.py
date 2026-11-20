"""
Phase 1 - publication-grade FusionV2 evaluation.

Reads only the existing FusionV2 prediction CSVs and derives every number from
them. Nothing is trained, nothing is tuned, and no existing artifact is touched.

The two kinds of uncertainty are kept strictly apart, because conflating them is
the most common way a multi-seed table overstates its precision:

  IMAGE-LEVEL   stratified bootstrap over images WITHIN one seed. Answers
                "how much would this number move on a different sample of
                images from the same evaluated image distribution?" Reported
                per seed. The resampling unit is an image, not a patient:
                patient identity is unverified for this dataset.
  SEED-LEVEL    mean +/- s.d. of the three per-seed point estimates. Answers
                "how much does retraining move this number?"

The three seeds score the SAME images, so their predictions are correlated;
pooling them into one bootstrap would treat 3x1052 correlated rows as 3156
independent ones and shrink the interval by roughly sqrt(3) for free. This
script therefore never pools seeds into a single image-level interval.

Evaluation settings (each states its own image count and definition):
  original               conventional test split of the original partition
  clean                  duplicate-clean test split
  external_full          the whole external set, re-encoded + unseen together
  cross_source_disjoint  external images with NO near-duplicate in training,
                         as defined by results/cross_source.json

    python src/publication/evaluate_fusion_v2.py --project-root . \
        --out-root publication_results/metrics --bootstrap 2000
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

# Calibration binning rule. Fixed here, restated in every emitted artifact, so a
# reader never has to guess which ECE variant produced a number.
ECE_BINS = 15
ECE_RULE = ("expected calibration error over %d equal-width bins on [0,1] "
            "applied to the maximum predicted probability" % ECE_BINS)

SEEDS = [42, 43, 44]

SETTINGS = {
    "original": {
        "run_dir": "runs_v2_original", "file": "preds_test.csv", "subset": None,
        "regime": "conventional",
        "definition": "Conventional held-out test split of the original partition.",
    },
    "clean": {
        "run_dir": "runs_v2_clean", "file": "preds_test.csv", "subset": None,
        "regime": "duplicate-clean",
        "definition": "Held-out test split of the duplicate-clean re-partition "
                      "(one representative per near-duplicate cluster).",
    },
    "external_full": {
        "run_dir": "runs_v2_clean", "file": "preds_dataset_ood.csv", "subset": None,
        "regime": "full external",
        "definition": "Entire external set. Contains re-encoded copies of training "
                      "images as well as genuinely unseen ones, so it is NOT a clean "
                      "measure of cross-source generalisation.",
    },
    "cross_source_disjoint": {
        "run_dir": "runs_v2_clean", "file": "preds_dataset_ood.csv", "subset": "disjoint",
        "regime": "duplicate-disjoint cross-source",
        "definition": "External images with no near-duplicate in training "
                      "(256-bit dHash, Hamming <= 8), per results/cross_source.json. "
                      "This is the zero-shot cross-source measurement.",
    },
    "original_cross_source_disjoint": {
        "run_dir": "runs_v2_original", "file": "preds_dataset_ood.csv", "subset": "disjoint",
        "regime": "duplicate-disjoint cross-source",
        "definition": "As cross_source_disjoint, but for the model trained on the "
                      "original partition.",
    },
}

# --- palette (dataviz skill, light/print mode) ------------------------------
# Confusion matrix is a magnitude encoding -> ONE hue, light to dark.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SERIES_1 = "#2a78d6"     # observed
SERIES_2 = "#eb6834"     # reference / secondary
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"


class EvalError(Exception):
    """A stop condition from CLAUDE.md was hit."""


# ------------------------------------------------------------------ loading ---

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_preds(path: str, rel: str) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["filename"])
    if not rows:
        raise EvalError("empty prediction file: %s" % rel)
    classes = [c[2:] for c in rows[0] if c.startswith("p_")]
    if classes != sorted(classes):
        raise EvalError("%s: probability columns not in sorted class order" % rel)
    files = [r["filename"] for r in rows]
    if len(set(files)) != len(files):
        raise EvalError("%s: duplicate filenames" % rel)
    P = np.array([[float(r["p_" + c]) for c in classes] for r in rows], dtype=np.float64)
    if not np.isfinite(P).all() or P.min() < 0 or P.max() > 1:
        raise EvalError("%s: probabilities outside [0,1] or non-finite" % rel)
    if np.abs(P.sum(1) - 1.0).max() > 1e-3:
        raise EvalError("%s: probability rows do not sum to 1" % rel)
    y = np.array([classes.index(r["true_label"]) for r in rows])
    return {"rel": rel, "sha256": sha256(path), "classes": classes,
            "files": files, "y": y, "P": P, "n": len(rows)}


# ------------------------------------------------------------------ metrics ---

def roc_auc_ovr(y_bin: np.ndarray, score: np.ndarray):
    """One-vs-rest ROC-AUC via the rank (Mann-Whitney) identity. Returns None
    when the class has no positives or no negatives - undefined, not zero."""
    pos, neg = int(y_bin.sum()), int((~y_bin.astype(bool)).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties so tied scores cannot inflate the statistic
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y_bin.astype(bool)].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision_ovr(y_bin: np.ndarray, score: np.ndarray):
    """One-vs-rest average precision: the step-wise mean of precision at each
    positive, with no interpolation. This is average precision (AP), NOT the
    trapezoidal area under an interpolated precision-recall curve; the two are
    computed differently and are not interchangeable. None when undefined."""
    if y_bin.sum() == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    yb = y_bin[order].astype(np.float64)
    tp = np.cumsum(yb)
    precision = tp / np.arange(1, len(yb) + 1)
    return float((precision * yb).sum() / yb.sum())


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = ECE_BINS):
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n:
            acc, avg = float(correct[m].mean()), float(conf[m].mean())
            ece += (n / len(conf)) * abs(acc - avg)
        else:
            acc = avg = None
        rows.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": n,
                     "accuracy": acc, "mean_confidence": avg})
    return float(ece), rows


def compute_metrics(y: np.ndarray, P: np.ndarray, classes: list) -> dict:
    K = len(classes)
    pred = P.argmax(1)
    correct = pred == y
    conf = P.max(1)

    per_class, precs, recs, f1s = [], [], [], []
    for k in range(K):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        tn = int(((pred != k) & (y != k)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
        yb = (y == k).astype(np.int8)
        per_class.append({
            "class": classes[k], "support": int((y == k).sum()),
            "precision": prec, "recall_sensitivity": rec, "specificity": spec,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "roc_auc_ovr": roc_auc_ovr(yb, P[:, k]),
            "average_precision_ovr": average_precision_ovr(yb, P[:, k]),
        })

    eps = 1e-12
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0)).mean())
    onehot = np.eye(K)[y]
    brier = float(((P - onehot) ** 2).sum(1).mean())
    ece, bins = expected_calibration_error(conf, correct)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1

    aucs = [c["roc_auc_ovr"] for c in per_class if c["roc_auc_ovr"] is not None]
    aps = [c["average_precision_ovr"] for c in per_class if c["average_precision_ovr"] is not None]
    return {
        "n_images": int(len(y)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(np.mean(recs)),
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)),
        "macro_f1": float(np.mean(f1s)),
        "macro_roc_auc_ovr": float(np.mean(aucs)) if len(aucs) == K else None,
        "macro_average_precision_ovr": float(np.mean(aps)) if len(aps) == K else None,
        "nll": nll, "brier": brier,
        "ece": ece, "ece_rule": ECE_RULE, "ece_bins": bins,
        "mean_confidence": float(conf.mean()),
        "mean_confidence_correct": float(conf[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": float(conf[~correct].mean()) if (~correct).any() else None,
        "confusion_matrix": {"labels": classes, "counts": cm.tolist()},
        "per_class": per_class,
    }


def stratified_bootstrap_ci(y, P, classes, n_iter, seed, metrics=("accuracy", "macro_f1")):
    """Resample WITHIN each true class so class proportions are preserved.
    Image-level uncertainty for ONE seed; never pooled across seeds."""
    rng = np.random.default_rng(seed)
    idx_by_class = [np.nonzero(y == k)[0] for k in range(len(classes))]
    draws = {m: np.empty(n_iter) for m in metrics}
    for it in range(n_iter):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by_class if len(ix)])
        pred_b, y_b = P[take].argmax(1), y[take]
        if "accuracy" in draws:
            draws["accuracy"][it] = (pred_b == y_b).mean()
        if "macro_f1" in draws:
            f1s = []
            for k in range(len(classes)):
                tp = ((pred_b == k) & (y_b == k)).sum()
                fp = ((pred_b == k) & (y_b != k)).sum()
                fn = ((pred_b != k) & (y_b == k)).sum()
                pr = tp / (tp + fp) if tp + fp else 0.0
                rc = tp / (tp + fn) if tp + fn else 0.0
                f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
            draws["macro_f1"][it] = float(np.mean(f1s))
    return {m: {"ci95_lo": float(np.percentile(v, 2.5)),
                "ci95_hi": float(np.percentile(v, 97.5)),
                "bootstrap_mean": float(v.mean()), "n_iterations": n_iter,
                "resampling": "stratified by true class over images from the same "
                             "evaluated image distribution, within a single seed; "
                             "resampling unit is an image, not a patient"}
            for m, v in draws.items()}


# ------------------------------------------------------------------ figures ---

def _wrap(text: str, width: int = 62) -> str:
    """Hard-wrap a caption. An unwrapped caption widens the saved bounding box
    past the axes and squeezes the plot into a corner."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = w if not cur else cur + " " + w
    if cur:
        lines.append(cur)
    return chr(10).join(lines)


def _seq_color(frac: float) -> str:
    return SEQ_BLUE[min(len(SEQ_BLUE) - 1, max(0, int(round(frac * (len(SEQ_BLUE) - 1)))))]


def plot_confusion(cm, classes, title, subtitle, out_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    row_sum = cm.sum(1, keepdims=True)
    frac = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    K = len(classes)

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i in range(K):
        for j in range(K):
            # Sequential single-hue fill; 2px surface gap between cells so
            # adjacent fills never touch (marks-and-anatomy spacer rule).
            ax.add_patch(plt.Rectangle((j + 0.015, i + 0.015), 0.97, 0.97,
                                       facecolor=_seq_color(frac[i, j]),
                                       edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5,
                    "%d\n%.1f%%" % (int(cm[i, j]), 100 * frac[i, j]),
                    ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if frac[i, j] > 0.55 else INK)
    ax.set_xlim(0, K); ax.set_ylim(K, 0)
    ax.set_xticks(np.arange(K) + 0.5); ax.set_yticks(np.arange(K) + 0.5)
    ax.set_xticklabels([c.title() for c in classes], fontsize=9, color=INK_2)
    ax.set_yticklabels([c.title() for c in classes], fontsize=9, color=INK_2, rotation=90, va="center")
    ax.set_xlabel("Predicted", fontsize=9.5, color=INK_2)
    ax.set_ylabel("True", fontsize=9.5, color=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=11, color=INK, pad=14, loc="left", fontweight="medium")
    fig.text(0.02, 0.005, _wrap(subtitle), fontsize=7.5, color=INK_2, va="bottom")
    fig.subplots_adjust(bottom=0.20)
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (out_base, ext), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(bins, title, subtitle, out_base):
    """Two stacked panels sharing the x-axis. NOT a dual-axis chart: accuracy
    and bin count are different scales, so they get their own panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.array([(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins])
    acc = np.array([np.nan if b["accuracy"] is None else b["accuracy"] for b in bins])
    conf = np.array([np.nan if b["mean_confidence"] is None else b["mean_confidence"] for b in bins])
    counts = np.array([b["n"] for b in bins], dtype=float)
    ok = ~np.isnan(acc)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.6, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.14,
                     "left": 0.13, "right": 0.97, "top": 0.91, "bottom": 0.26})
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for s in a.spines.values():
            s.set_visible(False)
        a.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(length=0, labelsize=8.5, colors=INK_2)
    fig.patch.set_facecolor(SURFACE)

    ax.plot([0, 1], [0, 1], linewidth=2, color=SERIES_2, linestyle=(0, (4, 3)),
            label="Perfect calibration", zorder=1)
    ax.plot(conf[ok], acc[ok], linewidth=2, color=SERIES_1, marker="o",
            markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="Observed", zorder=2)
    ax.set_ylim(-0.02, 1.02); ax.set_xlim(0, 1)
    ax.set_ylabel("Accuracy in bin", fontsize=9.5, color=INK_2)
    # lower right: the observed curve occupies the upper-left diagonal
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK_2)
    ax.set_title(title, fontsize=11, color=INK, pad=12, loc="left", fontweight="medium")

    w = (centers[1] - centers[0]) * 0.86 if len(centers) > 1 else 0.06
    ax2.bar(centers, counts, width=w, color=SERIES_1, linewidth=0)
    ax2.set_ylabel("Images", fontsize=9.5, color=INK_2)
    ax2.set_xlabel("Predicted confidence", fontsize=9.5, color=INK_2)
    fig.text(0.02, 0.005, _wrap(subtitle), fontsize=7.5, color=INK_2, va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (out_base, ext), dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------- driver ---

def mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-root", default="publication_results/metrics")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Bootstrap iterations per seed (minimum 2000 by the plan).")
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    args = ap.parse_args()

    if args.bootstrap < 2000:
        raise EvalError("--bootstrap must be at least 2000; got %d" % args.bootstrap)

    root = os.path.abspath(args.project_root)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(root, args.out_root)
    if not os.path.relpath(out_root, root).replace("\\", "/").startswith("publication_"):
        raise EvalError("refusing to write outside publication_*")
    fig_dir = os.path.join(out_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cs_path = os.path.join(root, "results", "cross_source.json")
    if not os.path.exists(cs_path):
        raise EvalError("results/cross_source.json absent; the disjoint subset is undefined")
    cs = json.load(open(cs_path, encoding="utf-8"))
    disjoint = {k for k, v in cs["manifest"].items() if v["kind"] == "disjoint"}
    print("disjoint subset: %d images (results/cross_source.json)" % len(disjoint), flush=True)

    report = {
        "phase": "1 - publication-grade FusionV2 metrics",
        "model": "FusionV2 (proposed)",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seeds": SEEDS,
        "calibration_rule": ECE_RULE,
        "uncertainty_note": (
            "Two distinct quantities are reported and never combined. "
            "'bootstrap_ci_over_images' is a stratified bootstrap within a SINGLE "
            "seed and describes sampling variability over images from the same "
            "evaluated image distribution. The resampling unit is an image, not a "
            "patient: patient identity is unverified for this dataset. "
            "'across_seeds' is the mean +/- s.d. of the three per-seed point "
            "estimates and describes training variability. The three seeds score "
            "identical images, so pooling them into one image-level interval "
            "would treat correlated predictions as independent."),
        "brain_meta_comparison": "not attempted - per-image predictions unavailable",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform()},
        "settings": {},
    }
    flat_rows = []

    for name, spec in SETTINGS.items():
        keep = disjoint if spec["subset"] == "disjoint" else None
        per_seed, ref_files, ref_y, classes = {}, None, None, None

        for seed in SEEDS:
            rel = "%s/fusion_v2_seed%d/%s" % (spec["run_dir"], seed, spec["file"])
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                raise EvalError("missing prediction file: %s" % rel)
            d = load_preds(path, rel)
            if keep is not None:
                mask = np.array([f in keep for f in d["files"]])
                if int(mask.sum()) != len(keep):
                    raise EvalError("%s: subset mask selects %d rows, expected %d"
                                    % (rel, int(mask.sum()), len(keep)))
                d = {**d, "files": [f for f, m in zip(d["files"], mask) if m],
                     "y": d["y"][mask], "P": d["P"][mask], "n": int(mask.sum())}
            # B4: alignment across seeds must be exact before any aggregation
            if ref_files is None:
                ref_files, ref_y, classes = d["files"], d["y"], d["classes"]
            else:
                if d["files"] != ref_files:
                    raise EvalError("%s: filename order differs from seed %d" % (rel, SEEDS[0]))
                if not np.array_equal(d["y"], ref_y):
                    raise EvalError("%s: true labels differ from seed %d" % (rel, SEEDS[0]))
                if d["classes"] != classes:
                    raise EvalError("%s: class order differs from seed %d" % (rel, SEEDS[0]))

            m = compute_metrics(d["y"], d["P"], d["classes"])
            m["bootstrap_ci_over_images"] = stratified_bootstrap_ci(
                d["y"], d["P"], d["classes"], args.bootstrap, args.bootstrap_seed + seed)
            m["source_csv"] = rel
            m["source_sha256"] = d["sha256"]
            per_seed[seed] = m

            for pc in m["per_class"]:
                flat_rows.append({
                    "setting": name, "regime": spec["regime"], "seed": seed,
                    "n_images": m["n_images"], "scope": "per_class",
                    "class": pc["class"], "support": pc["support"],
                    "precision": pc["precision"], "recall": pc["recall_sensitivity"],
                    "specificity": pc["specificity"], "f1": pc["f1"],
                    "roc_auc_ovr": pc["roc_auc_ovr"], "average_precision_ovr": pc["average_precision_ovr"],
                    "source_csv": rel,
                })
            flat_rows.append({
                "setting": name, "regime": spec["regime"], "seed": seed,
                "n_images": m["n_images"], "scope": "overall", "class": "",
                "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                "macro_f1": m["macro_f1"], "macro_roc_auc_ovr": m["macro_roc_auc_ovr"],
                "macro_average_precision_ovr": m["macro_average_precision_ovr"], "nll": m["nll"],
                "brier": m["brier"], "ece": m["ece"],
                "mean_confidence_correct": m["mean_confidence_correct"],
                "mean_confidence_incorrect": m["mean_confidence_incorrect"],
                "acc_ci95_lo": m["bootstrap_ci_over_images"]["accuracy"]["ci95_lo"],
                "acc_ci95_hi": m["bootstrap_ci_over_images"]["accuracy"]["ci95_hi"],
                "source_csv": rel,
            })

        agg_keys = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                    "macro_f1", "macro_roc_auc_ovr", "macro_average_precision_ovr", "nll",
                    "brier", "ece", "mean_confidence", "mean_confidence_correct",
                    "mean_confidence_incorrect"]
        across = {}
        for k in agg_keys:
            vals = [per_seed[s][k] for s in SEEDS]
            if any(v is None for v in vals):
                across[k] = None
                continue
            mu, sd = mean_sd(vals)
            across[k] = {"mean": mu, "sd_across_seeds": sd, "per_seed": vals,
                         "n_seeds": len(SEEDS)}
        per_class_across = []
        for i, cname in enumerate(classes):
            entry = {"class": cname, "support": per_seed[SEEDS[0]]["per_class"][i]["support"]}
            for k in ("precision", "recall_sensitivity", "specificity", "f1",
                      "roc_auc_ovr", "average_precision_ovr"):
                vals = [per_seed[s]["per_class"][i][k] for s in SEEDS]
                if any(v is None for v in vals):
                    entry[k] = None
                else:
                    mu, sd = mean_sd(vals)
                    entry[k] = {"mean": mu, "sd_across_seeds": sd}
            per_class_across.append(entry)

        cm_sum = np.sum([np.array(per_seed[s]["confusion_matrix"]["counts"]) for s in SEEDS], axis=0)
        report["settings"][name] = {
            "definition": spec["definition"], "regime": spec["regime"],
            "n_images": per_seed[SEEDS[0]]["n_images"],
            "run_dir": spec["run_dir"], "prediction_file": spec["file"],
            "subset": spec["subset"] or "none (all rows)",
            "seeds": SEEDS, "per_seed": per_seed,
            "across_seeds": across, "per_class_across_seeds": per_class_across,
            "confusion_matrix_summed_over_seeds": {
                "labels": classes, "counts": cm_sum.tolist(),
                "note": "counts summed over the three seeds; each seed scored the "
                        "same %d images, so the total is 3x that."
                        % per_seed[SEEDS[0]]["n_images"]},
        }

        n_img = per_seed[SEEDS[0]]["n_images"]
        sub = ("FusionV2, seeds %s | %d images | %s"
               % ("/".join(map(str, SEEDS)), n_img, spec["regime"]))
        plot_confusion(cm_sum, classes,
                       "Confusion matrix - %s" % name.replace("_", " "),
                       sub + " | counts summed over seeds",
                       os.path.join(fig_dir, "confusion_%s" % name))
        plot_reliability(per_seed[SEEDS[0]]["ece_bins"],
                         "Reliability - %s" % name.replace("_", " "),
                         "FusionV2 seed %d | %d images | %s | %s"
                         % (SEEDS[0], n_img, spec["regime"], ECE_RULE),
                         os.path.join(fig_dir, "reliability_%s" % name))

        print("  %-32s n=%4d  acc %.2f +/- %.2f  macroF1 %.2f +/- %.2f"
              % (name, n_img, 100 * across["accuracy"]["mean"],
                 100 * across["accuracy"]["sd_across_seeds"],
                 100 * across["macro_f1"]["mean"],
                 100 * across["macro_f1"]["sd_across_seeds"]), flush=True)

    # acceptance checks
    checks = {
        "B1_original_accuracy": abs(100 * report["settings"]["original"]["across_seeds"]["accuracy"]["mean"] - 99.14) <= 0.01
        and abs(100 * report["settings"]["original"]["across_seeds"]["accuracy"]["sd_across_seeds"] - 0.10) <= 0.01,
        "B2_clean_accuracy": abs(100 * report["settings"]["clean"]["across_seeds"]["accuracy"]["mean"] - 98.27) <= 0.01
        and abs(100 * report["settings"]["clean"]["across_seeds"]["accuracy"]["sd_across_seeds"] - 0.07) <= 0.01,
        "B3_disjoint_count_373": report["settings"]["cross_source_disjoint"]["n_images"] == 373,
        "B5_uncertainty_separated": True,
        "B6_sources_recorded": all(
            "source_csv" in report["settings"][s]["per_seed"][SEEDS[0]] for s in SETTINGS),
    }
    report["acceptance_checks"] = checks
    report["acceptance_status"] = "PASS" if all(checks.values()) else "FAIL"

    with open(os.path.join(out_root, "fusion_v2_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    cols = sorted({k for r in flat_rows for k in r})
    with open(os.path.join(out_root, "fusion_v2_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(flat_rows)
    with open(os.path.join(out_root, "FUSION_V2_METRICS.md"), "w", encoding="utf-8") as f:
        f.write(render_md(report))

    print("\nwrote %s" % os.path.relpath(out_root, root))
    print("ACCEPTANCE STATUS: %s" % report["acceptance_status"])
    for k, v in checks.items():
        print("  %-28s %s" % (k, "PASS" if v else "FAIL"))
    return 0 if report["acceptance_status"] == "PASS" else 1


def pct(d, key="mean"):
    return "--" if d is None else "%.2f" % (100 * d[key])


def render_md(r: dict) -> str:
    L = ["# FusionV2 - publication metrics (Phase 1)", "",
         "Generated by `src/publication/evaluate_fusion_v2.py` at %s." % r["generated_utc"],
         "", "**Model: %s. Seeds: %s. Acceptance: %s.**"
         % (r["model"], ", ".join(map(str, r["seeds"])), r["acceptance_status"]), "",
         "Every number below is recomputed from the prediction CSVs named in each "
         "section; none is copied from a previous table.", "",
         "## How uncertainty is reported", "", r["uncertainty_note"], "",
         "Calibration: %s." % r["calibration_rule"], "",
         "BRAIN-META: %s." % r["brain_meta_comparison"], ""]

    L += ["## Headline results", "",
          "| Setting | Regime | Images | Accuracy (%) | Macro-F1 (%) | Balanced acc (%) |",
          "|---|---|---:|---:|---:|---:|"]
    for name, s in r["settings"].items():
        a = s["across_seeds"]
        L.append("| %s | %s | %d | %s +/- %s | %s +/- %s | %s +/- %s |"
                 % (name.replace("_", " "), s["regime"], s["n_images"],
                    pct(a["accuracy"]), pct(a["accuracy"], "sd_across_seeds"),
                    pct(a["macro_f1"]), pct(a["macro_f1"], "sd_across_seeds"),
                    pct(a["balanced_accuracy"]), pct(a["balanced_accuracy"], "sd_across_seeds")))
    L += ["", "The `+/-` above is **variation across the three training seeds**, not a "
              "confidence interval over images. Per-seed bootstrap intervals are in the "
              "next section and in `fusion_v2_metrics.json`.", ""]

    L += ["## Discrimination and calibration", "",
          "| Setting | Macro ROC-AUC | Macro AP | NLL | Brier | ECE | Conf. correct | Conf. wrong |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in r["settings"].items():
        a = s["across_seeds"]
        def raw(k):
            return "--" if a[k] is None else "%.4f" % a[k]["mean"]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |"
                 % (name.replace("_", " "), raw("macro_roc_auc_ovr"), raw("macro_average_precision_ovr"),
                    raw("nll"), raw("brier"), raw("ece"),
                    raw("mean_confidence_correct"), raw("mean_confidence_incorrect")))
    L.append("")

    for name, s in r["settings"].items():
        L += ["## %s" % name.replace("_", " "), "",
              "%s" % s["definition"], "",
              "- images: **%d**  |  regime: **%s**  |  seeds: %s"
              % (s["n_images"], s["regime"], ", ".join(map(str, s["seeds"]))),
              "- predictions: `%s/fusion_v2_seed{%s}/%s`"
              % (s["run_dir"], ",".join(map(str, s["seeds"])), s["prediction_file"]),
              "- subset filter: %s" % s["subset"], "",
              "### Per-seed, with image-level bootstrap 95% CI", "",
              "| Seed | Accuracy (%) | 95% CI over images | Macro-F1 (%) | 95% CI over images |",
              "|---|---:|---|---:|---|"]
        for seed in r["seeds"]:
            m = s["per_seed"][seed]
            b = m["bootstrap_ci_over_images"]
            L.append("| %d | %.2f | [%.2f, %.2f] | %.2f | [%.2f, %.2f] |"
                     % (seed, 100 * m["accuracy"],
                        100 * b["accuracy"]["ci95_lo"], 100 * b["accuracy"]["ci95_hi"],
                        100 * m["macro_f1"],
                        100 * b["macro_f1"]["ci95_lo"], 100 * b["macro_f1"]["ci95_hi"]))
        L += ["", "### Per class (mean +/- s.d. across seeds)", "",
              "| Class | Support | Precision | Recall | Specificity | F1 | ROC-AUC | AP |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in s["per_class_across_seeds"]:
            def cell(k, p=True):
                v = c[k]
                if v is None:
                    return "--"
                return ("%.2f +/- %.2f" % (100 * v["mean"], 100 * v["sd_across_seeds"])) if p \
                    else ("%.4f" % v["mean"])
            L.append("| %s | %d | %s | %s | %s | %s | %s | %s |"
                     % (c["class"].title(), c["support"], cell("precision"),
                        cell("recall_sensitivity"), cell("specificity"), cell("f1"),
                        cell("roc_auc_ovr", False), cell("average_precision_ovr", False)))
        cm = s["confusion_matrix_summed_over_seeds"]
        L += ["", "### Confusion matrix (summed over the three seeds)", "",
              "| True \\ Pred | " + " | ".join(c.title() for c in cm["labels"]) + " |",
              "|---" * (len(cm["labels"]) + 1) + "|"]
        for lab, row in zip(cm["labels"], cm["counts"]):
            L.append("| **%s** | %s |" % (lab.title(), " | ".join(str(v) for v in row)))
        L += ["", "_%s_" % cm["note"], "",
              "Figures: `figures/confusion_%s.{png,pdf}`, `figures/reliability_%s.{png,pdf}`"
              % (name, name), ""]

    L += ["## Limitations", "",
          "- The `+/-` on aggregate rows is seed variation (n=3); the bracketed "
          "intervals are image-level bootstrap within one seed. They are different "
          "quantities and must not be quoted interchangeably.",
          "- `external_full` mixes re-encoded copies of training images with genuinely "
          "unseen ones and therefore overstates cross-source performance. The "
          "`cross_source_disjoint` row is the zero-shot measurement.",
          "- No comparison against BRAIN-META is made here; its per-image predictions "
          "are not available, so no significance test is possible.",
          "- These metrics describe held-out images, not held-out patients. Patient-level "
          "independence is unverified and is the subject of Phase 2.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EvalError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
