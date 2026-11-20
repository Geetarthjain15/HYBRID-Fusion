"""Generate Phase X7 PNG/PDF publication figures from X7 source tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"internal": "#2563eb", "external": "#dc2626", "cnn": "#f59e0b", "vit": "#14b8a6"}


def save_pair(fig, base: Path) -> None:
    for ext in ("png", "pdf"):
        p = base.with_suffix("." + ext)
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = root / "publication_results/explainability/x7_reporting"
    tables, figs = out / "tables", out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    table = tables / ("smoke_headline_performance.csv" if args.smoke else "headline_performance.csv")
    if not table.exists():
        raise FileNotFoundError(table)
    perf = pd.read_csv(table)
    labels = {"original": "Original", "clean": "Duplicate-clean", "cross_source_disjoint": "Cross-source"}
    x = np.arange(len(perf))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    means = perf.accuracy_mean.to_numpy() * 100
    errs = perf.accuracy_sd_across_seeds.to_numpy() * 100
    bars = ax.bar(x, means, yerr=errs, capsize=4,
                  color=[COLORS["internal"] if s != "cross_source_disjoint" else COLORS["external"] for s in perf.setting])
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(max(0, means.min()-8), 101)
    ax.set_xticks(x, [labels[s] for s in perf.setting]); ax.grid(axis="y", alpha=.25)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.7, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_title("FusionV2 image-level accuracy across evaluation regimes")
    save_pair(fig, figs / ("smoke_classification_performance" if args.smoke else "classification_performance"))
    if args.smoke:
        print("X7 figure smoke PASS: PNG/PDF pair generated from smoke table")
        return 0

    recall = pd.read_csv(tables / "x6_class_recall.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    y = recall.recall_seed_mean.to_numpy()*100; e = recall.recall_seed_sd.to_numpy()*100
    ax.bar(recall["class"], y, yerr=e, capsize=4, color="#7c3aed")
    ax.set_ylabel("Recall, seed mean ± SD (%)"); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.25)
    ax.set_title("Duplicate-disjoint cross-source class recall")
    save_pair(fig, figs / "cross_source_class_recall")

    faith = pd.read_csv(tables / "x5_faithfulness_summary.csv")
    methods = faith.method.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, metric, title in zip(axes, ["deletion_auc", "insertion_auc"],
                                 ["Deletion AUC (lower is better)", "Insertion AUC (higher is better)"]):
        d = faith[faith.metric == metric].set_index("method").loc[methods].reset_index()
        err = np.vstack([d["mean"]-d.ci_low, d.ci_high-d["mean"]])
        ax.barh(np.arange(len(d)), d["mean"], xerr=err, capsize=3, color="#64748b")
        ax.set_yticks(np.arange(len(d)), [m.replace("_", " ") for m in d.method]); ax.invert_yaxis()
        ax.set_xlabel("AUC"); ax.set_title(title); ax.grid(axis="x", alpha=.25)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.subplots_adjust(wspace=.10)
    fig.suptitle("Duplicate-clean explanation faithfulness under the locked protocol")
    save_pair(fig, figs / "clean_faithfulness")

    shift = pd.read_csv(tables / "x6_shift_effects.csv")
    pred_q = ["tta_confidence", "tta_margin", "tta_entropy", "gate_cnn", "gate_vit", "cnn_vit_agreement_spearman"]
    d = shift[shift.quantity.isin(pred_q)].copy().set_index("quantity").loc[pred_q].reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(d)); val = d.difference_external_minus_clean.to_numpy()
    err = np.vstack([val-d.ci_low.to_numpy(), d.ci_high.to_numpy()-val])
    ax.errorbar(val, y, xerr=err, fmt="o", color="#111827", capsize=3)
    ax.axvline(0, color="#dc2626", lw=1); ax.set_yticks(y, [q.replace("_", " ") for q in d.quantity])
    ax.invert_yaxis(); ax.set_xlabel("External − clean mean (95% bootstrap CI)"); ax.grid(axis="x", alpha=.25)
    ax.set_title("Predictive and branch-behaviour shifts")
    save_pair(fig, figs / "cross_source_predictive_shift")

    stab = shift[shift.quantity.str.startswith("stability_")].copy()
    transforms = stab.quantity.drop_duplicates().tolist(); y = np.arange(len(transforms))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for offset, (method, color, label) in zip([-.12, .12], [("cnn_spearman", COLORS["cnn"], "CNN"), ("vit_spearman", COLORS["vit"], "ViT")]):
        d = stab[stab.method == method].set_index("quantity").loc[transforms]
        val = d.difference_external_minus_clean.to_numpy(); err = np.vstack([val-d.ci_low, d.ci_high-val])
        ax.errorbar(val, y+offset, xerr=err, fmt="o", color=color, label=label, capsize=3)
    ax.axvline(0, color="#111827", lw=1); ax.set_yticks(y, [t.replace("stability_", "").replace("_", " ") for t in transforms])
    ax.invert_yaxis(); ax.set_xlabel("External − clean Spearman stability (95% CI)"); ax.legend(); ax.grid(axis="x", alpha=.25)
    ax.set_title("Cross-source attribution-stability shifts")
    save_pair(fig, figs / "cross_source_stability_shift")

    dis = pd.read_csv(tables / "x6_branch_disagreement.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    vals = dis.rate.to_numpy()*100
    ax.bar(np.arange(len(dis)), vals, color=[COLORS["cnn"], COLORS["vit"], "#7c3aed"])
    ax.set_xticks(np.arange(len(dis)), [x.replace("_rate", "").replace("_", " ") for x in dis.comparison], rotation=15, ha="right")
    ax.set_ylabel("Disagreement rate (%)"); ax.grid(axis="y", alpha=.25); ax.set_title("External branch and fused-prediction disagreement")
    save_pair(fig, figs / "cross_source_branch_disagreement")
    print(f"X7 figures generated at {figs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
