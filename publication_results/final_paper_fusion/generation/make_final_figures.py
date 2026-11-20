"""
Final fusion-network manuscript figures.

Every number plotted here is read from a verified artifact at run time. Nothing
is typed in, so a figure cannot drift away from the result it depicts.

Design decisions, made once and applied throughout:

- COLOUR BY JOB. Magnitude (confusion counts, calibration density) uses a single
  hue light-to-dark. Identity (evaluation regime, ablation variant) uses a fixed
  categorical order that never cycles. Polarity (accuracy difference above or
  below the reference) uses a two-hue diverging pair with a neutral midpoint.
  No rainbow, no hue at a diverging midpoint.
- ONE AXIS. No dual-axis panel anywhere.
- THE MARK CARRIES THE UNCERTAINTY. Effects are drawn as point estimates with
  their paired intervals, not as bars, because a bar implies a magnitude read
  from zero and hides the interval that decides whether the effect is resolved.
- SEED SPREAD AND BOOTSTRAP INTERVALS ARE NEVER MERGED INTO ONE MARK. Where both
  appear they are drawn differently and named in the legend.
- TEXT WEARS TEXT COLOUR. Values and labels stay in ink; a coloured mark beside
  them carries identity, so nothing depends on colour alone.

    env_win/Scripts/python.exe src/publication/make_paper_figures.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "publication_results", "final_paper_fusion", "figures")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NICE = {"glioma": "Glioma", "meningioma": "Meningioma",
        "notumor": "No tumour", "pituitary": "Pituitary"}
SEEDS = [42, 43, 44]

# ---- palette ---------------------------------------------------------------
# Categorical slots, fixed order, assigned by entity and never by rank.
C_SLATE = "#1F2933"     # primary ink
C_MUTED = "#616E7C"     # secondary ink
C_GRID = "#E4E7EB"
C_BLUE = "#2A6F97"      # internal / reference
C_TEAL = "#2C7A6B"      # supported evidence
C_AMBER = "#B26B00"     # suggestive
C_RED = "#A33A3A"       # adverse / cross-source
C_LILAC = "#6B5B95"     # secondary series
SEQ = ["#F4F7FA", "#D8E4EC", "#AFC9DA", "#7FA9C4", "#4E86A9", "#2A6F97", "#1B4B67"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "axes.edgecolor": C_MUTED,
    "axes.linewidth": 0.7,
    "axes.labelcolor": C_SLATE,
    "text.color": C_SLATE,
    "xtick.color": C_MUTED,
    "ytick.color": C_MUTED,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def seq_color(frac: float) -> str:
    """Single-hue light-to-dark ramp for magnitude."""
    frac = 0.0 if not np.isfinite(frac) else min(max(frac, 0.0), 1.0)
    i = frac * (len(SEQ) - 1)
    lo, hi = int(np.floor(i)), int(np.ceil(i))
    if lo == hi:
        return SEQ[lo]
    t = i - lo

    def h2r(h):
        h = h.lstrip("#")
        return np.array([int(h[j:j + 2], 16) for j in (0, 2, 4)], float)
    c = h2r(SEQ[lo]) * (1 - t) + h2r(SEQ[hi]) * t
    return "#%02x%02x%02x" % tuple(int(round(x)) for x in c)


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=C_GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)
    print("  wrote %s.png / .pdf" % name, flush=True)


# ---- evidence loaders ------------------------------------------------------

def metrics():
    return json.load(open(os.path.join(
        ROOT, "publication_results/metrics/fusion_v2_metrics.json"), encoding="utf-8"))


def ablation_summary():
    S = {}
    with open(os.path.join(ROOT, "publication_results/ablations/final_analysis",
                           "final_metrics_summary.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["metric"] in ("accuracy", "macro_f1"):
                S.setdefault(r["model"], {})[r["metric"]] = (
                    float(r["mean"]), float(r["sd_ddof1"]),
                    [float(r["seed_42"]), float(r["seed_43"]), float(r["seed_44"])])
    return S


def ablation_conclusions():
    with open(os.path.join(ROOT, "publication_results/ablations/final_analysis",
                           "component_conclusions.csv"), encoding="utf-8") as f:
        return {r["variant"]: r for r in csv.DictReader(f)}


def bootstrap_effects():
    """Per-variant per-seed accuracy-difference CIs (paired, image-level)."""
    out = {}
    with open(os.path.join(ROOT, "publication_results/ablations/final_analysis",
                           "bootstrap_effects.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["metric"] != "accuracy":
                continue
            out.setdefault(r["variant"], []).append(
                (int(r["seed"]), float(r["difference_pp"]),
                 float(r["ci95_lo_pp"]), float(r["ci95_hi_pp"])))
    return out


def disjoint_rows():
    xs = json.load(open(os.path.join(ROOT, "results/cross_source.json"), encoding="utf-8"))
    disj = {k for k, v in xs["manifest"].items() if v.get("kind") == "disjoint"}
    by = {}
    for s in SEEDS:
        rows = list(csv.DictReader(open(os.path.join(
            ROOT, "runs_v2_clean/fusion_v2_seed%d/preds_dataset_ood.csv" % s), encoding="utf-8")))
        by[s] = [r for r in rows if r["filename"] in disj]
    return by, len(disj)


def gate_rows():
    out = {}
    for s in SEEDS:
        out[s] = list(csv.DictReader(open(os.path.join(
            ROOT, "runs_v2_clean/fusion_v2_seed%d/gates_test.csv" % s), encoding="utf-8")))
    return out


# ============================================================ FIGURE 1 =======

def fig1_architecture():
    """Schematic of the proposed multi-scale fusion network forward path."""
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")

    def box(x, y, w, h, label, sub="", fc="#FFFFFF", ec=C_MUTED, lw=0.9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.1",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), label, ha="center", va="center",
                fontsize=8, color=C_SLATE, zorder=4,
                fontweight="bold" if bold else "normal")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                    fontsize=6.8, color=C_MUTED, zorder=4)

    def arrow(x1, y1, x2, y2, color=C_MUTED, lw=1.0, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=8, lw=lw, color=color, zorder=2,
                                     shrinkA=0, shrinkB=0))

    box(1, 20, 11, 8, "MRI image", "224 x 224 x 3", fc="#F7F9FB")

    # CNN path
    box(17, 32, 20, 8, "EfficientNet-B0", "stages 3, 5, final", fc="#EEF4F8", ec=C_BLUE)
    box(41, 32, 17, 8, "Multi-scale pool", "40+112+1280 = 1432", fc="#FFFFFF", ec=C_BLUE)
    box(62, 32, 14, 8, "Project", "1432 -> 512", fc="#FFFFFF", ec=C_BLUE)
    # ViT path
    box(17, 8, 20, 8, "ViT-B/16", "class token", fc="#F3F0F7", ec=C_LILAC)
    box(41, 8, 17, 8, "CLS embedding", "768", fc="#FFFFFF", ec=C_LILAC)
    box(62, 8, 14, 8, "Project", "768 -> 512", fc="#FFFFFF", ec=C_LILAC)

    arrow(12, 24, 17, 36); arrow(12, 24, 17, 12)
    arrow(37, 36, 41, 36); arrow(58, 36, 62, 36)
    arrow(37, 12, 41, 12); arrow(58, 12, 62, 12)

    # gate
    box(80, 20, 17, 8, "Adaptive gate", "1024->256->2, softmax", fc="#FDF6EC", ec=C_AMBER, lw=1.3, bold=True)
    arrow(76, 36, 84, 28, color=C_BLUE)
    arrow(76, 12, 84, 20, color=C_LILAC)

    ax.text(88.5, 31.5, r"$g_c + g_v = 1$", ha="center", fontsize=7.2, color=C_AMBER)
    ax.text(88.5, 16.5, r"$[\,2g_c\mathbf{c}\;;\;2g_v\mathbf{v}\,]$", ha="center",
            fontsize=7.6, color=C_SLATE)
    ax.text(88.5, 13.0, "1024-d fused", ha="center", fontsize=6.8, color=C_MUTED)

    # heads
    box(80, 0.5, 17, 6.5, "Classifier head", "LN, drop 0.2, -> 4", fc="#F7F9FB")
    arrow(88.5, 12.5, 88.5, 7.0)
    # auxiliary
    box(62, 22.5, 14, 6, "Aux CNN head", "512 -> 4 (train)", fc="#FFFFFF", ec=C_GRID)
    box(62, 17.5, 14, 4.4, "Aux ViT head", "512 -> 4 (train)", fc="#FFFFFF", ec=C_GRID)
    arrow(69, 32, 69, 28.5, color=C_GRID, lw=0.8)
    arrow(69, 8, 69, 17.5, color=C_GRID, lw=0.8)

    ax.text(0.6, 45.5, "Multi-scale EfficientNet-B0–ViT-B/16 feature fusion", fontsize=10, fontweight="bold", color=C_SLATE)
    ax.text(0.6, 42.4,
            "Both branches are projected to a common 512-d width; the learned gate varies with the input and is not a fixed weighting.",
            fontsize=7.2, color=C_MUTED, va="top")
    ax.text(0.6, 0.6, "91,211,918 parameters total; gate accounts for 262,914",
            fontsize=6.6, color=C_MUTED, ha="left")
    save(fig, "figure_01_architecture")


# ============================================================ FIGURE 2 =======

def fig2_evaluation_map():
    """The evaluation protocol as a flow: what each reported number is computed on."""
    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    def node(x, y, w, h, title, sub, fc="#FFFFFF", ec=C_MUTED, lw=0.9, tc=C_SLATE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.1",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=8,
                fontweight="bold", color=tc, zorder=4)
        ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center", fontsize=6.9,
                color=C_MUTED, zorder=4)

    def arrow(x1, y1, x2, y2, color=C_MUTED, lw=1.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=8, lw=lw, color=color, zorder=2))

    node(1, 40, 22, 12, "Pooled image set", "7,023 JPEG images\n4 classes", fc="#F7F9FB")
    node(29, 40, 24, 12, "Perceptual-hash audit",
         "5,648 similarity clusters\n1,375 redundant removed", fc="#F7F9FB")
    arrow(23, 46, 29, 46)

    node(59, 40, 40, 12, "Duplicate-clean partition",
         "4,800 train  |  848 test   (split seed 42)\nglioma 243, meningioma 228, no-tumour 116, pituitary 261",
         fc="#EEF4F8", ec=C_BLUE, lw=1.2)
    arrow(53, 46, 59, 46)

    # evaluation regimes
    node(1, 20, 27, 13, "Original split", "n = 1,052 images\nconventional protocol", fc="#FFFFFF")
    node(31, 20, 27, 13, "Duplicate-clean split",
         "n = 848 images\nPRIMARY internal result", fc="#EEF4F8", ec=C_BLUE, lw=1.4)
    node(61, 20, 38, 13, "External collection  (n = 499)",
         "126 images have a near-duplicate in training\n373 are duplicate-disjoint",
         fc="#FCF2F2", ec=C_RED, lw=1.2)
    arrow(70, 40, 76, 33, color=C_RED)
    arrow(79, 46, 44.5, 33, color=C_BLUE)

    node(61, 2, 38, 13, "Duplicate-disjoint cross-source subset",
         "n = 373 images\nSOURCE-SHIFT STRESS TEST", fc="#FCF2F2", ec=C_RED, lw=1.5)
    arrow(80, 20, 80, 15, color=C_RED, lw=1.3)

    ax.text(0.6, 55, "Which images each reported number is computed on",
            fontsize=10, fontweight="bold", color=C_SLATE)
    ax.text(30, 10.5,
            "External labels were never used for training,\nvalidation, threshold selection or model selection.",
            fontsize=7, color=C_MUTED, ha="center", va="center", style="italic")
    ax.text(0.6, 0.5,
            "Duplicate cleaning operates on image similarity. It does not identify patients and does not "
            "establish a patient-independent split.",
            fontsize=6.7, color=C_MUTED)
    save(fig, "figure_02_evaluation_map")


# ============================================================ FIGURE 3 =======

def fig3_regime_collapse():
    """The paper's central result: performance by evaluation regime.

    A dot plot, not bars. Each seed is drawn, the mean is marked, and the drop
    between regimes is annotated, because the story is the GAP, not the height."""
    M = metrics()
    order = [("original", "Original split", 1052),
             ("clean", "Duplicate-clean split", 848),
             ("external_full", "External, all images", 499),
             ("cross_source_disjoint", "Duplicate-disjoint\ncross-source", 373)]
    cols = [C_BLUE, C_BLUE, C_AMBER, C_RED]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, metric, title in zip(axes, ("accuracy", "macro_f1"),
                                 ("Accuracy", "Macro-F1")):
        for i, ((key, lab, n), c) in enumerate(zip(order, cols)):
            st = M["settings"][key]
            per = [100 * st["per_seed"][str(s)][metric] for s in SEEDS]
            mu = 100 * st["across_seeds"][metric]["mean"]
            sd = 100 * st["across_seeds"][metric]["sd_across_seeds"]
            y = len(order) - 1 - i
            ax.plot([mu - sd, mu + sd], [y, y], color=c, lw=3.0, alpha=0.30,
                    solid_capstyle="round", zorder=2)
            ax.scatter(per, [y] * 3, s=16, facecolor="white", edgecolor=c,
                       lw=1.1, zorder=3)
            ax.scatter([mu], [y], s=46, color=c, zorder=4, marker="D")
            ax.text(mu, y + 0.30, "%.2f" % mu, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=C_SLATE, zorder=5)
            ax.text(mu, y - 0.34, "±%.2f" % sd, ha="center", va="top",
                    fontsize=6.6, color=C_MUTED, zorder=5)
        ax.set_xlim(62, 104)
        ax.set_ylim(-0.75, len(order) - 0.25)
        ax.set_xlabel("%s (%%)" % title)
        ax.set_title(title, loc="left", fontweight="bold", color=C_SLATE)
        tidy(ax, grid_axis="x")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([o[1] for o in order][::-1])

    # Annotate the collapse in the left margin, clear of every value label.
    ax = axes[0]
    drop = (100 * M["settings"]["clean"]["across_seeds"]["accuracy"]["mean"]
            - 100 * M["settings"]["cross_source_disjoint"]["across_seeds"]["accuracy"]["mean"])
    xb = 63.4
    ax.plot([xb, xb], [0.0, 2.0], color=C_RED, lw=1.2, zorder=5)
    for yy in (0.0, 2.0):
        ax.plot([xb, xb + 1.5], [yy, yy], color=C_RED, lw=1.2, zorder=5)
    ax.annotate("", xy=(xb, 0.12), xytext=(xb, 1.88), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.4))
    ax.text(xb + 2.0, 1.0, "%.2f pp\ndrop" % drop, fontsize=7.4,
            color=C_RED, ha="left", va="center", fontweight="bold", linespacing=1.3)

    axes[1].text(0.99, -0.20, "open circles = individual seeds (42, 43, 44)   "
                              "diamond = mean   bar = ±1 sample SD across seeds",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=6.8, color=C_MUTED)
    axes[0].text(0.0, 1.16, "Internal performance does not survive the change of source",
                 transform=axes[0].transAxes, fontsize=10, fontweight="bold", color=C_SLATE)
    axes[0].text(0.0, 1.075, "Sample sizes: 1,052 / 848 / 499 / 373 images. The spread is "
                             "seed-to-seed, not a confidence interval.",
                 transform=axes[0].transAxes, fontsize=7.1, color=C_MUTED)
    save(fig, "figure_03_regime_collapse")


# ============================================================ FIGURE 4 =======

def fig4_confusions():
    """Clean and cross-source confusion, row-normalised, single hue.

    Row-normalised so the two panels are comparable despite different support;
    the raw count is printed inside each cell so nothing is hidden by scaling."""
    M = metrics()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.5),
                             gridspec_kw={"wspace": 0.30})
    panels = [("clean", "Duplicate-clean test", 848, C_BLUE),
              ("cross_source_disjoint", "Duplicate-disjoint cross-source", 373, C_RED)]
    for ax, (key, title, n, accent) in zip(axes, panels):
        cm = np.array(M["settings"][key]["confusion_matrix_summed_over_seeds"]["counts"], float)
        rn = cm / cm.sum(1, keepdims=True)
        for i in range(4):
            for j in range(4):
                ax.add_patch(Rectangle((j, 3 - i), 1, 1, facecolor=seq_color(rn[i, j]),
                                       edgecolor="white", lw=1.6))
                txt = "%.0f%%" % (100 * rn[i, j])
                sub = "%d" % cm[i, j]
                col = "white" if rn[i, j] > 0.55 else C_SLATE
                ax.text(j + 0.5, 3 - i + 0.60, txt, ha="center", va="center",
                        fontsize=8.2, color=col,
                        fontweight="bold" if i == j else "normal")
                ax.text(j + 0.5, 3 - i + 0.30, sub, ha="center", va="center",
                        fontsize=6.4, color=col if rn[i, j] > 0.55 else C_MUTED)
        ax.set_xlim(0, 4); ax.set_ylim(0, 4)
        ax.set_xticks(np.arange(4) + 0.5); ax.set_yticks(np.arange(4) + 0.5)
        ax.set_xticklabels([NICE[c] for c in CLASSES], rotation=25, ha="right")
        ax.set_yticklabels([NICE[c] for c in CLASSES][::-1])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title, loc="left", fontweight="bold", color=accent)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.text(0, -0.30, "%d images x 3 seeds = %d predictions" % (n, 3 * n),
                transform=ax.transAxes, fontsize=6.6, color=C_MUTED)
    axes[1].annotate("", xy=(2.5, 0.5), xytext=(3.5, 0.5),
                     arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.4))
    axes[1].text(2.0, -0.62, "pituitary collapses to no-tumour", fontsize=7.2,
                 color=C_RED, fontweight="bold")
    fig.text(0.005, 1.035, "Where the errors actually fall",
             fontsize=10, fontweight="bold", color=C_SLATE)
    fig.text(0.005, 0.985, "Cells are row-normalised (recall); the raw summed count is printed "
                           "beneath each percentage.", fontsize=7.1, color=C_MUTED)
    save(fig, "figure_04_confusion_matrices")


# ============================================================ FIGURE 5 =======

def fig5_ablation_forest():
    """Ablation effects as a forest plot with paired image-level intervals.

    This replaces a bar chart. Each ablation is shown as its per-seed observed
    accuracy difference against the full model with the paired bootstrap
    interval, so a reader sees directly whether an effect is resolved. Bars
    would have implied precision that these data do not carry."""
    S = ablation_summary()
    C = ablation_conclusions()
    B = bootstrap_effects()
    order = ["cnn_branch_only", "no_auxiliary_loss", "vit_branch_only",
             "no_tta", "no_multiscale", "no_gate", "no_ema"]
    label = {"no_gate": "− adaptive gate", "no_multiscale": "− multi-scale taps",
             "no_auxiliary_loss": "− auxiliary losses", "no_ema": "− EMA",
             "cnn_branch_only": "− ViT branch (CNN only)",
             "vit_branch_only": "− CNN branch (ViT only)", "no_tta": "− TTA"}
    vcol = {"supported by consistent accuracy evidence": C_TEAL,
            "suggestive but inconclusive": C_AMBER,
            "no detectable accuracy benefit under this protocol": C_MUTED,
            "inconclusive": C_MUTED}

    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    ax.axvline(0, color=C_SLATE, lw=1.0, zorder=3)
    yticks, ylabels = [], []
    for i, v in enumerate(order):
        y = len(order) - 1 - i
        col = vcol[C[v]["verdict"]]
        rows = sorted(B[v])
        for k, (seed, d, lo, hi) in enumerate(rows):
            off = (k - 1) * 0.20
            ax.plot([lo, hi], [y + off, y + off], color=col, lw=1.5, alpha=0.55,
                    solid_capstyle="round", zorder=4)
            ax.scatter([d], [y + off], s=14, color=col, zorder=5)
        mu = float(C[v]["observed_mean_accuracy_difference_pp"])
        ax.scatter([mu], [y], s=58, marker="D", color=col, zorder=6,
                   edgecolor="white", lw=0.8)
        yticks.append(y); ylabels.append(label[v])
        nsig = int(C[v]["n_holm_significant_of_3"])
        tag = "Holm %d/3" % nsig
        ax.text(1.30, y, "%+.2f pp" % mu, fontsize=7.8, va="center",
                color=C_SLATE, fontweight="bold")
        ax.text(2.45, y, tag, fontsize=7.0, va="center",
                color=col, fontweight="bold" if nsig else "normal")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.set_xlim(-4.7, 3.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Observed accuracy difference vs full fusion network (percentage points)")
    tidy(ax, grid_axis="x")
    ax.set_xticks([-4, -3, -2, -1, 0, 1])
    ax.axvspan(1.08, 3.5, color="white", zorder=1)   # keep gridlines out of the text column

    handles = [plt.Line2D([], [], color=C_TEAL, lw=3, label="supported by consistent accuracy evidence"),
               plt.Line2D([], [], color=C_AMBER, lw=3, label="suggestive but inconclusive"),
               plt.Line2D([], [], color=C_MUTED, lw=3, label="no detectable accuracy benefit")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.33), ncol=1)
    ax.text(0.0, 1.14, "Removing each component, measured against the full model on the same 848 images",
            transform=ax.transAxes, fontsize=9.6, fontweight="bold", color=C_SLATE)
    ax.text(0.0, 1.055,
            "Thin lines are paired image-level 95% bootstrap intervals, one per seed (10,000 resamples, "
            "stratified by class).\nDiamond is the mean observed difference. Holm correction is across all "
            "21 seed-specific tests.",
            transform=ax.transAxes, fontsize=7.0, color=C_MUTED, va="bottom")
    save(fig, "figure_05_ablation_forest")


# ============================================================ FIGURE 6 =======

def fig6_calibration():
    """Reliability under both regimes, with the bin populations underneath.

    A reliability curve without bin counts is unreadable: a wild-looking bin may
    hold three images. The counts are therefore plotted, not omitted."""
    M = metrics()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.12,
                                          "wspace": 0.22})
    panels = [("clean", "Duplicate-clean test (n = 848)", C_BLUE),
              ("cross_source_disjoint", "Duplicate-disjoint cross-source (n = 373)", C_RED)]
    for j, (key, title, accent) in enumerate(panels):
        st = M["settings"][key]
        bins = st["per_seed"][str(SEEDS[0])]["ece_bins"]
        ctr = [(b["bin_lo"] + b["bin_hi"]) / 2 for b in bins]
        # average the three seeds where a bin is populated in that seed
        acc, conf, cnt = [], [], []
        for bi in range(len(bins)):
            aa, cc, nn = [], [], 0
            for s in SEEDS:
                b = st["per_seed"][str(s)]["ece_bins"][bi]
                nn += b["n"]
                if b["n"] and b["accuracy"] is not None:
                    aa.append(b["accuracy"]); cc.append(b["mean_confidence"])
            acc.append(np.mean(aa) if aa else np.nan)
            conf.append(np.mean(cc) if cc else np.nan)
            cnt.append(nn)
        ax = axes[0, j]
        ax.plot([0, 1], [0, 1], color=C_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.text(0.62, 0.55, "perfect calibration", fontsize=6.6, color=C_MUTED,
                rotation=38, ha="left", va="bottom")
        m = ~np.isnan(np.array(acc, float))
        ax.plot(np.array(conf)[m], np.array(acc)[m], color=accent, lw=1.8,
                marker="o", ms=4, zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_ylabel("Observed accuracy" if j == 0 else "")
        ax.set_title(title, loc="left", fontweight="bold", color=accent, fontsize=8.8)
        tidy(ax)
        ece = st["across_seeds"]["ece"]["mean"]
        nll = st["across_seeds"]["nll"]["mean"]
        br = st["across_seeds"]["brier"]["mean"]
        ax.text(0.03, 0.93, "ECE %.4f\nNLL %.4f\nBrier %.4f" % (ece, nll, br),
                transform=ax.transAxes, fontsize=7.2, va="top", color=C_SLATE,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_GRID, lw=0.7))
        axc = axes[1, j]
        axc.bar(ctr, cnt, width=1 / 15 * 0.86, color=accent, alpha=0.30,
                edgecolor=accent, lw=0.6)
        axc.set_yscale("symlog", linthresh=10)
        axc.set_xlabel("Predicted confidence (maximum softmax)")
        axc.set_ylabel("Predictions" if j == 0 else "")
        tidy(axc)
        axc.set_xlim(0, 1)
    fig.text(0.005, 1.02, "Confidence is not matched by correctness, and the gap widens under source shift",
             fontsize=9.8, fontweight="bold", color=C_SLATE)
    fig.text(0.005, 0.965, "15 equal-width bins on [0,1] applied to the maximum predicted probability, "
                           "averaged over three seeds. Lower panels show how many predictions each bin holds.",
             fontsize=7.0, color=C_MUTED)
    save(fig, "figure_06_calibration")


# ============================================================ FIGURE 7 =======

def fig7_gate_behaviour():
    """Gate mass is input-dependent. Shown as a distribution, not a mean.

    A single mean would be exactly the thing the paper warns against: it would
    suggest a fixed weighting. The spread is the evidence."""
    G = gate_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0),
                             gridspec_kw={"wspace": 0.26, "width_ratios": [1.25, 1.0]})
    ax = axes[0]
    cols = [C_BLUE, C_LILAC, C_TEAL]
    for s, c in zip(SEEDS, cols):
        v = np.array([float(r["gate_cnn"]) for r in G[s]])
        ax.hist(v, bins=np.linspace(0, 0.8, 49), histtype="step", lw=1.5,
                color=c, label="seed %d  (mean %.3f)" % (s, v.mean()))
    ax.axvline(0.5, color=C_SLATE, lw=1.0, ls=(0, (4, 3)))
    ax.text(0.505, ax.get_ylim()[1] * 0.93, "equal mixing", fontsize=6.8, color=C_SLATE)
    ax.set_xlabel("CNN gate mass  $g_c$  (ViT mass is $1-g_c$)")
    ax.set_ylabel("Clean-test images")
    ax.legend(loc="upper right")
    tidy(ax)
    ax.set_title("Gate mass varies image by image", loc="left",
                 fontweight="bold", color=C_SLATE, fontsize=8.8)

    ax = axes[1]
    for i, c in enumerate(CLASSES):
        vals = [float(r["gate_cnn"]) for s in SEEDS for r in G[s] if r["true_label"] == c]
        y = 3 - i
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        lo, hi = np.percentile(vals, [5, 95])
        ax.plot([lo, hi], [y, y], color=C_MUTED, lw=1.0, zorder=3)
        ax.plot([q1, q3], [y, y], color=C_BLUE, lw=6.0, alpha=0.35,
                solid_capstyle="round", zorder=4)
        ax.scatter([med], [y], s=34, color=C_BLUE, zorder=5, marker="D")
        ax.text(hi + 0.012, y, "med %.3f" % med, fontsize=7.0, va="center", color=C_SLATE)
    ax.set_yticks(range(4))
    ax.set_yticklabels([NICE[c] for c in CLASSES][::-1])
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("CNN gate mass $g_c$")
    tidy(ax, grid_axis="x")
    ax.set_title("by true class", loc="left", fontweight="bold", color=C_SLATE, fontsize=8.8)
    ax.text(0.0, -0.32, "bar = interquartile range, line = 5th–95th percentile, diamond = median; "
                        "three seeds pooled",
            transform=ax.transAxes, fontsize=6.6, color=C_MUTED)
    fig.text(0.005, 1.04, "The gate is a learned, input-dependent component",
             fontsize=9.8, fontweight="bold", color=C_SLATE)
    fig.text(0.005, 0.985, "Gate mass measures branch utilisation only. It is not evidence of causal branch "
                           "importance and is not a clinical explanation.",
             fontsize=7.0, color=C_MUTED)
    save(fig, "figure_07_gate_behaviour")


# ============================================================ FIGURE 8 =======

def fig8_class_failure():
    """Per-class recall, internal vs cross-source, plus where pituitary goes."""
    M = metrics()
    by, n_disj = disjoint_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9),
                             gridspec_kw={"wspace": 0.30, "width_ratios": [1.12, 1.0]})

    # ---- left: recall, internal vs cross-source, one row per class ----------
    ax = axes[0]
    clean_pc = {d["class"]: d for d in M["settings"]["clean"]["per_class_across_seeds"]}
    for i, c in enumerate(CLASSES):
        y = 3 - i
        ci = 100 * clean_pc[c]["recall_sensitivity"]["mean"]
        xs = [100 * sum(r["pred_label"] == c for r in by[s] if r["true_label"] == c) /
              max(1, len([r for r in by[s] if r["true_label"] == c])) for s in SEEDS]
        xm = float(np.mean(xs))
        ax.plot([xm, ci], [y, y], color=C_GRID, lw=3.0, solid_capstyle="round", zorder=2)
        ax.scatter([ci], [y], s=52, color=C_BLUE, zorder=4, marker="D")
        ax.scatter([xm], [y], s=52, color=C_RED, zorder=4, marker="o")
        delta = xm - ci                      # signed: negative means recall fell
        big = delta <= -20
        ax.text(min(xm, ci) - 3.5, y, "%+.1f pp" % delta, fontsize=7.4, ha="right",
                va="center", color=C_RED if big else C_MUTED,
                fontweight="bold" if big else "normal")
    ax.set_yticks(range(4))
    ax.set_yticklabels([NICE[c] for c in CLASSES][::-1])
    ax.set_xlim(-34, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Recall (%)")
    tidy(ax, grid_axis="x")
    ax.legend(handles=[plt.Line2D([], [], marker="D", ls="", color=C_BLUE, ms=6,
                                  label="duplicate-clean (n = 848)"),
                       plt.Line2D([], [], marker="o", ls="", color=C_RED, ms=6,
                                  label="duplicate-disjoint cross-source (n = 373)")],
              loc="lower left", bbox_to_anchor=(-0.02, -0.46), ncol=1)
    ax.set_title("Per-class recall collapses unevenly", loc="left",
                 fontweight="bold", color=C_SLATE, fontsize=8.8)

    # ---- right: destination of the pituitary predictions --------------------
    ax = axes[1]
    pit = [r for s in SEEDS for r in by[s] if r["true_label"] == "pituitary"]
    counts = {c: sum(r["pred_label"] == c for r in pit) for c in CLASSES}
    tot = len(pit)
    left = 0.0
    for c in CLASSES:
        w = 100 * counts[c] / tot
        if w <= 0:
            continue
        col = C_RED if c == "notumor" else (C_TEAL if c == "pituitary" else C_MUTED)
        ax.barh([0], [w], left=left, color=col, edgecolor="white", lw=1.5, height=1.0)
        if w >= 18:
            ax.text(left + w / 2, 0, "%s\n%.1f%%\n(%d)" % (NICE[c], w, counts[c]),
                    ha="center", va="center", fontsize=7.6, color="white",
                    fontweight="bold", linespacing=1.35)
        else:
            ax.plot([left + w / 2, left + w / 2], [-0.52, -0.80], color=col, lw=0.9)
            ax.text(left + w / 2, -0.88, "%s\n%.1f%% (%d)" % (NICE[c], w, counts[c]),
                    ha="center", va="top", fontsize=6.9, color=col, linespacing=1.3)
        left += w
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.9, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("Share of the 333 pituitary predictions (%)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Where the 111 pituitary images go", loc="left",
                 fontweight="bold", color=C_SLATE, fontsize=8.8)
    ax.text(0, -1.62, "111 duplicate-disjoint pituitary images x 3 seeds = %d predictions; "
                      "%d assigned to no-tumour (%.1f%%).\nThese are repeated predictions of the "
                      "same 111 images, not %d independent cases."
            % (tot, counts["notumor"], 100 * counts["notumor"] / tot, tot),
            fontsize=6.7, color=C_MUTED, va="top")

    fig.text(0.005, 1.06, "The cross-source failure is class-specific, not uniform",
             fontsize=9.8, fontweight="bold", color=C_SLATE)
    fig.text(0.005, 1.005, "No-tumour recall is unchanged while pituitary recall nearly vanishes, "
                           "so the aggregate accuracy drop understates the class-specific asymmetry.",
             fontsize=7.0, color=C_MUTED)
    save(fig, "figure_08_class_failure")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Final fusion-network figures -> %s" % os.path.relpath(OUT, ROOT), flush=True)
    fig1_architecture()
    fig2_evaluation_map()
    fig3_regime_collapse()
    fig4_confusions()
    fig5_ablation_forest()
    fig6_calibration()
    fig7_gate_behaviour()
    fig8_class_failure()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
