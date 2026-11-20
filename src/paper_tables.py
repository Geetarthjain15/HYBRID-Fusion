"""
Regenerate every table the paper needs, from the prediction CSVs alone.

One entry point so the results in the manuscript can never drift from the runs
that produced them: delete results/ and re-run this, and every number comes
back identical. Emits Markdown (for drafting) and LaTeX (for submission).

    python src/paper_tables.py --out results/paper

Tables produced:
    T1  main results        arch x split, acc / macro-F1 / balanced acc, mean +- std
    T2  diagnostic protocol oracle ceiling, gate mass, paired equivalence
    T3  evaluation regimes  in-distribution / resampling / cross-source
    T4  per-class recall    where the architectures actually differ
    T5  calibration         ECE, Brier, confidence gate behaviour
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

CLS = ["glioma", "meningioma", "notumor", "pituitary"]


def load(path, keep=None):
    with open(path) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["filename"])
    if keep is not None:
        rows = [r for r in rows if r["filename"] in keep]
    if not rows:
        return None
    classes = [c[2:] for c in rows[0] if c.startswith("p_")]
    y = np.array([classes.index(r["true_label"]) for r in rows])
    P = np.array([[float(r["p_" + c]) for c in classes] for r in rows])
    return y, P, classes


def metrics(y, P, classes):
    pred = P.argmax(1)
    acc = (pred == y).mean()
    recs, f1s = [], []
    for k in range(len(classes)):
        tp = ((pred == k) & (y == k)).sum()
        fp = ((pred == k) & (y != k)).sum()
        fn = ((pred != k) & (y == k)).sum()
        r = tp / (tp + fn) if tp + fn else 0.0
        p = tp / (tp + fp) if tp + fp else 0.0
        recs.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    conf = P.max(1)
    correct = pred == y
    # 15-bin ECE
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    onehot = np.eye(len(classes))[y]
    brier = ((P - onehot) ** 2).sum(1).mean()
    # confidence gate at 0.40 / 0.60, as deployed in the app
    refused = conf < 0.40
    warned = (conf >= 0.40) & (conf < 0.60)
    accepted = conf >= 0.60
    return {
        "n": int(len(y)), "acc": float(acc),
        "balanced_acc": float(np.mean(recs)), "macro_f1": float(np.mean(f1s)),
        "recall": {c: float(r) for c, r in zip(classes, recs)},
        "ece": float(ece), "brier": float(brier),
        "mean_conf": float(conf.mean()),
        "conf_correct": float(conf[correct].mean()) if correct.any() else float("nan"),
        "conf_wrong": float(conf[~correct].mean()) if (~correct).any() else float("nan"),
        "gate_refused_pct": float(100 * refused.mean()),
        "gate_warned_pct": float(100 * warned.mean()),
        "gate_accepted_acc": float(correct[accepted].mean()) if accepted.any() else float("nan"),
    }


def agg(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def collect(runs_dirs, evalset, keep=None):
    """-> {(split, arm): {seed: metrics}}"""
    out = defaultdict(dict)
    for rd in runs_dirs:
        # runs_v2_original holds the SAME split as runs_original, just a later
        # architecture. Map it onto that split so fusion_v2 is a row in the
        # main table rather than a column of its own.
        base = os.path.basename(rd).replace("runs_", "")
        split = base[3:] if base.startswith("v2_") else base
        for p in sorted(glob.glob(os.path.join(rd, "*", "preds_%s.csv" % evalset))):
            run = os.path.basename(os.path.dirname(p))
            arch, _, rest = run.partition("_seed")
            seed_s, _, tag = rest.partition("_")
            arm = arch + ("/" + tag if tag else "")
            d = load(p, keep)
            if d:
                out[(split, arm)][int(seed_s)] = metrics(*d)
    return out


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def tex_table(headers, rows, caption, label):
    out = ["\\begin{table}[t]", "\\centering",
           "\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}", "\\toprule",
           " & ".join(headers) + " \\\\", "\\midrule"]
    out += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}",
            "\\caption{%s}" % caption, "\\label{tab:%s}" % label, "\\end{table}"]
    return "\n".join(out)


def fmt(m, s, pct=True, dp=2):
    scale = 100 if pct else 1
    return "%.*f $\\pm$ %.*f" % (dp, scale * m, dp, scale * s) if s else "%.*f" % (dp, scale * m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs_original", "runs_clean"])
    ap.add_argument("--cross_source", default="results/cross_source.json")
    ap.add_argument("--diagnostics", nargs="*", default=glob.glob("results/diagnostics_*.json"))
    ap.add_argument("--out", default="results/paper")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    md, tex = [], []

    disjoint = overlapping = None
    if os.path.exists(args.cross_source):
        cs = json.load(open(args.cross_source))
        disjoint = {k for k, v in cs["manifest"].items() if v["kind"] == "disjoint"}
        overlapping = {k for k, v in cs["manifest"].items() if v["kind"] == "overlapping"}

    # ---- T1 main results ------------------------------------------------
    D = collect(args.runs, "test")
    arms = sorted({a for _, a in D})
    # dedupe: runs_v2_* map onto the same split as runs_*, so the column list
    # must be the distinct splits, not the distinct directories.
    splits = []
    for r in args.runs:
        b = os.path.basename(r).replace("runs_", "")
        b = b[3:] if b.startswith("v2_") else b
        if b not in splits:
            splits.append(b)
    hdr = ["Model"] + ["%s: Acc" % s for s in splits] + ["%s: Macro-F1" % s for s in splits]
    rows = []
    for arm in arms:
        r = [arm]
        for key in ("acc", "macro_f1"):
            for sp in splits:
                d = D.get((sp, arm))
                r.append(fmt(*agg([v[key] for v in d.values()])) if d else "--")
        rows.append(r)
    md.append("## T1 Main results (in-distribution test)\n\n" + md_table(hdr, rows))
    tex.append(tex_table(hdr, rows, "Classification accuracy and macro-F1, mean $\\pm$ s.d. over three seeds.", "main"))

    # ---- T3 evaluation regimes -----------------------------------------
    if disjoint:
        O = collect(args.runs, "dataset_ood")
        Oo = collect(args.runs, "dataset_ood", overlapping)
        Od = collect(args.runs, "dataset_ood", disjoint)
        hdr3 = ["Model", "In-distribution", "Resampled (n=%d)" % len(overlapping),
                "Cross-source (n=%d)" % len(disjoint)]
        for sp in splits:
            rows3 = []
            for arm in arms:
                def g(src):
                    d = src.get((sp, arm))
                    return fmt(*agg([v["acc"] for v in d.values()])) if d else "--"
                rows3.append([arm, g(D), g(Oo), g(Od)])
            md.append("## T3 Evaluation regimes (%s split)\n\n" % sp + md_table(hdr3, rows3))
            tex.append(tex_table(hdr3, rows3,
                "Accuracy by evaluation regime (%s split). The resampled subset consists of "
                "re-encoded copies of training images; the cross-source subset has no near-duplicate "
                "in training." % sp, "regimes_%s" % sp))

    # ---- T4 per-class recall -------------------------------------------
    for sp in splits:
        hdr4 = ["Model"] + [c.title() for c in CLS]
        rows4 = []
        for arm in arms:
            d = D.get((sp, arm))
            if not d:
                continue
            rows4.append([arm] + [fmt(*agg([v["recall"][c] for v in d.values()])) for c in CLS])
        md.append("## T4 Per-class recall (%s split, in-distribution)\n\n" % sp + md_table(hdr4, rows4))
        tex.append(tex_table(hdr4, rows4, "Per-class recall (%s split)." % sp, "recall_%s" % sp))

    # ---- T5 calibration -------------------------------------------------
    hdr5 = ["Model", "Split", "ECE", "Brier", "Conf (correct)", "Conf (wrong)", "Gate-accepted acc"]
    rows5 = []
    for sp in splits:
        for arm in arms:
            d = D.get((sp, arm))
            if not d:
                continue
            rows5.append([arm, sp,
                          fmt(*agg([v["ece"] for v in d.values()]), pct=False, dp=4),
                          fmt(*agg([v["brier"] for v in d.values()]), pct=False, dp=4),
                          fmt(*agg([v["conf_correct"] for v in d.values()]), pct=False, dp=3),
                          fmt(*agg([v["conf_wrong"] for v in d.values()]), pct=False, dp=3),
                          fmt(*agg([v["gate_accepted_acc"] for v in d.values()]))])
    md.append("## T5 Calibration and confidence gate\n\n" + md_table(hdr5, rows5))
    tex.append(tex_table(hdr5, rows5,
        "Calibration. The confidence gate refuses below 0.40 and warns below 0.60.", "calib"))

    # ---- T2 diagnostic protocol ----------------------------------------
    rows2 = []
    for dpath in sorted(args.diagnostics):
        rep = json.load(open(dpath))
        tag = os.path.basename(dpath).replace("diagnostics_", "").replace(".json", "")
        for ev, sec in rep.get("evalsets", {}).items():
            t1 = sec.get("test1_oracle") or {}
            if t1:
                rows2.append(["T1 oracle", "%s / %s" % (tag, ev),
                              "%.2f" % (100 * t1["best_single_mean"]),
                              "%.2f" % (100 * t1["oracle_mean"]),
                              "%+.2f" % (100 * t1["headroom_mean"]),
                              t1["verdict"].split(" - ")[0]])
            for arm, v in (sec.get("test3_equivalence") or {}).items():
                if arm.startswith("vit"):
                    rows2.append(["T3 equivalence", "%s / %s vs %s" % (tag, ev, arm),
                                  "%+.2f" % (100 * v["mean_delta"]),
                                  "%.3f" % v["min_p"], "%.2f" % (100 * v["arm_seed_std"]),
                                  v["verdict"].split(" - ")[0]])
        if "test2_gate_mass" in rep:
            g = rep["test2_gate_mass"]
            rows2.append(["T2 gate mass", tag, "%.3f" % g["gate_cnn_overall"],
                          "%.3f" % (1 - g["gate_cnn_overall"]), "--",
                          g["verdict"].split(" - ")[0]])
    if rows2:
        hdr2 = ["Test", "Setting", "Value 1", "Value 2", "Value 3", "Verdict"]
        md.append("## T2 Diagnostic protocol\n\n" + md_table(hdr2, rows2))
        tex.append(tex_table(hdr2, rows2,
            "The three-test diagnostic protocol applied to the hybrid architecture.", "diagnostics"))

    open(os.path.join(args.out, "tables.md"), "w").write("\n\n".join(md) + "\n")
    open(os.path.join(args.out, "tables.tex"), "w").write("\n\n".join(tex) + "\n")
    print("\n\n".join(md))
    print("\nwrote %s/tables.{md,tex}" % args.out)


if __name__ == "__main__":
    main()
