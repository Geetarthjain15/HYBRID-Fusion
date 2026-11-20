"""
Collect every runs/<name>/metrics.json + preds_*.csv into the paper's tables.

Groups runs by (split, arch, variant) and reports mean +/- std over seeds for
accuracy, macro-F1 and per-class recall, plus a paired McNemar test of each
arm against the fusion arm at the same seed. Everything is recomputed from
the prediction CSVs, not from the training-time numbers.

    python src/aggregate_runs.py --runs runs --out results
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from math import comb

import numpy as np


def load_preds(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r["filename"]: (r["true_label"], r["pred_label"]) for r in rows}


def scores(preds, keep=None):
    items = [(t, p) for fn, (t, p) in preds.items() if keep is None or fn in keep]
    classes = sorted({t for t, _ in items})
    acc = np.mean([t == p for t, p in items])
    f1s, rec = [], {}
    for c in classes:
        tp = sum(1 for t, p in items if t == c and p == c)
        fp = sum(1 for t, p in items if t != c and p == c)
        fn = sum(1 for t, p in items if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rec[c] = r
        f1s.append(2 * prec * r / (prec + r) if prec + r else 0.0)
    return {"n": len(items), "acc": acc, "macro_f1": float(np.mean(f1s)), "recall": rec}


def mcnemar(a, b, keep=None):
    """Exact two-sided McNemar: a, b are {filename: (true, pred)} on the same set."""
    keys = [k for k in a if k in b and (keep is None or k in keep)]
    a_only = sum(1 for k in keys if a[k][0] == a[k][1] and b[k][0] != b[k][1])
    b_only = sum(1 for k in keys if b[k][0] == b[k][1] and a[k][0] != a[k][1])
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return a_only, b_only, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--clean_manifest", default=None,
                    help="Optional filename list; also reports metrics on that subset.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    keep = None
    if args.clean_manifest and os.path.exists(args.clean_manifest):
        keep = {l.strip() for l in open(args.clean_manifest) if l.strip()}

    runs = []
    for mpath in sorted(glob.glob(os.path.join(args.runs, "*", "metrics.json"))):
        d = os.path.dirname(mpath)
        m = json.load(open(mpath))
        pp = os.path.join(d, "preds_test.csv")
        if not os.path.exists(pp):
            continue
        variant = "unfrozen" if m.get("unfreeze_all_cnn") else "frozen"
        split = os.path.basename(os.path.dirname(m["train_dir"].rstrip("/\\")))
        runs.append({
            "dir": d, "arch": m["arch"], "seed": m["seed"], "variant": variant,
            "split": split, "params_trainable": m["params_trainable"],
            "epochs_run": m.get("epochs_run"), "minutes": m.get("total_minutes"),
            "preds": load_preds(pp),
            "extra": {os.path.basename(p)[6:-4]: load_preds(p)
                      for p in glob.glob(os.path.join(d, "preds_*.csv"))
                      if not p.endswith("preds_test.csv")},
        })
    if not runs:
        print("no runs with preds_test.csv under", args.runs)
        return

    groups = defaultdict(list)
    for r in runs:
        groups[(r["split"], r["arch"], r["variant"])].append(r)

    table, lines = [], []
    hdr = "{:<14} {:<7} {:<9} {:>5} {:>10} {:>16} {:>16} {:>32}".format(
        "split", "arch", "variant", "seeds", "trainable", "acc", "macro-F1", "recall (gl/me/no/pi)")
    lines += [hdr, "-" * len(hdr)]
    for (split, arch, variant), rs in sorted(groups.items()):
        sc = [scores(r["preds"]) for r in rs]
        acc = np.array([s["acc"] for s in sc]); f1 = np.array([s["macro_f1"] for s in sc])
        classes = sorted(sc[0]["recall"])
        rec = {c: np.array([s["recall"][c] for s in sc]) for c in classes}
        row = {
            "split": split, "arch": arch, "variant": variant, "n_seeds": len(rs),
            "seeds": sorted(r["seed"] for r in rs),
            "params_trainable": rs[0]["params_trainable"],
            "acc_mean": acc.mean(), "acc_std": acc.std(ddof=1) if len(rs) > 1 else 0.0,
            "f1_mean": f1.mean(), "f1_std": f1.std(ddof=1) if len(rs) > 1 else 0.0,
            "recall_mean": {c: rec[c].mean() for c in classes},
            "recall_std": {c: (rec[c].std(ddof=1) if len(rs) > 1 else 0.0) for c in classes},
            "n_test": sc[0]["n"],
        }
        if keep:
            scc = [scores(r["preds"], keep) for r in rs]
            row["clean_n"] = scc[0]["n"]
            row["clean_acc_mean"] = float(np.mean([s["acc"] for s in scc]))
            row["clean_acc_std"] = float(np.std([s["acc"] for s in scc], ddof=1)) if len(rs) > 1 else 0.0
        extras = defaultdict(list)
        for r in rs:
            for name, p in r["extra"].items():
                extras[name].append(scores(p)["acc"])
        row["extra_acc"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                            for k, v in extras.items()}
        table.append(row)
        lines.append("{:<14} {:<7} {:<9} {:>5} {:>10,} {:>7.2f} +- {:<5.2f} {:>7.2f} +- {:<5.2f} {:>32}".format(
            split, arch, variant, len(rs), row["params_trainable"],
            100 * row["acc_mean"], 100 * row["acc_std"], 100 * row["f1_mean"], 100 * row["f1_std"],
            "/".join("{:.1f}".format(100 * row["recall_mean"][c]) for c in classes)))

    # paired tests vs fusion at matching split+seed
    lines += ["", "Paired McNemar vs fusion (same split, same seed):"]
    by_key = {(r["split"], r["arch"], r["variant"], r["seed"]): r for r in runs}
    pairs = []
    for r in runs:
        if r["arch"] == "fusion":
            continue
        f = by_key.get((r["split"], "fusion", "frozen", r["seed"]))
        if f is None:
            continue
        a_only, f_only, p = mcnemar(r["preds"], f["preds"])
        pairs.append({"split": r["split"], "arm": r["arch"] + "/" + r["variant"], "seed": r["seed"],
                      "arm_only_right": a_only, "fusion_only_right": f_only, "p": p})
        lines.append("  {:<14} {:<14} seed {:<4} arm-only-right {:>3}  fusion-only-right {:>3}  p={:.2e}".format(
            r["split"], r["arch"] + "/" + r["variant"], r["seed"], a_only, f_only, p))

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.out, "ablation_table.txt"), "w") as f:
        f.write(txt + "\n")
    with open(os.path.join(args.out, "ablation_table.json"), "w") as f:
        json.dump({"groups": table, "mcnemar_vs_fusion": pairs}, f, indent=2)
    print("\nwrote", os.path.join(args.out, "ablation_table.{txt,json}"))


if __name__ == "__main__":
    main()
