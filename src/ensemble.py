"""
Logit-level fusion over already-trained checkpoints. No training, no GPU.

Feature-concatenation fusion (v1) failed because the optimiser could delete the
weak branch. Probability-level fusion cannot: each member votes with its own
full-strength prediction, and the blend weight is fitted on validation data
rather than learned jointly. On a saturated benchmark this is usually the
single highest-yield thing you can do with models you already have.

Three combiners, each fitted on a held-out split and reported on test:

  mean      unweighted average of member softmaxes
  weighted  per-member weight, grid-searched on the fitting split
  power     weighted geometric mean (log-prob average) - sharper than the
            arithmetic mean when members are confident and well calibrated

    python src/ensemble.py --preds runs_original/cnn_seed42_unfrozen/preds_test.csv \\
                                   runs_original/vit_seed42/preds_test.csv \\
                           --fit_frac 0.4 --out results/ensemble.json
"""

import argparse
import csv
import itertools
import json
import os
import random

import numpy as np


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    classes = [c[2:] for c in rows[0] if c.startswith("p_")]
    files = [r["filename"] for r in rows]
    y = np.array([classes.index(r["true_label"]) for r in rows])
    P = np.array([[float(r["p_" + c]) for c in classes] for r in rows])
    return files, y, P, classes


def acc(P, y):
    return float((P.argmax(1) == y).mean())


def combine(Ps, w, mode):
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    if mode == "power":
        L = sum(wi * np.log(np.clip(P, 1e-12, 1)) for wi, P in zip(w, Ps))
        E = np.exp(L)
        return E / E.sum(1, keepdims=True)
    return sum(wi * P for wi, P in zip(w, Ps))


def search_weights(Ps, y, mode, step=0.1):
    """Grid search over the simplex. With <=4 members this is exhaustive and
    instant; fitted on the fitting split only, never on test."""
    n = len(Ps)
    grid = [g for g in itertools.product(np.arange(0, 1 + step, step), repeat=n)
            if abs(sum(g) - 1.0) < 1e-9 and sum(g) > 0]
    best, best_w = -1, None
    for g in grid:
        a = acc(combine(Ps, g, mode), y)
        if a > best:
            best, best_w = a, g
    return list(best_w), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="Prediction CSVs to ensemble (same images, any order).")
    ap.add_argument("--fit_frac", type=float, default=0.4,
                    help="Fraction of the test set held out to fit blend weights. "
                         "Weights are fitted here and scored on the remainder, so the "
                         "reported number is not fitted on what it reports.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--out", default="results/ensemble.json")
    args = ap.parse_args()

    loaded = [load(p) for p in args.preds]
    ref_files, y, _, classes = loaded[0]
    order = {f: i for i, f in enumerate(ref_files)}
    Ps = []
    for path, (files, yy, P, cc) in zip(args.preds, loaded):
        assert cc == classes, "class order differs in " + path
        idx = [order[f] for f in files]
        Q = np.empty_like(P)
        Q[idx] = P
        Ps.append(Q)
        print("  member {:<50} acc={:.4f}".format(os.path.basename(os.path.dirname(path)), acc(Q, y)))

    n = len(y)
    rng = random.Random(args.seed)
    idx = list(range(n))
    rng.shuffle(idx)
    k = int(args.fit_frac * n)
    fit, hold = np.array(idx[:k]), np.array(idx[k:])

    result = {"members": args.preds, "n_total": n, "n_fit": len(fit), "n_holdout": len(hold),
              "member_acc": [acc(P, y) for P in Ps], "combiners": {}}

    for mode in ("mean", "weighted", "power"):
        if mode == "mean":
            w = [1.0] * len(Ps)
            fit_acc = acc(combine([P[fit] for P in Ps], w, "mean"), y[fit])
        else:
            w, fit_acc = search_weights([P[fit] for P in Ps], y[fit], mode, args.step)
        hold_acc = acc(combine([P[hold] for P in Ps], w, mode), y[hold])
        full_acc = acc(combine(Ps, w, mode), y)
        result["combiners"][mode] = {"weights": [round(x, 3) for x in w],
                                     "fit_acc": fit_acc, "holdout_acc": hold_acc,
                                     "full_acc": full_acc,
                                     "holdout_errors": int(round((1 - hold_acc) * len(hold)))}
        print("  {:<9} w={}  fit={:.4f}  HOLDOUT={:.4f} ({} errors)  full={:.4f}".format(
            mode, [round(x, 2) for x in w], fit_acc, hold_acc,
            result["combiners"][mode]["holdout_errors"], full_acc))

    best = max(result["combiners"], key=lambda m: result["combiners"][m]["holdout_acc"])
    result["best"] = best
    print("\n  best combiner on held-out data: {} -> {:.4f}".format(
        best, result["combiners"][best]["holdout_acc"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("  wrote", args.out)


if __name__ == "__main__":
    main()
