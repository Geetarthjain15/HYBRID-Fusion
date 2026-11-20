"""
Fusion diagnostics: the three tests a hybrid-architecture claim must pass.

This is the paper's method contribution. Each test answers one question that
accuracy alone cannot, and each is cheap enough to run BEFORE committing to an
architecture.

  TEST 1 - ORACLE CEILING.  Is there complementary information to fuse at all?
      Take the branch models trained alone. Count images at least one gets
      right. That is the hard upper bound on any fusion of those branches - no
      gating, attention or stacking can exceed it. If the ceiling sits barely
      above the best single branch, the fusion cannot pay for itself and the
      experiment should not be run.

  TEST 2 - GATE MASS.  Does the trained fusion actually use both branches?
      Concatenation fusion lets the optimiser drive one branch's contribution
      to zero; the model then reports as a fusion while behaving as a single
      branch. A gated fusion exposes the mixing weight directly, so branch
      utilisation becomes an observable rather than an assumption.
      (Computed by train_ablation.py at eval time; read from gates_*.csv here.)

  TEST 3 - PAIRED EQUIVALENCE.  Is the gain larger than seed noise?
      Exact McNemar on matched predictions, plus the seed-variance band. A
      difference smaller than the seed standard deviation is not a result.
      Reported as an equivalence statement, not a failed superiority test.

    python src/diagnostics.py --runs runs_original --out results/diagnostics_original.json
"""

import argparse
import csv
import glob
import json
import os
import random
from collections import defaultdict
from math import comb

import numpy as np


# ---------------------------------------------------------------- loading ---

def load_preds(path, keep=None):
    with open(path) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: r["filename"])
    if keep is not None:
        rows = [r for r in rows if r["filename"] in keep]
    classes = [c[2:] for c in rows[0] if c.startswith("p_")]
    files = [r["filename"] for r in rows]
    y = np.array([classes.index(r["true_label"]) for r in rows])
    P = np.array([[float(r["p_" + c]) for c in classes] for r in rows])
    return {"files": files, "y": y, "P": P, "classes": classes,
            "correct": P.argmax(1) == y}


def discover(runs_dir, evalset, keep=None):
    """runs_dir/<arch>_seed<N>[_<tag>]/preds_<evalset>.csv -> {(arch,tag): {seed: data}}"""
    out = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "preds_%s.csv" % evalset))):
        run = os.path.basename(os.path.dirname(p))
        parts = run.split("_seed")
        arch = parts[0]
        rest = parts[1].split("_", 1)
        seed = int(rest[0])
        tag = rest[1] if len(rest) > 1 else ""
        out[(arch, tag)][seed] = load_preds(p, keep)
    return out


# ------------------------------------------------------------ test 1: oracle ---

def oracle_ceiling(members):
    """members: list of {'correct': bool array}. All must cover the same images."""
    C = np.stack([m["correct"] for m in members])
    any_right = C.any(0)
    all_right = C.all(0)
    accs = C.mean(1)
    return {
        "n": int(C.shape[1]),
        "member_acc": [float(a) for a in accs],
        "best_single": float(accs.max()),
        "oracle": float(any_right.mean()),
        "headroom": float(any_right.mean() - accs.max()),
        "all_correct": int(all_right.sum()),
        "none_correct": int((~any_right).sum()),
        "exclusive": [int((C[i] & ~np.delete(C, i, 0).any(0)).sum()) for i in range(len(members))],
    }


# -------------------------------------------------------- test 3: equivalence ---

def mcnemar(a_correct, b_correct):
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    return a_only, b_only, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_delta(a_correct, b_correct, iters=5000, seed=0):
    """Paired bootstrap CI on (b - a) accuracy. Equivalence is claimed when the
    interval is tight and contains zero - a much stronger statement than p>0.05."""
    rng = np.random.default_rng(seed)
    n = len(a_correct)
    idx = rng.integers(0, n, size=(iters, n))
    d = b_correct[idx].mean(1) - a_correct[idx].mean(1)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------- driver ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="e.g. runs_original")
    ap.add_argument("--evalsets", nargs="+", default=["test", "dataset_ood"])
    ap.add_argument("--branches", nargs=2, default=["cnn:unfrozen", "vit:"],
                    help="arch:tag of the two single-branch arms, for the oracle test")
    ap.add_argument("--fusion", default="fusion:", help="arch:tag of the fusion arm")
    ap.add_argument("--gates", default=None, help="glob for gates_*.csv (test 2)")
    ap.add_argument("--subset", default=None,
                    help="File listing filenames to restrict every eval set to "
                         "(e.g. the disjoint cross-source manifest).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def key(spec):
        a, _, t = spec.partition(":")
        return (a, t)

    keep = None
    if args.subset:
        keep = {l.strip() for l in open(args.subset) if l.strip()}
        print("restricting to %d images from %s" % (len(keep), args.subset))
    report = {"runs": args.runs, "subset": args.subset, "evalsets": {}}

    for evalset in args.evalsets:
        groups = discover(args.runs, evalset, keep if evalset != "test" else None)
        if not groups:
            continue
        seeds = sorted(set.intersection(*[set(v) for v in groups.values()]))
        sec = {"seeds": seeds, "arms": {}, "test1_oracle": {}, "test3_equivalence": {}}

        for (arch, tag), per_seed in sorted(groups.items()):
            accs = [per_seed[s]["correct"].mean() for s in seeds]
            sec["arms"]["%s%s" % (arch, "/" + tag if tag else "")] = {
                "acc_mean": float(np.mean(accs)), "n_seeds": len(seeds),
                "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                "acc_per_seed": [float(a) for a in accs],
            }

        # TEST 1 - per seed, then averaged
        b1, b2 = key(args.branches[0]), key(args.branches[1])
        if b1 in groups and b2 in groups:
            per = [oracle_ceiling([groups[b1][s], groups[b2][s]]) for s in seeds]
            sec["test1_oracle"] = {
                "branches": args.branches,
                "per_seed": per,
                "oracle_mean": float(np.mean([p["oracle"] for p in per])),
                "best_single_mean": float(np.mean([p["best_single"] for p in per])),
                "headroom_mean": float(np.mean([p["headroom"] for p in per])),
                "none_correct_mean": float(np.mean([p["none_correct"] for p in per])),
                "verdict": None,
            }
            h = sec["test1_oracle"]["headroom_mean"]
            sec["test1_oracle"]["verdict"] = (
                "NO HEADROOM - branches fail on the same images; fusion cannot pay for itself"
                if h < 0.005 else
                "HEADROOM EXISTS - branches are complementary; a working fusion can gain up to %.2f pts" % (100 * h))

        # TEST 3 - fusion vs every other arm
        fk = key(args.fusion)
        if fk in groups:
            for (arch, tag), per_seed in sorted(groups.items()):
                if (arch, tag) == fk:
                    continue
                name = "%s%s" % (arch, "/" + tag if tag else "")
                rows = []
                for s in seeds:
                    a = per_seed[s]["correct"]
                    b = groups[fk][s]["correct"]
                    a_only, f_only, p = mcnemar(a, b)
                    d, lo, hi = bootstrap_delta(a, b, seed=s)
                    rows.append({"seed": s, "arm_only_right": a_only, "fusion_only_right": f_only,
                                 "p": p, "delta_fusion_minus_arm": d, "ci95": [lo, hi]})
                seed_std = sec["arms"][name]["acc_std"]
                mean_d = float(np.mean([r["delta_fusion_minus_arm"] for r in rows]))
                min_p = float(min(r["p"] for r in rows))
                # Three outcomes, not two. A significant McNemar with a delta
                # inside seed noise means the models make DIFFERENT decisions at
                # the SAME accuracy - they are swapping errors, not improving.
                # Reporting that as "significant" would badly overclaim.
                within_noise = abs(mean_d) <= max(seed_std, 0.002)
                if within_noise and min_p > 0.05:
                    verdict = "EQUIVALENT - same accuracy, same decisions"
                elif within_noise:
                    verdict = "ERROR-SWAPPING - different decisions, accuracy difference within seed noise"
                else:
                    verdict = "DIFFERENT - accuracy difference exceeds seed noise"
                sec["test3_equivalence"][name] = {
                    "per_seed": rows, "mean_delta": mean_d, "min_p": min_p,
                    "arm_seed_std": seed_std, "within_seed_noise": bool(within_noise),
                    "verdict": verdict,
                }
        report["evalsets"][evalset] = sec

    # TEST 2 - gate mass
    if args.gates:
        gate_files = sorted(glob.glob(args.gates))
        g = {}
        for gf in gate_files:
            with open(gf) as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            cnn = np.array([float(r["gate_cnn"]) for r in rows])
            by_class = defaultdict(list)
            for r in rows:
                by_class[r["true_label"]].append(float(r["gate_cnn"]))
            g[os.path.basename(gf)] = {
                "n": len(rows),
                "gate_cnn_mean": float(cnn.mean()), "gate_cnn_std": float(cnn.std()),
                "gate_vit_mean": float(1 - cnn.mean()),
                "frac_images_cnn_dominant": float((cnn > 0.5).mean()),
                "per_class_gate_cnn": {k: float(np.mean(v)) for k, v in sorted(by_class.items())},
            }
        if g:
            allc = [v["gate_cnn_mean"] for v in g.values()]
            report["test2_gate_mass"] = {
                "files": g,
                "gate_cnn_overall": float(np.mean(allc)),
                "verdict": ("COLLAPSED - one branch carries the decision; this is a fusion in name only"
                            if min(allc) < 0.1 or max(allc) > 0.9 else
                            "BOTH BRANCHES USED - gate mass is shared, the fusion is genuinely mixing"),
            }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # ---- console summary -------------------------------------------------
    for evalset, sec in report["evalsets"].items():
        print("\n=== %s / %s ===" % (args.runs, evalset))
        print("  %-18s %8s %8s" % ("arm", "acc", "std"))
        for name, v in sorted(sec["arms"].items()):
            print("  %-18s %7.2f%% %7.2f" % (name, 100 * v["acc_mean"], 100 * v["acc_std"]))
        t1 = sec.get("test1_oracle")
        if t1:
            print("\n  TEST 1 oracle ceiling (%s + %s)" % tuple(args.branches))
            print("    best single %.2f%%  ->  oracle %.2f%%   headroom %+.2f pts"
                  % (100 * t1["best_single_mean"], 100 * t1["oracle_mean"], 100 * t1["headroom_mean"]))
            print("    images no branch gets right: %.1f" % t1["none_correct_mean"])
            print("    %s" % t1["verdict"])
        if sec.get("test3_equivalence"):
            print("\n  TEST 3 paired equivalence (fusion vs arm)")
            for name, v in sorted(sec["test3_equivalence"].items()):
                print("    %-16s delta %+.2f pts | min p %.3f | seed sd %.2f -> %s"
                      % (name, 100 * v["mean_delta"], v["min_p"], 100 * v["arm_seed_std"], v["verdict"]))
    if "test2_gate_mass" in report:
        print("\n  TEST 2 gate mass")
        for fn, v in report["test2_gate_mass"]["files"].items():
            print("    %-28s cnn %.3f / vit %.3f  (cnn-dominant on %.0f%% of images)"
                  % (fn, v["gate_cnn_mean"], v["gate_vit_mean"], 100 * v["frac_images_cnn_dominant"]))
        print("    %s" % report["test2_gate_mass"]["verdict"])
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
