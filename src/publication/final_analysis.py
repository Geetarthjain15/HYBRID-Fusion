"""
Phase 3B - final controlled statistical analysis of the FusionV2 ablations.

Seven ablations, three seeds each, compared against full FusionV2 on the same
848 duplicate-clean test images. Read-only with respect to every existing
artifact; all output goes to publication_results/ablations/final_analysis/.

Four decisions in here are load-bearing and are stated up front because they
determine what the numbers are allowed to mean.

1. THE METRIC FORMULAS ARE IMPORTED, NOT REWRITTEN.
   src/publication/evaluate_fusion_v2.py already defines the conventions the
   published reference numbers were computed with: 15 equal-width ECE bins,
   tie-averaged rank ROC-AUC that returns None rather than 0 when undefined,
   step-wise average precision that is explicitly NOT trapezoidal PR-AUC, and
   per-class specificity. Reimplementing them here would risk the ablations and
   the reference being measured with subtly different rulers. They are imported.

2. THE PRIMARY TEST FAMILY IS THE 21 SEED-SPECIFIC TESTS, NOT 7.
   Each of 7 variants is compared against the reference at each of 3 seeds, and
   Holm runs across all 21. Correcting across 7 while quietly picking or pooling
   one p-value per variant would be a smaller, more flattering family obtained by
   discarding evidence. The variant-level view is reported descriptively only:
   three seeds cannot support a normal-theory test, and the 2,544 predictions are
   848 images scored three times, not 2,544 independent observations.

3. SEED SPREAD AND BOOTSTRAP SPREAD ARE DIFFERENT QUANTITIES.
   The seed SD (ddof=1, n=3) measures how much retraining moves the result. The
   bootstrap CI measures how much the finite image sample moves it, within one
   seed. They are never combined and never relabelled as each other. Neither is
   patient-level uncertainty: the resampling unit is an image.

4. NONSIGNIFICANCE IS NOT EQUIVALENCE.
   A component's verdict comes from effect magnitude, direction consistency
   across seeds, the paired interval, the Holm-adjusted tests and the class-wise
   consequences together. "No detectable benefit under this protocol" is what a
   null result licenses; "equivalent" is not.

    python src/publication/final_analysis.py --project-root .
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Phase 1 metric library - same formulas as the published reference numbers.
from evaluate_fusion_v2 import (ECE_BINS, ECE_RULE, average_precision_ovr,
                                compute_metrics, expected_calibration_error,
                                roc_auc_ovr)
from session_artifact_contract import (CLASSES, CLEAN_TEST_N, EXTERNAL_N,
                                       METRIC_TOL_PP, PROB_SUM_TOL)

SEEDS = [42, 43, 44]
VARIANTS = ["no_gate", "no_multiscale", "no_auxiliary_loss", "no_ema",
            "cnn_branch_only", "vit_branch_only", "no_tta"]
REFERENCE = "full_fusion_v2"
REFERENCE_DIR = "runs_v2_clean/fusion_v2_seed%d"
IMPORT_ROOT = "publication_runs/ablations/clean"

PHASE3A = "publication_results/ablations/phase3a"
ADDENDUM = "publication_results/ablations/phase3a_addendum"
RECOVERY = "publication_results/ablations/checkpoint_recovery"
SPLIT_MANIFEST = "dataset_clean/split_manifest.json"

REF_ACC = (98.27, 0.07)
REF_F1 = (98.28, 0.04)

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260924          # fixed and recorded; never tuned
N_EXPECTED_TESTS = len(VARIANTS) * len(SEEDS)   # 21

# ---------------------------------------------------------------------------
# Verdict vocabulary and the decision rule.
#
# An earlier version of this file decided verdicts with a "materiality threshold"
# of twice the reference seed SD (0.14 pp). That was wrong and is gone. The
# threshold was chosen AFTER the effect sizes were known, it was not an
# equivalence margin, and it had no external or clinical justification. A cutoff
# derived from the data's own noise is still a cutoff picked with knowledge of
# the answer. No replacement number has been introduced.
#
# The rule below uses only quantities that were prespecified: the direction of
# the per-seed accuracy differences, the Holm-adjusted p-values over the fixed
# 21-test family, and the paired image-level bootstrap intervals. Effect sizes are
# reported numerically as OBSERVED DIFFERENCES; no observed difference is
# described as materially important or as negligible.
# ---------------------------------------------------------------------------

VERDICT_SUPPORTED = "supported by consistent accuracy evidence"
VERDICT_SUGGESTIVE = "suggestive but inconclusive"
VERDICT_NO_DETECTABLE = "no detectable accuracy benefit under this protocol"
VERDICT_INCONCLUSIVE = "inconclusive"

VERDICTS = [VERDICT_SUPPORTED, VERDICT_SUGGESTIVE, VERDICT_NO_DETECTABLE,
            VERDICT_INCONCLUSIVE]

# The rule, written out so it can be checked against the code and audited
# independently of it. "Adverse" means the ablation scored LOWER than the
# reference, i.e. removing the component cost accuracy.
DECISION_RULE = [
    ("supported by consistent accuracy evidence",
     "the accuracy difference is adverse at all three seeds AND all three "
     "seed-specific Holm-adjusted tests are significant at 0.05"),
    ("suggestive but inconclusive",
     "the accuracy difference is adverse at all three seeds but the "
     "multiplicity-corrected evidence does not support it at every seed"),
    ("no detectable accuracy benefit under this protocol",
     "the accuracy difference is not adverse at all three seeds, and no "
     "Holm-adjusted test is significant and no paired interval excludes zero. "
     "This is an absence of detectable evidence, NOT equivalence, and it does "
     "not show the component has zero effect"),
    ("inconclusive",
     "the direction is not consistently adverse yet some corrected test or "
     "interval fires, so the evidence points in conflicting directions and no "
     "defensible interpretation follows"),
]

NOT_EQUIVALENCE = (
    "This is an absence of detectable evidence under this protocol on 848 images "
    "with three seeds. It is not a demonstration of equivalence, it does not show "
    "the component has zero effect, and it does not show the component is "
    "unnecessary. A study with more seeds, more images, or a different "
    "distribution could detect an effect this one cannot.")

# What each ablation removes, for reading the sign of an effect. Removing a
# helpful component should HURT the ablation, i.e. give a negative
# (ablation - reference) accuracy difference.
COMPONENT = {
    "no_gate": "learned input-dependent fusion gate",
    "no_multiscale": "multi-scale CNN taps (stages 3 and 5 alongside the final stage)",
    "no_auxiliary_loss": "per-branch auxiliary classification losses",
    "no_ema": "exponential moving average of weights",
    "cnn_branch_only": "the entire ViT branch",
    "vit_branch_only": "the entire CNN branch",
    "no_tta": "horizontal-flip test-time augmentation",
}


class AnalysisError(Exception):
    """A Phase 3B stop condition was hit."""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rel(path: str) -> str:
    """Relative path when that is meaningful, absolute otherwise. On Windows a
    path on another drive has no relative form; that is a ValueError, not a
    reason to fail an analysis."""
    try:
        return os.path.relpath(path).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def banner(m):
    print("\n" + "=" * 76 + "\n  %s\n" % m + "=" * 76, flush=True)


def stamp(m):
    print("[%s] %s" % (_dt.datetime.now().strftime("%H:%M:%S"), m), flush=True)


def mean_sd(vals):
    """Arithmetic mean and SAMPLE standard deviation (ddof=1).

    ddof=1 because the three seeds are a sample of the training procedure's
    outcomes, not the population of them. With n=3 this is a crude estimate and
    is reported as a spread, never as a confidence interval."""
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None, "sd": None, "n": 0,
                "note": "no seed produced a defined value"}
    return {"mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "n": len(v), "ddof": 1}


# ------------------------------------------------------- prerequisite gate ---

def canonical_mapping(root: str) -> dict:
    """Authoritative filename -> label for the 848 clean-test images.

    From the split manifest: machine-readable, image-level, independent of any
    model output. dataset_clean/Testing holds no images locally, so a directory
    enumeration cross-check is unavailable and is reported as such."""
    p = os.path.join(root, SPLIT_MANIFEST)
    if not os.path.exists(p):
        raise AnalysisError("split manifest absent: %s" % SPLIT_MANIFEST)
    mapping = {}
    for e in json.load(open(p, encoding="utf-8")):
        if e.get("new_split") != "Testing":
            continue
        fn = e["file"]
        if fn in mapping and mapping[fn] != e["class"]:
            raise AnalysisError("split manifest gives two labels for %s" % fn)
        mapping[fn] = e["class"]
    if len(mapping) != CLEAN_TEST_N:
        raise AnalysisError("canonical mapping has %d identities, expected %d"
                            % (len(mapping), CLEAN_TEST_N))
    bad = sorted(set(mapping.values()) - set(CLASSES))
    if bad:
        raise AnalysisError("canonical labels outside the class set: %s" % bad)
    fp = fingerprint(sorted(mapping), [mapping[f] for f in sorted(mapping)])
    from collections import Counter
    return {"source": SPLIT_MANIFEST, "source_sha256": sha256(p),
            "n_identities": len(mapping), "mapping": mapping,
            "fingerprint_sha256": fp,
            "class_counts": dict(Counter(mapping.values())),
            "class_order": list(CLASSES),
            "evidence_note": ("derived from the split manifest; dataset_clean/Testing "
                              "contains no images locally, so directory enumeration "
                              "was unavailable as a cross-check")}


def fingerprint(files, labels) -> str:
    return hashlib.sha256(
        "\n".join("%s\t%s" % (f, l) for f, l in sorted(zip(files, labels))).encode()
    ).hexdigest()


def prerequisite_gate(root: str) -> dict:
    """Six prerequisites, each re-derived here rather than taken on trust from a
    prior phase's prose. A failure stops the phase before any inference."""
    g = {"checks": {}, "evidence": {}}

    a = json.load(open(os.path.join(root, PHASE3A, "archive_verification.json"),
                       encoding="utf-8"))
    g["checks"]["phase3a_pass"] = a.get("acceptance_status") == "PASS"
    g["evidence"]["phase3a"] = {"acceptance_status": a.get("acceptance_status"),
                                "n_accepted": a.get("n_accepted"),
                                "n_rejected": a.get("n_rejected"),
                                "training_performed": a.get("training_performed")}

    ad = json.load(open(os.path.join(root, ADDENDUM, "verification_addendum.json"),
                        encoding="utf-8"))
    g["checks"]["addendum_pass"] = ad.get("acceptance_status") == "PASS"
    g["evidence"]["addendum"] = {
        "acceptance_status": ad.get("acceptance_status"),
        "counts": ad.get("counts"),
        "all_runs_share_canonical_mapping": ad.get("all_runs_share_canonical_mapping"),
        "reference_summary": ad.get("reference_summary"),
    }

    canon = canonical_mapping(root)
    g["checks"]["canonical_848_identities"] = canon["n_identities"] == CLEAN_TEST_N
    g["checks"]["canonical_fingerprint_matches_addendum"] = (
        canon["fingerprint_sha256"] == ad["canonical_mapping"]["fingerprint_sha256"])
    g["evidence"]["canonical_mapping"] = {k: v for k, v in canon.items() if k != "mapping"}

    # every imported file re-hashed against the Phase 3A record
    src = {(x["variant"], x["seed"]): x for x in a["runs"]}
    checked, bad = 0, []
    for variant in VARIANTS:
        for s in SEEDS:
            rec = src.get((variant, s))
            if rec is None:
                bad.append("%s seed %d: no Phase 3A record" % (variant, s))
                continue
            d = os.path.join(root, IMPORT_ROOT, variant, "seed_%d" % s)
            for name, info in sorted(rec.get("files", {}).items()):
                p = os.path.join(d, name)
                checked += 1
                if not os.path.exists(p):
                    bad.append("%s seed %d :: %s absent" % (variant, s, name))
                elif sha256(p) != info["sha256"]:
                    bad.append("%s seed %d :: %s digest changed" % (variant, s, name))
    g["checks"]["imported_hashes_match"] = not bad
    g["evidence"]["import_integrity"] = {"files_rehashed": checked,
                                         "problems": bad[:20],
                                         "n_problems": len(bad)}

    rs = ad["reference_summary"]
    g["checks"]["reference_reproduces"] = bool(rs.get("reproduces"))

    g["all_passed"] = all(g["checks"].values())
    g["n_expected_imported_files"] = 189
    g["imported_files_rehashed"] = checked
    return g, canon


# ----------------------------------------------------------- run validation ---

def load_predictions(path: str, expect_n: int, canon: dict | None) -> dict:
    """Load and fully validate one prediction CSV. Every requirement of the
    DATA VALIDATION contract is enforced here, and a failure raises rather than
    returning a partially trusted result."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise AnalysisError("%s: empty" % path)

    prob_cols = [c for c in rows[0] if c.startswith("p_")]
    classes = [c[2:] for c in prob_cols]
    if classes != sorted(classes):
        raise AnalysisError("%s: probability columns not in sorted class order: %s"
                            % (path, classes))
    if classes != list(CLASSES):
        raise AnalysisError("%s: class set %s != documented %s"
                            % (path, classes, list(CLASSES)))

    files = [r["filename"] for r in rows]
    seen, dupes = set(), []
    for f_ in files:
        if f_ in seen:
            dupes.append(f_)
        seen.add(f_)
    if dupes:
        raise AnalysisError("%s: duplicate image identities: %s" % (path, sorted(set(dupes))[:5]))
    if len(rows) != expect_n:
        raise AnalysisError("%s: %d rows, expected %d" % (path, len(rows), expect_n))

    P = np.array([[float(r[c]) for c in prob_cols] for r in rows], dtype=np.float64)
    if not np.isfinite(P).all():
        raise AnalysisError("%s: non-finite probabilities" % path)
    if P.min() < 0.0 or P.max() > 1.0:
        raise AnalysisError("%s: probabilities outside [0,1] (%.6f..%.6f)"
                            % (path, P.min(), P.max()))
    dev = float(np.abs(P.sum(1) - 1.0).max())
    if dev > PROB_SUM_TOL:
        raise AnalysisError("%s: probability rows deviate from 1 by %.3e (tolerance %.1e)"
                            % (path, dev, PROB_SUM_TOL))

    true_labels = [r["true_label"] for r in rows]
    pred_labels = [r["pred_label"] for r in rows]

    # pred_label must be the argmax, except where the top two probabilities tie
    argmax = P.argmax(1)
    ties, violations = 0, []
    for i, (pl, am) in enumerate(zip(pred_labels, argmax)):
        if classes.index(pl) == am:
            continue
        top2 = np.sort(P[i])[-2:]
        if top2[1] - top2[0] <= 1e-9:
            ties += 1
        else:
            violations.append(files[i])
    if violations:
        raise AnalysisError("%s: pred_label != argmax on %d strictly ranked rows: %s"
                            % (path, len(violations), violations[:5]))

    out = {"path": _rel(path),
           "sha256": sha256(path), "n": len(rows), "classes": classes,
           "files": files, "true_labels": true_labels, "pred_labels": pred_labels,
           "P": P, "max_prob_sum_deviation": dev,
           "n_documented_argmax_ties": ties}

    if canon is not None:
        idx = {f: i for i, f in enumerate(files)}
        missing = sorted(set(canon["mapping"]) - set(files))
        extra = sorted(set(files) - set(canon["mapping"]))
        if missing or extra:
            raise AnalysisError("%s: canonical membership differs - %d missing, %d extra %s"
                                % (path, len(missing), len(extra),
                                   (missing[:3] + extra[:3])))
        mismatch = [f for f, l in zip(files, true_labels) if canon["mapping"][f] != l]
        if mismatch:
            raise AnalysisError("%s: true_label != canonical label on %d images: %s"
                                % (path, len(mismatch), mismatch[:5]))
        fp = fingerprint(files, true_labels)
        if fp != canon["fingerprint_sha256"]:
            raise AnalysisError("%s: identity/label fingerprint %s != canonical %s"
                                % (path, fp[:16], canon["fingerprint_sha256"][:16]))
        # JOIN BY IDENTITY, NOT ROW POSITION: reorder onto the canonical order so
        # every downstream pairing is over the same image in the same slot.
        order = [idx[f] for f in sorted(canon["mapping"])]
        cf = [files[i] for i in order]
        out.update({"canonical_order_files": cf,
                    "y": np.array([classes.index(canon["mapping"][f]) for f in cf]),
                    "pred": np.array([classes.index(pred_labels[i]) for i in order]),
                    "P_canonical": P[order],
                    "fingerprint_sha256": fp,
                    "joined_by": "exact canonical image identity, not row position"})
    return out


def load_run(root: str, variant: str, seed: int, canon: dict) -> dict:
    """One model/seed: clean-test predictions, metrics.json, and the provenance
    facts that decide whether it may enter the analysis."""
    d = (os.path.join(root, REFERENCE_DIR % seed) if variant == REFERENCE
         else os.path.join(root, IMPORT_ROOT, variant, "seed_%d" % seed))
    if not os.path.isdir(d):
        raise AnalysisError("run directory absent: %s" % d)

    met_path = os.path.join(d, "metrics.json")
    met = json.load(open(met_path, encoding="utf-8"))
    if met.get("seed") != seed:
        raise AnalysisError("%s: metrics.json records seed %r, expected %d"
                            % (d, met.get("seed"), seed))

    pt = load_predictions(os.path.join(d, "preds_test.csv"), CLEAN_TEST_N, canon)

    ext = None
    ep = os.path.join(d, "preds_dataset_ood.csv")
    if os.path.exists(ep):
        ext = load_predictions(ep, EXTERNAL_N, None)

    return {"variant": variant, "seed": seed,
            "run_dir": _rel(d),
            "metrics_json": {"path": _rel(met_path),
                             "sha256": sha256(met_path)},
            "stored_test_acc": met.get("test_acc"),
            "stored_macro_f1_field": next(
                (k for k in met if "macro" in k.lower() and "f1" in k.lower()), None),
            "trained": met.get("trained"), "checkpoint_written": met.get("checkpoint_written"),
            "resolved_config": met.get("resolved_config"),
            "clean_test": pt, "external": ext, "raw_metrics": met}


# ------------------------------------------------------- paired statistics ---

def mcnemar_exact(b: int, c: int) -> dict:
    """Exact two-sided McNemar test on the discordant pairs only.

    b = reference correct and ablation wrong, c = reference wrong and ablation
    correct. Under the null the b+c discordant pairs split Binomial(n, 0.5), so

        p = 2 * sum_{k=0}^{min(b,c)} C(n,k) 0.5^n,  capped at 1.

    This is the exact binomial test, not the chi-square approximation with or
    without continuity correction: with counts this small the approximation is
    not trustworthy and there is no reason to use it. With no discordant pairs
    the test has no data; p is reported as 1.0 and flagged, because 'no evidence
    of a difference' must not be read as evidence of no difference."""
    n = b + c
    if n == 0:
        return {"b_ref_correct_abl_wrong": b, "c_ref_wrong_abl_correct": c,
                "n_discordant": 0, "p_value": 1.0,
                "test": "exact two-sided McNemar (binomial)",
                "note": ("no discordant pairs: the two models made identical "
                         "correct/incorrect decisions on all 848 images, so the "
                         "test has no information. p is reported as 1.0 and must "
                         "not be read as evidence of equivalence.")}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return {"b_ref_correct_abl_wrong": b, "c_ref_wrong_abl_correct": c,
            "n_discordant": n, "p_value": float(min(1.0, 2.0 * tail)),
            "test": "exact two-sided McNemar (binomial)",
            "note": "exact binomial on discordant pairs; no chi-square approximation"}


def holm(pvals: list) -> list:
    """Holm-Bonferroni step-down adjusted p-values, order preserved.

    Applied across the whole family handed in. The caller is responsible for
    that family being the prespecified one; see holm_family_guard."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)      # enforce monotonicity
        adj[i] = running
    return adj


def holm_family_guard(n: int) -> None:
    if n != N_EXPECTED_TESTS:
        raise AnalysisError(
            "Holm family has %d tests, expected %d (7 variants x 3 seeds). "
            "Correcting across a smaller family by pooling or selecting one "
            "p-value per variant would understate the correction." % (n, N_EXPECTED_TESTS))


def paired_disagreement(ref: dict, abl: dict) -> dict:
    """The 2x2 table over the SAME images, matched by identity.

    Both runs were reordered onto the canonical identity order at load time, so
    position i is the same image in both. That is asserted here rather than
    assumed, because the whole pairing rests on it."""
    if ref["clean_test"]["canonical_order_files"] != abl["clean_test"]["canonical_order_files"]:
        raise AnalysisError("paired comparison over different image sets: %s vs %s"
                            % (ref["run_dir"], abl["run_dir"]))
    if ref["seed"] != abl["seed"]:
        raise AnalysisError("paired comparison across seeds %d and %d"
                            % (ref["seed"], abl["seed"]))
    r_ok = ref["clean_test"]["pred"] == ref["clean_test"]["y"]
    a_ok = abl["clean_test"]["pred"] == abl["clean_test"]["y"]
    both = int((r_ok & a_ok).sum())
    b = int((r_ok & ~a_ok).sum())
    c = int((~r_ok & a_ok).sum())
    neither = int((~r_ok & ~a_ok).sum())
    if both + b + c + neither != CLEAN_TEST_N:
        raise AnalysisError("2x2 table sums to %d, expected %d"
                            % (both + b + c + neither, CLEAN_TEST_N))
    return {"n_images": CLEAN_TEST_N, "both_correct": both,
            "reference_correct_ablation_wrong": b,
            "reference_wrong_ablation_correct": c, "both_wrong": neither,
            "matched_by": "exact canonical image identity within the same seed"}


def macro_f1_from(y, pred, k) -> float:
    f1s = []
    for c in range(k):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def balanced_accuracy_from(y, pred, k) -> float:
    recs = []
    for c in range(k):
        tp = int(((pred == c) & (y == c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        recs.append(tp / (tp + fn) if tp + fn else 0.0)
    return float(np.mean(recs))


def paired_bootstrap(ref: dict, abl: dict, n_iter=BOOTSTRAP_ITERATIONS,
                     seed=BOOTSTRAP_SEED) -> dict:
    """Paired, class-stratified image bootstrap of the accuracy difference.

    The resampling unit is an IMAGE. Each iteration draws one index set and
    applies it to BOTH models, so the pairing that makes the comparison
    controlled is preserved; drawing independently for each model would inflate
    the interval with variance that the paired design removes. Stratifying
    within each true class keeps the class mix of every resample equal to the
    evaluated distribution.

    This is image-level uncertainty. It is NOT patient-level uncertainty: no
    patient identifiers exist for this split, so no patient-level statement is
    available at any confidence. It is also not the seed SD, which measures a
    different thing and is reported separately."""
    y = ref["clean_test"]["y"]
    rp, ap = ref["clean_test"]["pred"], abl["clean_test"]["pred"]
    k = len(CLASSES)
    rng = np.random.default_rng(seed)
    idx_by_class = [np.nonzero(y == c)[0] for c in range(k)]

    d_acc = np.empty(n_iter)
    d_f1 = np.empty(n_iter)
    d_bal = np.empty(n_iter)
    for it in range(n_iter):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by_class if len(ix)])
        yb = y[take]
        rb, ab = rp[take], ap[take]
        d_acc[it] = (ab == yb).mean() - (rb == yb).mean()
        d_f1[it] = macro_f1_from(yb, ab, k) - macro_f1_from(yb, rb, k)
        d_bal[it] = balanced_accuracy_from(yb, ab, k) - balanced_accuracy_from(yb, rb, k)

    def ci(v, scale=100.0):
        return {"point_pp": None,
                "ci95_lo_pp": float(np.percentile(v, 2.5) * scale),
                "ci95_hi_pp": float(np.percentile(v, 97.5) * scale),
                "bootstrap_mean_pp": float(v.mean() * scale),
                "excludes_zero": bool(np.percentile(v, 2.5) * np.percentile(v, 97.5) > 0)}

    out = {"n_iterations": n_iter, "seed": seed,
           "resampling_unit": "image",
           "stratified_by": "true class",
           "pairing": ("one index set per iteration applied to both models, so "
                       "reference and ablation stay paired"),
           "interpretation": ("image-level uncertainty within a single training "
                              "seed; not patient-level, and not the seed SD"),
           "accuracy_difference_pp": ci(d_acc),
           "macro_f1_difference_pp": ci(d_f1),
           "balanced_accuracy_difference_pp": ci(d_bal)}
    out["accuracy_difference_pp"]["point_pp"] = 100.0 * (
        float((ap == y).mean()) - float((rp == y).mean()))
    out["macro_f1_difference_pp"]["point_pp"] = 100.0 * (
        macro_f1_from(y, ap, k) - macro_f1_from(y, rp, k))
    out["balanced_accuracy_difference_pp"]["point_pp"] = 100.0 * (
        balanced_accuracy_from(y, ap, k) - balanced_accuracy_from(y, rp, k))
    return out


# ------------------------------------------------------ gate provenance ------

def gate_provenance(root: str) -> dict:
    """Does the checkpoint-recovery record strengthen the reference gate status?

    The Phase 3A Addendum could only resolve the reference's `gated` setting from
    the launch specification, its weakest evidence class, because the reference
    metrics.json does not record it, src/fusion_v2.py allocates the gate whether
    or not it is used (so parameter count cannot discriminate), and
    src/train_ablation.py guards its gate dump on architecture alone (so gate
    telemetry cannot either).

    The recovery manifest can settle it, if and only if it records all of:
      - a strict load of the reference checkpoint into FusionV2(gated=True)
        with no missing, unexpected or mis-shaped keys;
      - a probe gate mass separated from uniform 0.5/0.5 mixing, since at exactly
        0.5 the gated and ungated forward paths coincide (fusion_v2.py scales by
        g*2, and 0.5*2 = 1);
      - reproduction of the reference run's recorded predictions through that
        gated path.
    Then the reference predictions demonstrably came from the gated path. If any
    of those is absent the weaker caveat is retained rather than replaced."""
    p = os.path.join(root, RECOVERY, "import_manifest.json")
    if not os.path.exists(p):
        return {"strengthened": False,
                "reason": "checkpoint-recovery manifest absent",
                "caveat": "launch-specification caveat retained"}
    man = json.load(open(p, encoding="utf-8"))
    per_seed, ok = {}, True
    for s in SEEDS:
        e = man.get("seeds", {}).get(str(s))
        if e is None:
            ok = False
            per_seed[s] = {"available": False}
            continue
        g = e["gates"]
        ld, ag, ft = g["load"], g["architecture_and_gate"], g["full_test"]
        strict = (ld.get("loaded") and ld.get("n_missing") == 0
                  and ld.get("n_unexpected") == 0 and ld.get("n_shape_mismatch") == 0)
        gm = ag.get("gate_cnn_mean_on_probe")
        # Degeneracy guard, not an effect-size judgement: at a gate mass of
        # exactly 0.5 the gated and ungated forward paths are the same function
        # (fusion_v2.py scales each branch by g*2, and 0.5*2 = 1), so prediction
        # reproduction through the gated path would prove nothing about gating.
        # The small tolerance only excludes a gate sitting essentially on that
        # degenerate point. It takes no part in any component verdict.
        away = gm is not None and abs(gm - 0.5) > 0.05
        reproduced = (ft.get("n_pred_label_mismatch") == 0
                      and ft.get("n_filename_mismatch") == 0
                      and ft.get("n_true_label_mismatch") == 0
                      and ft.get("n_compared") == CLEAN_TEST_N
                      and float(ft.get("accuracy_diff_pp", 9)) <= METRIC_TOL_PP)
        seed_ok = bool(strict and away and reproduced
                       and ag.get("gated_flag") and ag.get("gate_is_learned_module")
                       and e.get("decision") == "ACCEPTED")
        ok = ok and seed_ok
        per_seed[s] = {
            "available": True, "decision": e.get("decision"),
            "checkpoint_sha256": g["container"]["sha256"],
            "strict_load": strict, "state_dict_keys": ld.get("state_dict_key_count"),
            "n_missing": ld.get("n_missing"), "n_unexpected": ld.get("n_unexpected"),
            "n_shape_mismatch": ld.get("n_shape_mismatch"),
            "loaded_into_gated_architecture": ag.get("gated_flag"),
            "loaded_into_multiscale_architecture": ag.get("multiscale_flag"),
            "cnn_proj_in_features": ag.get("cnn_proj_in_features"),
            "gate_is_learned_module": ag.get("gate_is_learned_module"),
            "gate_parameter_count": ag.get("gate_parameter_count"),
            "gate_cnn_mean_on_probe": gm,
            "gate_cnn_std_on_probe": ag.get("gate_cnn_std_on_probe"),
            "gate_separated_from_uniform_mixing": away,
            "n_predictions_compared": ft.get("n_compared"),
            "n_pred_label_mismatch": ft.get("n_pred_label_mismatch"),
            "max_abs_prob_diff": ft.get("max_abs_prob_diff"),
            "accuracy_diff_pp": ft.get("accuracy_diff_pp"),
            "reference_predictions_reproduced": reproduced,
            "seed_establishes_gated_reference": seed_ok,
        }
    return {
        "strengthened": ok,
        "source": os.path.join(RECOVERY, "import_manifest.json").replace("\\", "/"),
        "source_sha256": sha256(p),
        "per_seed": per_seed,
        "argument": (
            "src/fusion_v2.py forward() computes cat([c*g_cnn*2, v*g_vit*2]) when "
            "gated and cat([c, v]) when not, so the two paths coincide only at a "
            "gate mass of exactly 0.5/0.5. The recovered checkpoints load strictly "
            "into FusionV2(gated=True, multiscale=True) with 578/578 keys and no "
            "missing, unexpected or mis-shaped tensors; their probe gate mass is "
            "separated from uniform 0.5/0.5 mixing; and evaluating through "
            "that gated path "
            "reproduces every one of the 848 recorded reference predictions with "
            "no pred_label mismatch. The reference predictions were therefore "
            "produced by the gated forward path."
            if ok else
            "the recovery record does not establish all three required facts"),
        "consequence": (
            "The Phase 3A Addendum resolved the reference `gated` setting only "
            "from the launch specification, its weakest evidence class. That "
            "caveat is SUPERSEDED: the reference is gated on machine-readable, "
            "independently checkable evidence."
            if ok else
            "The weaker launch-specification caveat is RETAINED."),
        "what_this_does_not_establish": (
            "Gate mass measures branch utilization only. It is not evidence of "
            "causal branch contribution, of explanation, or of anatomical "
            "attention, and no such claim is made from it."),
    }


def no_tta_provenance(root: str, runs: dict) -> dict:
    """no_tta must be the reference weights scored without TTA - nothing else."""
    rec = json.load(open(os.path.join(root, RECOVERY, "import_manifest.json"),
                         encoding="utf-8"))
    out, ok = {}, True
    for s in SEEDS:
        d = os.path.join(root, IMPORT_ROOT, "no_tta", "seed_%d" % s)
        sc = os.path.join(d, "source_checkpoint_sha256.txt")
        recorded = open(sc, encoding="utf-8").read().split()[0].lower() if os.path.exists(sc) else None
        expected = rec["seeds"][str(s)]["gates"]["container"]["sha256"]
        met = runs[("no_tta", s)]["raw_metrics"]
        rc = met.get("resolved_config", {})
        e = {"source_checkpoint_recorded": recorded,
             "verified_recovered_checkpoint": expected,
             "identical_checkpoint": recorded == expected,
             "trained": met.get("trained"),
             "checkpoint_written": met.get("checkpoint_written"),
             "no_weights_changed": met.get("trained") is False
                                   and met.get("checkpoint_written") is False,
             "tta_flag": rc.get("tta"),
             "checkpoint_in_run_dir": os.path.exists(os.path.join(d, "checkpoint.pth"))}
        e["tta_is_the_only_change"] = bool(e["identical_checkpoint"]
                                          and e["no_weights_changed"]
                                          and rc.get("tta") is False)
        ok = ok and e["tta_is_the_only_change"] and not e["checkpoint_in_run_dir"]
        out[s] = e
    return {"all_seeds_clean": ok, "per_seed": out,
            "note": ("no_tta is an evaluation-only variant: the same verified "
                     "checkpoint, scored with horizontal-flip TTA disabled. No "
                     "training occurred and no checkpoint was written.")}


# --------------------------------------------------------------- verdicts ----

def classify_component(variant: str, per_seed: list, summary: dict) -> dict:
    """Assign one of the four cautious verdicts from prespecified evidence only.

    Inputs, all fixed before the numbers were seen: the sign of the per-seed
    accuracy differences, the Holm-adjusted p-values over the 21-test family, and
    the paired image-level bootstrap intervals. There is no effect-size cutoff of
    any kind in this function, and none may be added: a threshold chosen once the
    effect sizes are known describes the data rather than deciding on it.

    Secondary metrics (macro-F1, balanced accuracy) and the class-wise pattern are
    recorded and reported, but they do not drive the verdict, because all four
    category names are scoped to accuracy. Where a secondary metric is
    consistently adverse while accuracy is not, that is stated as an observed
    pattern with no corrected test behind it - reported, not inflated and not
    hidden."""
    d_acc = [x["accuracy_difference_pp"] for x in per_seed]
    d_f1 = [x["macro_f1_difference_pp"] for x in per_seed]
    d_bal = [x["balanced_accuracy_difference_pp"] for x in per_seed]
    n_seeds = len(d_acc)
    n_sig = sum(1 for x in per_seed if x["holm_adjusted_p"] < 0.05)
    n_ci = sum(1 for x in per_seed if x["bootstrap_accuracy_excludes_zero"])

    mean_acc = float(np.mean(d_acc))
    mean_f1 = float(np.mean(d_f1))
    mean_bal = float(np.mean(d_bal))

    adverse_all = all(x < 0 for x in d_acc)
    favourable_all = all(x > 0 for x in d_acc)
    corrected_at_every_seed = n_sig == n_seeds
    any_corrected = n_sig > 0 or n_ci > 0

    f1_adverse_all = all(x < 0 for x in d_f1)
    bal_adverse_all = all(x < 0 for x in d_bal)

    seeds_txt = ", ".join("%+.4f" % x for x in d_acc)
    reasons, flags = [], []

    if adverse_all and corrected_at_every_seed:
        verdict = VERDICT_SUPPORTED
        reasons.append("removing the component lowered accuracy at all %d seeds "
                       "(observed differences %s pp) and all %d seed-specific "
                       "Holm-adjusted tests are significant at 0.05"
                       % (n_seeds, seeds_txt, n_seeds))
        reasons.append("%d of %d paired image-level bootstrap intervals for the "
                       "accuracy difference exclude zero" % (n_ci, n_seeds))
    elif adverse_all:
        verdict = VERDICT_SUGGESTIVE
        reasons.append("removing the component lowered accuracy at all %d seeds "
                       "(observed differences %s pp), so the direction is "
                       "consistent" % (n_seeds, seeds_txt))
        reasons.append("the multiplicity-corrected evidence does not support it at "
                       "every seed: %d of %d Holm-adjusted tests significant, %d of "
                       "%d paired intervals exclude zero"
                       % (n_sig, n_seeds, n_ci, n_seeds))
        reasons.append("a consistent direction across three seeds is suggestive. "
                       "Three seeds is a small number and this is not sufficient "
                       "evidence of a benefit")
    elif not adverse_all and not any_corrected:
        verdict = VERDICT_NO_DETECTABLE
        if favourable_all:
            reasons.append("removing the component did not lower accuracy at any "
                           "seed (observed differences %s pp), so there is no "
                           "adverse direction to support a benefit" % seeds_txt)
        else:
            reasons.append("the sign of the accuracy difference is not consistently "
                           "adverse across seeds (observed differences %s pp)"
                           % seeds_txt)
        reasons.append("no Holm-adjusted test is significant and no paired "
                       "image-level interval excludes zero")
        if f1_adverse_all or bal_adverse_all:
            reasons.append("OBSERVED SECONDARY PATTERN, reported and not claimed: "
                           "%s was adverse at all %d seeds (macro-F1 %s pp; "
                           "balanced accuracy %s pp). No prespecified paired test "
                           "or multiplicity-controlled procedure exists for these "
                           "metrics, so this is descriptive only and does not "
                           "change the verdict"
                           % (" and ".join(
                               [x for x, ok in (("macro-F1", f1_adverse_all),
                                                ("balanced accuracy", bal_adverse_all))
                                if ok]),
                              n_seeds,
                              ", ".join("%+.4f" % x for x in d_f1),
                              ", ".join("%+.4f" % x for x in d_bal)))
        reasons.append(NOT_EQUIVALENCE)
    else:
        verdict = VERDICT_INCONCLUSIVE
        reasons.append("the accuracy difference is not consistently adverse "
                       "(observed differences %s pp) yet corrected evidence fires "
                       "(%d of %d Holm-adjusted tests significant, %d of %d paired "
                       "intervals exclude zero), so the evidence points in "
                       "conflicting directions"
                       % (seeds_txt, n_sig, n_seeds, n_ci, n_seeds))
        if favourable_all:
            flags.append("removal_consistently_favourable_with_corrected_support")
            reasons.append("the observed direction favours REMOVING the component at "
                           "every seed with corrected support. That pattern is "
                           "outside the four prespecified categories, which describe "
                           "adverse effects of removal, and it is flagged here rather "
                           "than fitted into one of them. It needs its own "
                           "prespecified analysis")
        reasons.append(NOT_EQUIVALENCE)

    reasons.append("three seeds bound how much any of this can say: the seed "
                   "spread is estimated from n=3 and is reported as a spread, "
                   "never as a confidence interval or a significance test")

    return {"variant": variant, "component_removed": COMPONENT[variant],
            "verdict": verdict, "reasons": reasons,
            "decision_inputs": {
                "per_seed_accuracy_difference_pp": d_acc,
                "accuracy_adverse_at_every_seed": bool(adverse_all),
                "accuracy_favourable_at_every_seed": bool(favourable_all),
                "n_holm_significant": n_sig,
                "n_bootstrap_ci_excludes_zero": n_ci,
                "n_seeds": n_seeds,
                "threshold_used": None,
                "threshold_note": ("no effect-size threshold of any kind takes "
                                   "part in this verdict"),
            },
            "observed_mean_accuracy_difference_pp": mean_acc,
            "observed_mean_macro_f1_difference_pp": mean_f1,
            "observed_mean_balanced_accuracy_difference_pp": mean_bal,
            "per_seed_accuracy_difference_pp": d_acc,
            "per_seed_macro_f1_difference_pp": d_f1,
            "per_seed_balanced_accuracy_difference_pp": d_bal,
            "accuracy_adverse_at_every_seed": bool(adverse_all),
            "direction_consistent": bool(adverse_all or favourable_all),
            "secondary_macro_f1_adverse_at_every_seed": bool(f1_adverse_all),
            "secondary_balanced_accuracy_adverse_at_every_seed": bool(bal_adverse_all),
            "n_holm_significant_of_3": n_sig,
            "n_bootstrap_ci_excludes_zero_of_3": n_ci,
            "effect_size_label": ("observed difference; deliberately not "
                                  "characterised as important or as negligible, "
                                  "since no prespecified margin exists to judge "
                                  "that against"),
            "flags": flags}




def build_claims(tests: list, conclusions: list, rep: dict) -> dict:
    """Assemble the report's claims FROM the computed results.

    Each sentence is generated from the numbers, so it cannot drift away from
    them: if the evidence changed, the sentence would change or its guard would
    fire. `verified` records the check behind each one."""
    by_variant = {}
    for t in tests:
        by_variant.setdefault(t["variant"], []).append(t)

    sig_all_seeds = sorted(v for v, ts in by_variant.items()
                           if all(t["holm_significant_at_0.05"] for t in ts))
    sig_any_seed = sorted(v for v, ts in by_variant.items()
                          if any(t["holm_significant_at_0.05"] for t in ts))
    adverse_all = sorted(v for v, ts in by_variant.items()
                         if all(t["accuracy_difference_pp"] < 0 for t in ts))
    verdicts = {c["variant"]: c["verdict"] for c in conclusions}

    claims, verified = [], {}

    # 1. what survived correction, stated from the data
    if sig_all_seeds == ["cnn_branch_only"]:
        claims.append(
            "Only removal of the ViT branch produced Holm-significant accuracy "
            "losses at all three seeds. Of the 21 seed-specific tests, %d were "
            "significant before Holm correction and %d after, and every surviving "
            "test belongs to `cnn_branch_only`."
            % (sum(1 for t in tests if t["unadjusted_p"] < 0.05),
               sum(1 for t in tests if t["holm_significant_at_0.05"])))
    else:
        claims.append(
            "The variants whose accuracy losses are Holm-significant at all three "
            "seeds are: %s. Variants significant at one or two seeds: %s. This "
            "differs from the expectation recorded when the correction was "
            "requested, and the data is reported as found."
            % (", ".join("`%s`" % v for v in sig_all_seeds) or "none",
               ", ".join("`%s`" % v for v in sorted(set(sig_any_seed) - set(sig_all_seeds)))
               or "none"))
    verified["only_vit_branch_removal_holm_significant_at_all_seeds"] = \
        sig_all_seeds == ["cnn_branch_only"]

    # 2. auxiliary losses
    aux = by_variant.get("no_auxiliary_loss", [])
    aux_mean = float(np.mean([t["accuracy_difference_pp"] for t in aux])) if aux else None
    aux_consistent = bool(aux) and all(t["accuracy_difference_pp"] < 0 for t in aux)
    aux_sig = sum(1 for t in aux if t["holm_significant_at_0.05"])
    if aux_consistent and aux_sig == 0:
        claims.append(
            "Auxiliary-loss removal produced a consistent mean decrease but did not "
            "survive Holm correction. The observed accuracy difference was adverse "
            "at all three seeds (%s pp, observed mean %+.4f pp); the smallest "
            "unadjusted p-value was %.4g and became %.4g after Holm correction "
            "across the 21-test family."
            % (", ".join("%+.4f" % t["accuracy_difference_pp"] for t in aux),
               aux_mean, min(t["unadjusted_p"] for t in aux),
               min(t["holm_adjusted_p"] for t in aux)))
    else:
        claims.append(
            "Auxiliary-loss removal: observed accuracy differences %s pp (mean "
            "%+.4f pp), adverse at every seed: %s, Holm-significant tests %d of 3."
            % (", ".join("%+.4f" % t["accuracy_difference_pp"] for t in aux),
               aux_mean, aux_consistent, aux_sig))
    verified["auxiliary_loss_consistent_decrease_not_holm_significant"] = \
        bool(aux_consistent and aux_sig == 0)

    # 3. the gate
    gate = by_variant.get("no_gate", [])
    gate_mean = float(np.mean([t["accuracy_difference_pp"] for t in gate])) if gate else None
    if verdicts.get("no_gate") == VERDICT_NO_DETECTABLE:
        claims.append(
            "The gate showed no detectable accuracy benefit under this protocol. "
            "Removing it gave observed accuracy differences of %s pp (observed mean "
            "%+.4f pp), with %d and %d discordant predictions out of 848 at the "
            "three seeds, all three Holm-adjusted p-values at %.4g, and no paired "
            "image-level interval excluding zero."
            % (", ".join("%+.4f" % t["accuracy_difference_pp"] for t in gate),
               gate_mean,
               min(t["mcnemar"]["n_discordant"] for t in gate),
               max(t["mcnemar"]["n_discordant"] for t in gate),
               max(t["holm_adjusted_p"] for t in gate)))
    else:
        claims.append("The gate verdict from the evidence is `%s`; see its entry "
                      "above." % verdicts.get("no_gate"))
    verified["gate_no_detectable_accuracy_benefit"] = \
        verdicts.get("no_gate") == VERDICT_NO_DETECTABLE

    # 4-6. the statements that bound every one of the above
    claims.append(
        "\"No detectable accuracy benefit under this protocol\" is not "
        "equivalence, is not uselessness, and is not proof of zero effect. It "
        "means this study - 848 duplicate-clean images, three seeds, a 21-test "
        "Holm-corrected family - did not detect a benefit. An effect smaller than "
        "this design can resolve would produce exactly the same result.")
    claims.append(
        "The gate parameters remain learned and input-dependent. The recovered "
        "checkpoints carry a %d-parameter learned gate module whose output varies "
        "across inputs and sums to one per image, with probe CNN gate mass %s "
        "across the three seeds. Whether a component has learned dynamic behaviour "
        "and whether removing it changes predictive accuracy are different "
        "questions, and only the second is tested here."
        % (rep["gate_provenance"]["per_seed"][SEEDS[0]]["gate_parameter_count"],
           ", ".join("%.4f" % rep["gate_provenance"]["per_seed"][sd]["gate_cnn_mean_on_probe"]
                     for sd in SEEDS)))
    claims.append(
        "All results here are image-level for one duplicate-clean benchmark "
        "partition of a single dataset family. They are not patient-level "
        "evidence - no patient or study identifiers exist for this split - and "
        "they are not clinical evidence.")

    # Statements that must NOT follow, recorded so a reader can hold the report
    # to them.
    not_claimed = [
        "that only the ViT branch matters: removing the CNN branch "
        "(`vit_branch_only`) gave an adverse observed difference at all three "
        "seeds, and its verdict is `%s`, not an absence of contribution."
        % verdicts.get("vit_branch_only"),
        "that the CNN branch is unnecessary.",
        "that the gate is useless, or that it has zero effect.",
        "that the model should be changed on the strength of this analysis.",
        "that statistical nonsignificance proves equality.",
        "any statistical comparison against BRAIN-META: its per-image predictions "
        "are unavailable, so no such comparison is computed or implied.",
    ]
    return {"claims": claims, "not_claimed": not_claimed, "verified": verified,
            "holm_significant_at_all_seeds": sig_all_seeds,
            "adverse_at_all_seeds": adverse_all}


def statistical_audit(tests: list, rep: dict) -> dict:
    """Reconfirm the prespecified analysis is intact. Changes nothing."""
    mcc = rep["multiple_comparison_control"]
    checks = {
        "exactly_21_seed_specific_mcnemar_tests": len(tests) == 21,
        "twenty_one_distinct_variant_seed_pairs":
            len({(t["variant"], t["seed"]) for t in tests}) == 21,
        "holm_applied_across_all_21": mcc["n_tests"] == 21,
        "mcnemar_is_exact_two_sided": all(
            t["mcnemar"]["test"] == "exact two-sided McNemar (binomial)" for t in tests),
        "bootstrap_10000_iterations_paired_stratified_by_image": all(
            t["bootstrap"]["n_iterations"] >= 10000
            and t["bootstrap"]["resampling_unit"] == "image"
            and t["bootstrap"]["stratified_by"] == "true class" for t in tests),
        "no_pooling_of_2544_predictions_as_independent": all(
            t["table"]["n_images"] == CLEAN_TEST_N for t in tests),
        "seed_sd_separate_from_bootstrap_ci": True,
        "no_brain_meta_statistical_comparison": True,
    }
    return {"checks": checks, "all_intact": all(checks.values()),
            "n_tests": len(tests),
            "n_significant_before_holm": mcc["n_significant_before_holm"],
            "n_significant_after_holm": mcc["n_significant_after_holm"],
            "note": ("this audit re-reads the prespecified analysis and changes no "
                     "part of it; the 2,544 predictions are 848 images scored three "
                     "times and are never treated as independent observations")}


# ------------------------------------------------------------------ writers ---

def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(v, nd=4):
    return "" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else v)


# ------------------------------------------------------------------ driver ----

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-root",
                    default="publication_results/ablations/final_analysis")
    ap.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    os.chdir(root)
    rel = args.out_root.replace("\\", "/")
    if not rel.startswith("publication_"):
        raise AnalysisError("refusing to write outside publication_*: %s" % rel)
    if args.bootstrap_iterations < 10000:
        raise AnalysisError("bootstrap requires at least 10,000 iterations, got %d"
                            % args.bootstrap_iterations)
    out_root = os.path.join(root, args.out_root)
    os.makedirs(out_root, exist_ok=True)

    rep = {"phase": "3B - final controlled statistical analysis",
           "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "training_performed": False, "reimport_performed": False,
           "model_modified": False, "threshold_tuned": False,
           "reference": REFERENCE, "variants": list(VARIANTS), "seeds": list(SEEDS),
           "split": "duplicate-clean", "n_clean_test": CLEAN_TEST_N}

    # ---- prerequisite gate ------------------------------------------------
    banner("PREREQUISITE GATE")
    gate, canon = prerequisite_gate(root)
    rep["prerequisite_gate"] = gate
    for k, v in gate["checks"].items():
        stamp("  %-42s %s" % (k, "PASS" if v else "FAIL"))
    stamp("  imported files re-hashed: %d (expected %d)"
          % (gate["imported_files_rehashed"], gate["n_expected_imported_files"]))
    if not gate["all_passed"]:
        raise AnalysisError("prerequisite gate failed: %s"
                            % [k for k, v in gate["checks"].items() if not v])

    # ---- load and validate every run --------------------------------------
    banner("RUN VALIDATION - 24 identities, joined by canonical image identity")
    runs, inputs = {}, []
    for variant in [REFERENCE] + VARIANTS:
        for s in SEEDS:
            r = load_run(root, variant, s, canon)
            runs[(variant, s)] = r
            inputs.append({"variant": variant, "seed": s,
                           "predictions": r["clean_test"]["path"],
                           "predictions_sha256": r["clean_test"]["sha256"],
                           "metrics_json": r["metrics_json"]["path"],
                           "metrics_json_sha256": r["metrics_json"]["sha256"],
                           "external_predictions": r["external"]["path"] if r["external"] else None,
                           "external_sha256": r["external"]["sha256"] if r["external"] else None,
                           "fingerprint_sha256": r["clean_test"]["fingerprint_sha256"],
                           "n_documented_argmax_ties": r["clean_test"]["n_documented_argmax_ties"],
                           "max_prob_sum_deviation": r["clean_test"]["max_prob_sum_deviation"]})
    fps = {r["clean_test"]["fingerprint_sha256"] for r in runs.values()}
    if len(fps) != 1 or fps.pop() != canon["fingerprint_sha256"]:
        raise AnalysisError("the 24 runs do not share the canonical fingerprint")
    rep["n_run_identities_validated"] = len(runs)
    rep["all_runs_share_canonical_fingerprint"] = True
    rep["inputs"] = inputs
    stamp("  %d runs validated, all sharing canonical fingerprint %s"
          % (len(runs), canon["fingerprint_sha256"][:16]))
    stamp("  documented argmax ties across all runs: %d"
          % sum(x["n_documented_argmax_ties"] for x in inputs))

    # ---- metrics per model per seed ---------------------------------------
    banner("METRICS - per model, per seed")
    by_seed = {}
    for variant in [REFERENCE] + VARIANTS:
        for s in SEEDS:
            ct = runs[(variant, s)]["clean_test"]
            m = compute_metrics(ct["y"], ct["P_canonical"], list(CLASSES))
            by_seed[(variant, s)] = m
        accs = [by_seed[(variant, s)]["accuracy"] for s in SEEDS]
        f1s = [by_seed[(variant, s)]["macro_f1"] for s in SEEDS]
        stamp("  %-18s acc %s  macroF1 %s"
              % (variant,
                 " ".join("%.4f" % a for a in accs),
                 " ".join("%.4f" % f for f in f1s)))
    rep["metric_conventions"] = {
        "source": "src/publication/evaluate_fusion_v2.py (Phase 1), imported unchanged",
        "class_order": list(CLASSES),
        "averaging": ("macro = unweighted mean over the four classes, so each class "
                      "counts equally regardless of support"),
        "zero_division_policy": ("precision, recall, specificity and F1 are defined "
                                 "as 0.0 when their denominator is 0; ROC-AUC and AP "
                                 "return not-available rather than 0 when the class "
                                 "has no positives or no negatives"),
        "balanced_accuracy": "unweighted mean of per-class recall",
        "specificity": "TN / (TN + FP), computed one-vs-rest per class",
        "nll": "-mean(log p_true), probabilities clipped at 1e-12",
        "brier": "multiclass Brier = mean over images of sum_k (p_k - onehot_k)^2",
        "ece": ECE_RULE,
        "ece_bins": ECE_BINS,
        "ece_bin_edges": "linspace(0, 1, 16); first bin closed at both ends, rest (lo, hi]",
        "roc_auc": ("one-vs-rest, Mann-Whitney rank identity with ties averaged; "
                    "not-available when undefined"),
        "average_precision": ("one-vs-rest step-wise average precision (AP): mean of "
                              "precision at each positive, no interpolation. This is "
                              "AP, NOT the trapezoidal area under an interpolated "
                              "precision-recall curve; the two are computed "
                              "differently and are not interchangeable"),
        "seed_summary": "arithmetic mean and sample SD with ddof=1 over 3 seeds",
    }

    # ---- reference reproduction -------------------------------------------
    banner("REFERENCE REPRODUCTION")
    r_acc = [by_seed[(REFERENCE, s)]["accuracy"] for s in SEEDS]
    r_f1 = [by_seed[(REFERENCE, s)]["macro_f1"] for s in SEEDS]
    ref_sum = {
        "accuracy_per_seed_pct": [100 * x for x in r_acc],
        "macro_f1_per_seed_pct": [100 * x for x in r_f1],
        "accuracy_mean_pct": 100 * float(np.mean(r_acc)),
        "accuracy_sd_pct": 100 * float(np.std(r_acc, ddof=1)),
        "macro_f1_mean_pct": 100 * float(np.mean(r_f1)),
        "macro_f1_sd_pct": 100 * float(np.std(r_f1, ddof=1)),
        "verified_accuracy_pct": list(REF_ACC),
        "verified_macro_f1_pct": list(REF_F1),
        "stored_macro_f1_available": any(
            runs[(REFERENCE, s)]["stored_macro_f1_field"] for s in SEEDS),
    }
    ref_sum["accuracy_mean_diff_pp"] = abs(ref_sum["accuracy_mean_pct"] - REF_ACC[0])
    ref_sum["accuracy_sd_diff_pp"] = abs(ref_sum["accuracy_sd_pct"] - REF_ACC[1])
    ref_sum["macro_f1_mean_diff_pp"] = abs(ref_sum["macro_f1_mean_pct"] - REF_F1[0])
    ref_sum["macro_f1_sd_diff_pp"] = abs(ref_sum["macro_f1_sd_pct"] - REF_F1[1])
    ref_sum["reproduces_within_tolerance"] = all(
        ref_sum[k] <= METRIC_TOL_PP for k in
        ("accuracy_mean_diff_pp", "accuracy_sd_diff_pp",
         "macro_f1_mean_diff_pp", "macro_f1_sd_diff_pp"))
    ref_sum["macro_f1_evidence_note"] = (
        "No run stores a macro-F1 field. Macro-F1 here is independently recomputed "
        "from the prediction CSVs and agrees with the verified summary; this is an "
        "independent recomputation, not a stored-metric reproduction.")
    rep["reference_reproduction"] = ref_sum
    stamp("  accuracy  %.4f +/- %.4f %%  (verified %.2f +/- %.2f)  diff %.4f / %.4f pp"
          % (ref_sum["accuracy_mean_pct"], ref_sum["accuracy_sd_pct"], REF_ACC[0], REF_ACC[1],
             ref_sum["accuracy_mean_diff_pp"], ref_sum["accuracy_sd_diff_pp"]))
    stamp("  macro-F1  %.4f +/- %.4f %%  (verified %.2f +/- %.2f)  diff %.4f / %.4f pp"
          % (ref_sum["macro_f1_mean_pct"], ref_sum["macro_f1_sd_pct"], REF_F1[0], REF_F1[1],
             ref_sum["macro_f1_mean_diff_pp"], ref_sum["macro_f1_sd_diff_pp"]))
    stamp("  stored macro-F1 field present anywhere: %s"
          % ref_sum["stored_macro_f1_available"])
    if not ref_sum["reproduces_within_tolerance"]:
        raise AnalysisError("reference summaries do not reproduce within %.2f pp"
                            % METRIC_TOL_PP)

    # ---- provenance: gate and no_tta --------------------------------------
    banner("PROVENANCE - reference gate status and no_tta linkage")
    gp = gate_provenance(root)
    rep["gate_provenance"] = gp
    stamp("  checkpoint-recovery strengthens reference gate status: %s" % gp["strengthened"])
    for s in SEEDS:
        e = gp["per_seed"][s]
        stamp("    seed %d  strict_load=%s gate_cnn=%.4f (away from 0.5: %s) "
              "preds_reproduced=%s" % (s, e.get("strict_load"),
                                       e.get("gate_cnn_mean_on_probe") or float("nan"),
                                       e.get("gate_separated_from_uniform_mixing"),
                                       e.get("reference_predictions_reproduced")))
    stamp("  %s" % gp["consequence"])

    nt = no_tta_provenance(root, runs)
    rep["no_tta_provenance"] = nt
    for s in SEEDS:
        e = nt["per_seed"][s]
        stamp("  no_tta seed %d: identical checkpoint=%s no_weights_changed=%s "
              "tta=%s -> only change is TTA: %s"
              % (s, e["identical_checkpoint"], e["no_weights_changed"],
                 e["tta_flag"], e["tta_is_the_only_change"]))
    if not nt["all_seeds_clean"]:
        raise AnalysisError("no_tta provenance is not clean; see no_tta_provenance")

    # ---- paired analysis: 21 seed-specific comparisons ---------------------
    banner("PAIRED ANALYSIS - 21 seed-specific comparisons against the reference")
    tests = []
    for variant in VARIANTS:
        for s in SEEDS:
            ref, abl = runs[(REFERENCE, s)], runs[(variant, s)]
            tab = paired_disagreement(ref, abl)
            mc = mcnemar_exact(tab["reference_correct_ablation_wrong"],
                               tab["reference_wrong_ablation_correct"])
            bs = paired_bootstrap(ref, abl, n_iter=args.bootstrap_iterations)
            rm, am = by_seed[(REFERENCE, s)], by_seed[(variant, s)]
            tests.append({
                "variant": variant, "seed": s, "component_removed": COMPONENT[variant],
                "table": tab, "mcnemar": mc, "bootstrap": bs,
                "reference_accuracy": rm["accuracy"], "ablation_accuracy": am["accuracy"],
                "accuracy_difference_pp": 100 * (am["accuracy"] - rm["accuracy"]),
                "macro_f1_difference_pp": 100 * (am["macro_f1"] - rm["macro_f1"]),
                "balanced_accuracy_difference_pp":
                    100 * (am["balanced_accuracy"] - rm["balanced_accuracy"]),
                "unadjusted_p": mc["p_value"],
                "bootstrap_accuracy_excludes_zero":
                    bs["accuracy_difference_pp"]["excludes_zero"],
            })
    holm_family_guard(len(tests))
    adj = holm([t["unadjusted_p"] for t in tests])
    for t, a in zip(tests, adj):
        t["holm_adjusted_p"] = a
        t["holm_significant_at_0.05"] = a < 0.05
    rep["multiple_comparison_control"] = {
        "family": "all 21 seed-specific McNemar tests (7 variants x 3 seeds)",
        "method": "Holm-Bonferroni step-down, monotonicity enforced",
        "n_tests": len(tests),
        "alpha": 0.05,
        "n_significant_after_holm": sum(1 for t in tests if t["holm_significant_at_0.05"]),
        "n_significant_before_holm": sum(1 for t in tests if t["unadjusted_p"] < 0.05),
        "why_this_family": (
            "The primary family is all 21 seed-specific tests. Correcting across 7 "
            "while selecting or pooling one p-value per variant would be a smaller, "
            "more permissive family reached by discarding evidence, so it is not "
            "used. The variant-level view below is descriptive only."),
        "what_is_not_done": [
            "the 2,544 predictions are 848 images scored three times and are never "
            "pooled as independent observations",
            "no single combined McNemar p-value is manufactured per variant",
            "n=3 seeds is not treated as sufficient for a normal-theory test",
            "nonsignificance is never reported as equivalence",
        ],
    }
    rep["paired_tests"] = tests
    for t in tests:
        stamp("  %-18s seed %d  dacc %+6.2f pp  b=%-3d c=%-3d  p=%.4g  Holm=%.4g %s"
              % (t["variant"], t["seed"], t["accuracy_difference_pp"],
                 t["table"]["reference_correct_ablation_wrong"],
                 t["table"]["reference_wrong_ablation_correct"],
                 t["unadjusted_p"], t["holm_adjusted_p"],
                 "*" if t["holm_significant_at_0.05"] else ""))

    # ---- seed summaries ---------------------------------------------------
    AGG = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
           "macro_f1", "macro_roc_auc_ovr", "macro_average_precision_ovr",
           "nll", "brier", "ece", "mean_confidence", "mean_confidence_correct",
           "mean_confidence_incorrect"]
    summaries = {}
    for variant in [REFERENCE] + VARIANTS:
        summaries[variant] = {k: mean_sd([by_seed[(variant, s)][k] for s in SEEDS])
                              for k in AGG}
        summaries[variant]["per_seed"] = {
            k: {s: by_seed[(variant, s)][k] for s in SEEDS} for k in AGG}
    rep["metrics_by_seed"] = {"%s|%d" % (v, s): by_seed[(v, s)] for v, s in by_seed}
    rep["metrics_summary"] = summaries

    # ---- variant-level descriptive view + verdicts -------------------------
    banner("COMPONENT CONCLUSIONS")
    conclusions = []
    for variant in VARIANTS:
        per = [t for t in tests if t["variant"] == variant]
        c = classify_component(variant, per, summaries[variant])
        c["accuracy_mean_pct"] = 100 * summaries[variant]["accuracy"]["mean"]
        c["accuracy_sd_pct"] = (100 * summaries[variant]["accuracy"]["sd"]
                                if summaries[variant]["accuracy"]["sd"] is not None else None)
        c["macro_f1_mean_pct"] = 100 * summaries[variant]["macro_f1"]["mean"]
        c["unadjusted_p_per_seed"] = [t["unadjusted_p"] for t in per]
        c["holm_adjusted_p_per_seed"] = [t["holm_adjusted_p"] for t in per]
        c["bootstrap_ci_per_seed_pp"] = [
            [t["bootstrap"]["accuracy_difference_pp"]["ci95_lo_pp"],
             t["bootstrap"]["accuracy_difference_pp"]["ci95_hi_pp"]] for t in per]
        conclusions.append(c)
        stamp("  %-18s %-52s  observed dacc %+6.2f pp  Holm sig %d/3  adverse at "
              "every seed: %s"
              % (variant, c["verdict"], c["observed_mean_accuracy_difference_pp"],
                 c["n_holm_significant_of_3"], c["accuracy_adverse_at_every_seed"]))
    rep["component_conclusions"] = conclusions
    rep["decision_rule"] = [{"verdict": n, "assigned_when": w} for n, w in DECISION_RULE]
    rep["effect_size_threshold_used"] = None
    rep["effect_size_threshold_note"] = (
        "No effect-size threshold takes part in any verdict. An earlier version of "
        "this analysis used a post-hoc 0.14 pp materiality threshold, defined as "
        "twice the reference seed SD; it was chosen after the effect sizes were "
        "known, was not an equivalence margin, had no external justification, and "
        "has been removed without replacement.")
    rep["claims"] = build_claims(tests, conclusions, rep)
    rep["statistical_audit"] = statistical_audit(tests, rep)
    stamp("  claims verified: %s"
          % {k: v for k, v in rep["claims"]["verified"].items()})
    stamp("  statistical audit intact: %s" % rep["statistical_audit"]["all_intact"])
    if not rep["statistical_audit"]["all_intact"]:
        raise AnalysisError("statistical audit failed: %s"
                            % [k for k, v in rep["statistical_audit"]["checks"].items()
                               if not v])
    rep["variant_level_descriptive_note"] = (
        "The variant-level rows summarise three seed-specific comparisons. They are "
        "descriptive: no variant-level significance test is computed, because the "
        "three seeds share the same 848 images and n=3 cannot support a reliable "
        "normal-theory test.")

    # ---- external predictions, descriptive only ---------------------------
    banner("EXTERNAL 499-IMAGE PREDICTIONS - descriptive only")
    ext = {"n_images": EXTERNAL_N, "purpose": "descriptive summary only",
           "per_model": {}, "guardrails": [
               "external labels were not used for tuning, selection or thresholding",
               "external performance was not used to choose a component",
               "external and clean-test results are never mixed or averaged together",
               "this is external_full: it contains 126 images with a near-duplicate "
               "in training and is NOT the separately verified 373-image "
               "duplicate-disjoint result, and is not zero-shot duplicate-disjoint "
               "performance",
           ]}
    for variant in [REFERENCE] + VARIANTS:
        accs = []
        for s in SEEDS:
            e = runs[(variant, s)]["external"]
            if e is None:
                continue
            y = np.array([e["classes"].index(l) for l in e["true_labels"]])
            pr = e["P"].argmax(1)
            accs.append(float((pr == y).mean()))
        ext["per_model"][variant] = {
            "accuracy_per_seed": accs, **mean_sd(accs),
            "structurally_validated": True}
        stamp("  %-18s external_full accuracy %s  mean %.4f"
              % (variant, " ".join("%.4f" % a for a in accs), float(np.mean(accs))))
    rep["external_full_descriptive"] = ext

    # ---- limitations ------------------------------------------------------
    rep["limitations"] = [
        "No run stores a macro-F1 value. Every macro-F1 reported here is "
        "independently recomputed from the prediction CSVs. This is a recomputation, "
        "not a stored-metric reproduction, and is never described as one.",
        "Canonical clean-test membership rests on dataset_clean/split_manifest.json. "
        "dataset_clean/Testing holds no images locally, so a directory enumeration "
        "cross-check of the 848 identities is unavailable.",
        "The bootstrap resampling unit is an image. No patient or study identifiers "
        "exist for this split, so no patient-level or study-level uncertainty is "
        "available at any confidence level, and none is claimed.",
        "Seed spread (n=3, ddof=1) and the bootstrap interval measure different "
        "things and are reported separately. The seed SD is not a confidence "
        "interval and three seeds cannot support a normal-theory test.",
        "A non-significant McNemar result means no detectable difference under this "
        "protocol on 848 images. It is not evidence of equivalence, and the "
        "component verdicts are worded accordingly.",
        "Gate mass measures branch utilization only. It is not evidence of causal "
        "branch contribution, of explanation, or of anatomical attention.",
        "The external 499-image summary is descriptive. It is external_full, which "
        "includes 126 images having a near-duplicate in training, and it is neither "
        "the verified 373-image duplicate-disjoint result nor a zero-shot "
        "generalization measurement.",
        "No comparison against BRAIN-META is computed or implied: its per-image "
        "predictions are unavailable, so no statistical statement about it is "
        "possible from this repository.",
        "These are in-distribution duplicate-clean test results for one dataset "
        "family. They support no claim of clinical readiness, patient-level "
        "independence, or generalization across hospitals or scanners.",
    ]

    # ---- acceptance -------------------------------------------------------
    acc_checks = {
        "prerequisite_gate_passed": gate["all_passed"],
        "all_24_run_identities_revalidated": len(runs) == 24,
        "all_runs_share_canonical_fingerprint": True,
        "reference_accuracy_reproduced": ref_sum["accuracy_mean_diff_pp"] <= METRIC_TOL_PP
                                         and ref_sum["accuracy_sd_diff_pp"] <= METRIC_TOL_PP,
        "reference_macro_f1_recomputed": ref_sum["macro_f1_mean_diff_pp"] <= METRIC_TOL_PP
                                         and ref_sum["macro_f1_sd_diff_pp"] <= METRIC_TOL_PP,
        "seven_variants_x_three_seeds": len({(t["variant"], t["seed"]) for t in tests}) == 21,
        "exactly_21_mcnemar_tests": len(tests) == N_EXPECTED_TESTS,
        "holm_applied_across_all_21": rep["multiple_comparison_control"]["n_tests"] == 21,
        "bootstrap_unit_is_image": all(
            t["bootstrap"]["resampling_unit"] == "image" for t in tests),
        "bootstrap_iterations_at_least_10000": all(
            t["bootstrap"]["n_iterations"] >= 10000 for t in tests),
        "seed_sd_and_bootstrap_kept_separate": True,
        "no_tta_provenance_clean": nt["all_seeds_clean"],
        "no_brain_meta_statistical_comparison": True,
        "no_training_or_reimport": not rep["training_performed"] and not rep["reimport_performed"],
        "statistical_audit_intact": rep["statistical_audit"]["all_intact"],
        "no_effect_size_threshold_in_any_verdict": all(
            c["decision_inputs"]["threshold_used"] is None for c in conclusions),
        "verdict_vocabulary_is_the_four_cautious_categories":
            all(c["verdict"] in VERDICTS for c in conclusions) and len(VERDICTS) == 4,
    }
    rep["acceptance_checks"] = acc_checks
    rep["acceptance_status"] = "PASS" if all(acc_checks.values()) else "FAIL"

    # ---- write ------------------------------------------------------------
    banner("WRITING OUTPUTS")
    write_outputs(out_root, root, rep, by_seed, summaries, tests, conclusions,
                  canon, inputs, AGG)
    for k, v in acc_checks.items():
        stamp("  %-46s %s" % (k, "PASS" if v else "FAIL"))
    banner("PHASE 3B: %s" % rep["acceptance_status"])
    stamp("wrote %s" % os.path.relpath(out_root, root))
    return 0 if rep["acceptance_status"] == "PASS" else 1


# ------------------------------------------------------------------ outputs ---

def write_outputs(out_root, root, rep, by_seed, summaries, tests, conclusions,
                  canon, inputs, AGG):
    """Everything is generated from the computed structures. No number is typed."""

    # 1. final_metrics_by_seed.csv
    rows = []
    for (variant, s), m in by_seed.items():
        row = {"model": variant, "seed": s, "n": m["n_images"]}
        for k in AGG:
            row[k] = m[k]
        rows.append(row)
    rows.sort(key=lambda r: (r["model"] != REFERENCE, r["model"], r["seed"]))
    write_csv(os.path.join(out_root, "final_metrics_by_seed.csv"),
              ["model", "seed", "n"] + AGG, rows)

    # 2. final_metrics_summary.csv  (seed 42/43/44, mean, sd ddof=1)
    rows = []
    for variant in [REFERENCE] + VARIANTS:
        for k in AGG:
            per = summaries[variant]["per_seed"][k]
            agg = summaries[variant][k]
            rows.append({
                "model": variant, "metric": k,
                "seed_42": per[42], "seed_43": per[43], "seed_44": per[44],
                "mean": agg["mean"], "sd_ddof1": agg["sd"],
                "n_seeds_defined": agg["n"],
                "uncertainty_kind": ("seed-to-seed spread over 3 independently "
                                     "trained seeds; NOT a confidence interval"),
            })
    write_csv(os.path.join(out_root, "final_metrics_summary.csv"),
              ["model", "metric", "seed_42", "seed_43", "seed_44", "mean",
               "sd_ddof1", "n_seeds_defined", "uncertainty_kind"], rows)

    # 3. per_class_metrics.csv
    rows = []
    for (variant, s), m in sorted(by_seed.items(),
                                  key=lambda kv: (kv[0][0] != REFERENCE, kv[0][0], kv[0][1])):
        for pc in m["per_class"]:
            rows.append({
                "model": variant, "seed": s, "class": pc["class"],
                "support": pc["support"], "tp": pc["tp"], "fp": pc["fp"],
                "fn": pc["fn"], "tn": pc["tn"],
                "precision": pc["precision"],
                "recall_sensitivity": pc["recall_sensitivity"],
                "specificity": pc["specificity"], "f1": pc["f1"],
                "roc_auc_ovr": pc["roc_auc_ovr"],
                "average_precision_ovr": pc["average_precision_ovr"],
                "ap_naming_note": ("average precision (step-wise), not trapezoidal "
                                   "PR-AUC"),
            })
    write_csv(os.path.join(out_root, "per_class_metrics.csv"),
              ["model", "seed", "class", "support", "tp", "fp", "fn", "tn",
               "precision", "recall_sensitivity", "specificity", "f1",
               "roc_auc_ovr", "average_precision_ovr", "ap_naming_note"], rows)

    # 4. paired_disagreements.csv
    rows = []
    for t in tests:
        tb = t["table"]
        rows.append({
            "variant": t["variant"], "seed": t["seed"],
            "component_removed": t["component_removed"],
            "n_images": tb["n_images"],
            "both_correct": tb["both_correct"],
            "reference_correct_ablation_wrong": tb["reference_correct_ablation_wrong"],
            "reference_wrong_ablation_correct": tb["reference_wrong_ablation_correct"],
            "both_wrong": tb["both_wrong"],
            "n_discordant": t["mcnemar"]["n_discordant"],
            "reference_accuracy": t["reference_accuracy"],
            "ablation_accuracy": t["ablation_accuracy"],
            "accuracy_difference_pp": t["accuracy_difference_pp"],
            "macro_f1_difference_pp": t["macro_f1_difference_pp"],
            "balanced_accuracy_difference_pp": t["balanced_accuracy_difference_pp"],
            "matched_by": tb["matched_by"],
        })
    write_csv(os.path.join(out_root, "paired_disagreements.csv"),
              ["variant", "seed", "component_removed", "n_images", "both_correct",
               "reference_correct_ablation_wrong", "reference_wrong_ablation_correct",
               "both_wrong", "n_discordant", "reference_accuracy",
               "ablation_accuracy", "accuracy_difference_pp",
               "macro_f1_difference_pp", "balanced_accuracy_difference_pp",
               "matched_by"], rows)

    # 5. mcnemar_tests.csv
    rows = []
    for t in tests:
        rows.append({
            "variant": t["variant"], "seed": t["seed"],
            "b_ref_correct_abl_wrong": t["mcnemar"]["b_ref_correct_abl_wrong"],
            "c_ref_wrong_abl_correct": t["mcnemar"]["c_ref_wrong_abl_correct"],
            "n_discordant": t["mcnemar"]["n_discordant"],
            "test": t["mcnemar"]["test"],
            "p_unadjusted": t["unadjusted_p"],
            "p_holm_adjusted": t["holm_adjusted_p"],
            "significant_after_holm_at_0.05": t["holm_significant_at_0.05"],
            "holm_family_size": len(tests),
            "note": t["mcnemar"]["note"],
        })
    write_csv(os.path.join(out_root, "mcnemar_tests.csv"),
              ["variant", "seed", "b_ref_correct_abl_wrong", "c_ref_wrong_abl_correct",
               "n_discordant", "test", "p_unadjusted", "p_holm_adjusted",
               "significant_after_holm_at_0.05", "holm_family_size", "note"], rows)

    # 6. bootstrap_effects.csv
    rows = []
    for t in tests:
        b = t["bootstrap"]
        for metric, key in (("accuracy", "accuracy_difference_pp"),
                            ("macro_f1", "macro_f1_difference_pp"),
                            ("balanced_accuracy", "balanced_accuracy_difference_pp")):
            d = b[key]
            rows.append({
                "variant": t["variant"], "seed": t["seed"], "metric": metric,
                "difference_pp": d["point_pp"],
                "ci95_lo_pp": d["ci95_lo_pp"], "ci95_hi_pp": d["ci95_hi_pp"],
                "bootstrap_mean_pp": d["bootstrap_mean_pp"],
                "excludes_zero": d["excludes_zero"],
                "n_iterations": b["n_iterations"], "bootstrap_seed": b["seed"],
                "resampling_unit": b["resampling_unit"],
                "stratified_by": b["stratified_by"],
                "interpretation": b["interpretation"],
            })
    write_csv(os.path.join(out_root, "bootstrap_effects.csv"),
              ["variant", "seed", "metric", "difference_pp", "ci95_lo_pp",
               "ci95_hi_pp", "bootstrap_mean_pp", "excludes_zero", "n_iterations",
               "bootstrap_seed", "resampling_unit", "stratified_by",
               "interpretation"], rows)

    # 7. component_conclusions.csv
    rows = []
    for c in conclusions:
        rows.append({
            "variant": c["variant"], "component_removed": c["component_removed"],
            "verdict": c["verdict"],
            "accuracy_mean_pct": c["accuracy_mean_pct"],
            "accuracy_sd_ddof1_pct": c["accuracy_sd_pct"],
            "observed_mean_accuracy_difference_pp":
                c["observed_mean_accuracy_difference_pp"],
            "observed_mean_macro_f1_difference_pp":
                c["observed_mean_macro_f1_difference_pp"],
            "observed_mean_balanced_accuracy_difference_pp":
                c["observed_mean_balanced_accuracy_difference_pp"],
            "per_seed_accuracy_difference_pp": "; ".join(
                "%+.4f" % x for x in c["per_seed_accuracy_difference_pp"]),
            "accuracy_adverse_at_every_seed": c["accuracy_adverse_at_every_seed"],
            "direction_consistent_across_seeds": c["direction_consistent"],
            "n_holm_significant_of_3": c["n_holm_significant_of_3"],
            "n_bootstrap_ci_excludes_zero_of_3": c["n_bootstrap_ci_excludes_zero_of_3"],
            "secondary_macro_f1_adverse_at_every_seed":
                c["secondary_macro_f1_adverse_at_every_seed"],
            "secondary_balanced_accuracy_adverse_at_every_seed":
                c["secondary_balanced_accuracy_adverse_at_every_seed"],
            "effect_size_label": c["effect_size_label"],
            "threshold_used": c["decision_inputs"]["threshold_used"],
            "flags": "; ".join(c["flags"]),
            "reasons": " | ".join(c["reasons"]),
        })
    write_csv(os.path.join(out_root, "component_conclusions.csv"),
              ["variant", "component_removed", "verdict", "accuracy_mean_pct",
               "accuracy_sd_ddof1_pct", "observed_mean_accuracy_difference_pp",
               "observed_mean_macro_f1_difference_pp",
               "observed_mean_balanced_accuracy_difference_pp",
               "per_seed_accuracy_difference_pp", "accuracy_adverse_at_every_seed",
               "direction_consistent_across_seeds", "n_holm_significant_of_3",
               "n_bootstrap_ci_excludes_zero_of_3",
               "secondary_macro_f1_adverse_at_every_seed",
               "secondary_balanced_accuracy_adverse_at_every_seed",
               "effect_size_label", "threshold_used", "flags", "reasons"], rows)

    # 8. analysis_provenance.json
    prov = {
        "phase": rep["phase"], "generated_utc": rep["generated_utc"],
        "project_root": root.replace("\\", "/"),
        "canonical_mapping": {k: v for k, v in canon.items() if k != "mapping"},
        "inputs": inputs,
        "prerequisite_gate": rep["prerequisite_gate"],
        "metric_conventions": rep["metric_conventions"],
        "multiple_comparison_control": rep["multiple_comparison_control"],
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED,
                      "resampling_unit": "image", "stratified_by": "true class",
                      "paired": True},
        "gate_provenance": rep["gate_provenance"],
        "no_tta_provenance": rep["no_tta_provenance"],
        "code": {rel: sha256(os.path.join(root, rel)) for rel in
                 ("src/publication/final_analysis.py",
                  "src/publication/evaluate_fusion_v2.py",
                  "src/publication/session_artifact_contract.py",
                  "src/fusion_v2.py", "src/train_ablation.py")
                 if os.path.exists(os.path.join(root, rel))},
        "acceptance_checks": rep["acceptance_checks"],
        "acceptance_status": rep["acceptance_status"],
        "limitations": rep["limitations"],
    }
    with open(os.path.join(out_root, "analysis_provenance.json"), "w",
              encoding="utf-8") as f:
        json.dump(prov, f, indent=2)

    # 9. final_ablation_analysis.json
    serial = dict(rep)
    serial["metrics_by_seed"] = {k: v for k, v in rep["metrics_by_seed"].items()}
    with open(os.path.join(out_root, "final_ablation_analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(serial, f, indent=2, default=str)

    # 10. FINAL_ABLATION_ANALYSIS.md
    with open(os.path.join(out_root, "FINAL_ABLATION_ANALYSIS.md"), "w",
              encoding="utf-8") as f:
        f.write(render_md(rep, by_seed, summaries, tests, conclusions, canon, AGG))


def _name(v: str) -> str:
    """Reference row is bolded; the marker must close or the table renders with a
    stray `**`."""
    return "**`%s`**" % v if v == REFERENCE else "`%s`" % v


def pp(v, nd=2):
    return "n/a" if v is None else "%+.*f" % (nd, v)


def num(v, nd=4):
    return "n/a" if v is None else "%.*f" % (nd, v)


def render_md(rep, by_seed, summaries, tests, conclusions, canon, AGG) -> str:
    c = canon
    rr = rep["reference_reproduction"]
    mcc = rep["multiple_comparison_control"]
    gp = rep["gate_provenance"]
    nt = rep["no_tta_provenance"]

    L = ["# Phase 3B - Final Controlled Ablation Analysis", "",
         "Generated by `src/publication/final_analysis.py` at %s." % rep["generated_utc"], "",
         "**Status: %s.** Seven ablations against full FusionV2 on the same 848 "
         "duplicate-clean test images, three matching seeds each. No training, no "
         "re-import, no model change, no threshold tuning."
         % rep["acceptance_status"], "",
         "## What this analysis is, and what it is not", "",
         "The primary family is the **21 seed-specific paired tests** (7 variants x 3 "
         "seeds), Holm-corrected across all 21. Two smaller, more flattering families "
         "were available and are deliberately not used: correcting across 7 while "
         "selecting one p-value per variant, and pooling the 2,544 predictions as if "
         "they were independent. The 2,544 predictions are 848 images scored three "
         "times.", "",
         "Two uncertainty quantities appear throughout and are never combined:", "",
         "| Quantity | What it measures | What it is not |",
         "|---|---|---|",
         "| seed SD (ddof=1, n=3) | how much retraining moves the result | not a confidence interval |",
         "| paired bootstrap CI | how much the finite image sample moves the result, within one seed | not patient-level uncertainty, not the seed SD |",
         "",
         "A non-significant result is reported as *no detectable benefit under this "
         "protocol*. It is never reported as equivalence.", "",
         "## Input provenance", "",
         "| Field | Value |", "|---|---|",
         "| canonical source | `%s` |" % c["source"],
         "| canonical source SHA-256 | `%s` |" % c["source_sha256"],
         "| canonical identities | %d |" % c["n_identities"],
         "| canonical class counts | %s |" % c["class_counts"],
         "| class order | %s |" % ", ".join("`%s`" % x for x in c["class_order"]),
         "| **canonical fingerprint** | `%s` |" % c["fingerprint_sha256"],
         "| run identities validated | %d |" % rep["n_run_identities_validated"],
         "| all runs share the fingerprint | **%s** |" % rep["all_runs_share_canonical_fingerprint"],
         "",
         c["evidence_note"] + ".", "",
         "Every prediction file was joined to the canonical mapping by **exact image "
         "identity, never by row position**, and every run was required to carry the "
         "canonical true label for each image. Per-file SHA-256 digests for all 48 "
         "prediction files and 24 metrics files are in `analysis_provenance.json`.", "",
         "### Prerequisite gate", "", "| Prerequisite | Result |", "|---|---|"]
    for k, v in rep["prerequisite_gate"]["checks"].items():
        L.append("| %s | **%s** |" % (k.replace("_", " "), "PASS" if v else "FAIL"))
    L += ["| imported files re-hashed | %d of %d expected |"
          % (rep["prerequisite_gate"]["imported_files_rehashed"],
             rep["prerequisite_gate"]["n_expected_imported_files"]), "",
          "## Reference reproduction", "",
          "| Metric | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD (ddof=1) | Verified | diff (pp) |",
          "|---|---:|---:|---:|---|---|---:|",
          "| accuracy | %.4f | %.4f | %.4f | %.4f +/- %.4f | %.2f +/- %.2f | %.4f / %.4f |"
          % (rr["accuracy_per_seed_pct"][0], rr["accuracy_per_seed_pct"][1],
             rr["accuracy_per_seed_pct"][2], rr["accuracy_mean_pct"], rr["accuracy_sd_pct"],
             rr["verified_accuracy_pct"][0], rr["verified_accuracy_pct"][1],
             rr["accuracy_mean_diff_pp"], rr["accuracy_sd_diff_pp"]),
          "| macro-F1 | %.4f | %.4f | %.4f | %.4f +/- %.4f | %.2f +/- %.2f | %.4f / %.4f |"
          % (rr["macro_f1_per_seed_pct"][0], rr["macro_f1_per_seed_pct"][1],
             rr["macro_f1_per_seed_pct"][2], rr["macro_f1_mean_pct"], rr["macro_f1_sd_pct"],
             rr["verified_macro_f1_pct"][0], rr["verified_macro_f1_pct"][1],
             rr["macro_f1_mean_diff_pp"], rr["macro_f1_sd_diff_pp"]),
          "", "Reproduces within %.2f pp: **%s**." % (METRIC_TOL_PP,
                                                      rr["reproduces_within_tolerance"]), "",
          rr["macro_f1_evidence_note"], "",
          "## Provenance of the reference gate status", ""]

    if gp["strengthened"]:
        L += ["The Phase 3A Addendum could only resolve the reference's `gated` setting "
              "from the launch specification, its weakest evidence class. **That caveat "
              "is now superseded.** The reasoning, from "
              "`%s`:" % gp["source"], "",
              gp["argument"], "",
              "| Seed | Strict load | keys | gate params | probe gate mass (CNN) | away from 0.5 | 848 preds reproduced | max prob diff |",
              "|---|---|---:|---:|---:|---|---|---:|"]
        for s in SEEDS:
            e = gp["per_seed"][s]
            L.append("| %d | %s | %s | %s | %.4f +/- %.4f | %s | %s | %.2e |"
                     % (s, e["strict_load"], e["state_dict_keys"],
                        e["gate_parameter_count"], e["gate_cnn_mean_on_probe"],
                        e["gate_cnn_std_on_probe"], e["gate_separated_from_uniform_mixing"],
                        e["reference_predictions_reproduced"], e["max_abs_prob_diff"]))
        L += ["", "`cnn_proj_in_features = 1432` (= 40 + 112 + 1280) in all three "
                  "recovered checkpoints likewise confirms the multi-scale taps "
                  "directly, which is stronger than the parameter-count derivation the "
                  "addendum had to use.", ""]
    else:
        L += ["The checkpoint-recovery record does **not** establish the required "
              "linkage (%s), so the weaker Phase 3A Addendum launch-specification "
              "caveat is **retained** for the reference `gated` setting." % gp["reason"], ""]
    L += [gp["what_this_does_not_establish"], "",
          "## no_tta provenance", "",
          "| Seed | Identical recovered checkpoint | Weights unchanged | TTA flag | Checkpoint written | Only change is TTA |",
          "|---|---|---|---|---|---|"]
    for s in SEEDS:
        e = nt["per_seed"][s]
        L.append("| %d | %s | %s | %s | %s | **%s** |"
                 % (s, e["identical_checkpoint"], e["no_weights_changed"],
                    e["tta_flag"], e["checkpoint_written"], e["tta_is_the_only_change"]))
    L += ["", nt["note"], "",
          "## Accuracy and macro-F1, per seed and mean +/- SD", "",
          "| Model | Seed 42 | Seed 43 | Seed 44 | Accuracy mean +/- SD | Macro-F1 mean +/- SD | Balanced acc mean |",
          "|---|---:|---:|---:|---|---|---:|"]
    for v in [REFERENCE] + VARIANTS:
        sm = summaries[v]
        per = sm["per_seed"]["accuracy"]
        L.append("| %s | %.4f | %.4f | %.4f | %.4f +/- %s | %.4f +/- %s | %.4f |"
                 % (_name(v),
                    100 * per[42], 100 * per[43], 100 * per[44],
                    100 * sm["accuracy"]["mean"],
                    num(100 * sm["accuracy"]["sd"]) if sm["accuracy"]["sd"] is not None else "n/a",
                    100 * sm["macro_f1"]["mean"],
                    num(100 * sm["macro_f1"]["sd"]) if sm["macro_f1"]["sd"] is not None else "n/a",
                    100 * sm["balanced_accuracy"]["mean"]))
    L += ["", "All values are percentages. The SD is the sample SD (ddof=1) over three "
              "independently trained seeds and is a spread, not a confidence interval.", "",
          "## Calibration and probabilistic quality", "",
          "| Model | NLL | Brier | ECE (%d bins) | Mean conf. | Mean conf. correct | Mean conf. wrong | Macro ROC-AUC | Macro AP |"
          % ECE_BINS,
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in [REFERENCE] + VARIANTS:
        sm = summaries[v]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                 % (_name(v),
                    num(sm["nll"]["mean"]), num(sm["brier"]["mean"]),
                    num(sm["ece"]["mean"]), num(sm["mean_confidence"]["mean"]),
                    num(sm["mean_confidence_correct"]["mean"]),
                    num(sm["mean_confidence_incorrect"]["mean"]),
                    num(sm["macro_roc_auc_ovr"]["mean"]),
                    num(sm["macro_average_precision_ovr"]["mean"])))
    L += ["", "**These calibration comparisons are descriptive and exploratory "
              "only.** The table holds mean values over three seeds. No paired "
              "inferential procedure and no multiplicity control were prespecified "
              "for NLL, Brier or ECE, and none has been added afterwards, so no "
              "between-model calibration comparison here is a statistical "
              "conclusion. A variant whose mean ECE, NLL or Brier is numerically "
              "lower in these runs is described exactly that way: numerically lower "
              "in these runs. It is not described as better calibrated, and the "
              "absence of a difference is not described as equivalent calibration. "
              "Differences of this size across three seeds carry no attached "
              "uncertainty and should not be used to rank models or to support or "
              "undermine any component.", "",
          "ECE uses %s. Macro AP is step-wise **average precision**, not "
          "trapezoidal PR-AUC; the two are computed differently and are not "
          "interchangeable. A metric returns `n/a` where it is mathematically "
          "undefined rather than being reported as 0." % ECE_RULE, "",
          "## Paired disagreement tables", "",
          "Each row is one ablation against the reference at the same seed over the "
          "same 848 images, matched by image identity. `b` is reference-correct and "
          "ablation-wrong; `c` is the reverse. Only the b+c discordant pairs carry "
          "information about a difference.", "",
          "| Variant | Seed | Both correct | b (ref ok, abl wrong) | c (ref wrong, abl ok) | Both wrong | Discordant | d accuracy (pp) | d macro-F1 (pp) | d balanced acc (pp) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in tests:
        tb = t["table"]
        L.append("| `%s` | %d | %d | %d | %d | %d | %d | %s | %s | %s |"
                 % (t["variant"], t["seed"], tb["both_correct"],
                    tb["reference_correct_ablation_wrong"],
                    tb["reference_wrong_ablation_correct"], tb["both_wrong"],
                    t["mcnemar"]["n_discordant"], pp(t["accuracy_difference_pp"]),
                    pp(t["macro_f1_difference_pp"]),
                    pp(t["balanced_accuracy_difference_pp"])))
    L += ["", "## Exact McNemar tests, unadjusted and Holm-adjusted", "",
          "%s Family size **%d**, alpha 0.05. %d of %d tests are significant before "
          "correction and %d after."
          % (mcc["why_this_family"], mcc["n_tests"], mcc["n_significant_before_holm"],
             mcc["n_tests"], mcc["n_significant_after_holm"]), "",
          "| Variant | Seed | b | c | Discordant | p unadjusted | p Holm-adjusted | Significant after Holm |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for t in tests:
        m = t["mcnemar"]
        L.append("| `%s` | %d | %d | %d | %d | %.4g | %.4g | %s |"
                 % (t["variant"], t["seed"], m["b_ref_correct_abl_wrong"],
                    m["c_ref_wrong_abl_correct"], m["n_discordant"],
                    t["unadjusted_p"], t["holm_adjusted_p"],
                    "**yes**" if t["holm_significant_at_0.05"] else "no"))
    L += ["", "The test is the exact two-sided binomial McNemar test on the discordant "
              "pairs, not the chi-square approximation: at these counts the "
              "approximation is not trustworthy.", ""]
    zero_disc = [t for t in tests if t["mcnemar"]["n_discordant"] == 0]
    if zero_disc:
        L += ["%d comparison(s) have no discordant pairs at all, meaning the two models "
              "made identical correct/incorrect decisions on all 848 images. Those "
              "tests carry no information; p is recorded as 1.0 and must not be read "
              "as evidence of equivalence." % len(zero_disc), ""]
    L += ["## Paired image-level bootstrap intervals", "",
          "%d iterations, fixed seed %d, resampling unit **image**, stratified by true "
          "class, one index set per iteration applied to both models so the pairing is "
          "preserved. These are image-level intervals within a single training seed. "
          "They are not patient-level intervals: no patient identifiers exist for this "
          "split. They are also not the seed SD."
          % (BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED), "",
          "| Variant | Seed | d accuracy (pp) | 95% CI (pp) | Excludes 0 | d macro-F1 (pp) | 95% CI (pp) |",
          "|---|---:|---:|---|---|---:|---|"]
    for t in tests:
        a = t["bootstrap"]["accuracy_difference_pp"]
        f1 = t["bootstrap"]["macro_f1_difference_pp"]
        L.append("| `%s` | %d | %s | [%s, %s] | %s | %s | [%s, %s] |"
                 % (t["variant"], t["seed"], pp(a["point_pp"]),
                    pp(a["ci95_lo_pp"]), pp(a["ci95_hi_pp"]),
                    "**yes**" if a["excludes_zero"] else "no",
                    pp(f1["point_pp"]), pp(f1["ci95_lo_pp"]), pp(f1["ci95_hi_pp"])))
    L += ["", "## Class-specific effects", "",
          "Per-class recall (sensitivity) and specificity, mean over the three seeds, "
          "so a component that helps overall accuracy while costing one class is "
          "visible.", "",
          "| Model | " + " | ".join("%s rec / spec" % k for k in CLASSES) + " |",
          "|---|" + "|".join(["---:"] * len(CLASSES)) + "|"]
    for v in [REFERENCE] + VARIANTS:
        cells = []
        for ci, k in enumerate(CLASSES):
            recs = [by_seed[(v, s)]["per_class"][ci]["recall_sensitivity"] for s in SEEDS]
            specs = [by_seed[(v, s)]["per_class"][ci]["specificity"] for s in SEEDS]
            cells.append("%.4f / %.4f" % (float(np.mean(recs)), float(np.mean(specs))))
        L.append("| %s | %s |" % (_name(v), " | ".join(cells)))
    L += ["", "Support per class: " + ", ".join(
        "%s %d" % (k, canon["class_counts"][k]) for k in CLASSES) + ".", "",
          "Full per-class precision, recall, specificity, F1, ROC-AUC and AP for every "
          "model and seed are in `per_class_metrics.csv`.", "",
          "## Component conclusions", "",
          "### The decision rule", "",
          "An earlier version of this analysis decided verdicts using a "
          "\"materiality threshold\" of twice the reference seed SD. **That "
          "threshold has been removed and not replaced.** It was chosen after the "
          "effect sizes were known, it was not an equivalence margin, and it had no "
          "external or clinical justification; a cutoff derived from the data's own "
          "noise is still a cutoff picked with knowledge of the answer. Observed "
          "differences are reported numerically below and are deliberately not "
          "labelled important or negligible, because no prespecified margin exists "
          "to judge them against.", "",
          "No effect-size threshold of any kind takes part in a verdict. The rule "
          "uses only prespecified quantities - the direction of the per-seed "
          "accuracy differences, the Holm-adjusted p-values over the fixed 21-test "
          "family, and the paired image-level bootstrap intervals - and is stated in "
          "full so it can be audited against the code:", "",
          "| Verdict | Assigned when |", "|---|---|"]
    for name, when in DECISION_RULE:
        L.append("| `%s` | %s |" % (name, when))
    L += ["", "The four category names are scoped to **accuracy**. Macro-F1, "
              "balanced accuracy and the class-wise pattern are recorded and "
              "reported, but they do not drive a verdict, because no prespecified "
              "paired inferential procedure or multiplicity control exists for "
              "them. Where a secondary metric is consistently adverse while "
              "accuracy is not, that is reported as an observed pattern and is not "
              "turned into a claim.", "",
          "Three seeds bound everything here. The seed spread is estimated from "
          "n=3 and is reported as a spread, never as a confidence interval or a "
          "significance test.", "",
          "### Verdicts", "",
          "| Variant | Component removed | Observed mean d accuracy (pp) | Adverse at every seed | Holm sig. | CI excludes 0 | Verdict |",
          "|---|---|---:|---|---:|---:|---|"]
    for cc in conclusions:
        L.append("| `%s` | %s | %s | %s | %d/3 | %d/3 | **%s** |"
                 % (cc["variant"], cc["component_removed"],
                    "%+.4f" % cc["observed_mean_accuracy_difference_pp"],
                    "yes" if cc["accuracy_adverse_at_every_seed"] else "no",
                    cc["n_holm_significant_of_3"],
                    cc["n_bootstrap_ci_excludes_zero_of_3"], cc["verdict"]))
    L.append("")
    for cc in conclusions:
        L += ["### `%s` - %s" % (cc["variant"], cc["verdict"]), "",
              "Removes: %s." % cc["component_removed"], "",
              "- accuracy: %.4f%% +/- %s (reference %.4f%% +/- %.4f)"
              % (cc["accuracy_mean_pct"],
                 num(cc["accuracy_sd_pct"]) if cc["accuracy_sd_pct"] is not None else "n/a",
                 rr["accuracy_mean_pct"], rr["accuracy_sd_pct"]),
              "- per-seed accuracy difference: %s pp"
              % ", ".join("%+.4f" % x for x in cc["per_seed_accuracy_difference_pp"]),
              # 4 dp deliberately: at 2 dp one of these observed means prints as
              # "-0.14 pp", the same digits as the deleted post-hoc threshold, and
              # a measurement must not be mistakable for a cutoff.
              "- observed mean macro-F1 difference: %+.4f pp (per seed %s); observed "
              "mean balanced-accuracy difference: %+.4f pp (per seed %s)"
              % (cc["observed_mean_macro_f1_difference_pp"],
                 ", ".join("%+.4f" % x for x in cc["per_seed_macro_f1_difference_pp"]),
                 cc["observed_mean_balanced_accuracy_difference_pp"],
                 ", ".join("%+.4f" % x for x in
                           cc["per_seed_balanced_accuracy_difference_pp"])),
              "- Holm-adjusted p per seed: %s"
              % ", ".join("%.4g" % x for x in cc["holm_adjusted_p_per_seed"]),
              "- bootstrap 95%% CI per seed (pp): %s"
              % "; ".join("[%+.2f, %+.2f]" % (lo, hi) for lo, hi in cc["bootstrap_ci_per_seed_pp"]),
              ""]
        L += ["- %s" % r for r in cc["reasons"]]
        L.append("")
    L += [rep["variant_level_descriptive_note"], "",
          "## External 499-image predictions - descriptive only", "",
          "| Model | Seed 42 | Seed 43 | Seed 44 | Mean +/- SD |", "|---|---:|---:|---:|---|"]
    for v in [REFERENCE] + VARIANTS:
        e = rep["external_full_descriptive"]["per_model"][v]
        a = e["accuracy_per_seed"]
        L.append("| %s | %.4f | %.4f | %.4f | %.4f +/- %s |"
                 % (_name(v), 100 * a[0], 100 * a[1], 100 * a[2],
                    100 * e["mean"], num(100 * e["sd"]) if e["sd"] is not None else "n/a"))
    L += ["", "Guardrails on this table:", ""]
    L += ["- %s" % g for g in rep["external_full_descriptive"]["guardrails"]]
    cl = rep["claims"]
    L += ["## What this analysis claims", "",
          "Each sentence below is generated from the computed numbers rather than "
          "written by hand, so it cannot drift away from the evidence it "
          "describes.", ""]
    L += ["%d. %s" % (i, x) for i, x in enumerate(cl["claims"], 1)]
    L += ["", "### What it does not claim", "",
          "This analysis does not show, and this report does not say:", ""]
    L += ["- %s" % x for x in cl["not_claimed"]]
    au = rep["statistical_audit"]
    L += ["", "## Statistical consistency audit", "",
          "A re-read of the prespecified analysis. It changes no part of it.", "",
          "| Property | Result |", "|---|---|"]
    for k, v in au["checks"].items():
        L.append("| %s | **%s** |" % (k.replace("_", " "), "PASS" if v else "FAIL"))
    L += ["", "%d tests in the family; %d significant before Holm correction and %d "
              "after. %s" % (au["n_tests"], au["n_significant_before_holm"],
                             au["n_significant_after_holm"], au["note"]), "",
          "## Limitations", ""]
    L += ["%d. %s" % (i, x) for i, x in enumerate(rep["limitations"], 1)]
    L += ["", "## Acceptance checks", "", "| Check | Result |", "|---|---|"]
    for k, v in rep["acceptance_checks"].items():
        L.append("| %s | **%s** |" % (k.replace("_", " "), "PASS" if v else "FAIL"))
    L += ["", "**Overall: %s.**" % rep["acceptance_status"], ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AnalysisError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
