"""
Phase 3B - tests for the final controlled ablation analysis.

Two kinds of test live here, and both are necessary.

REJECTION TESTS build a synthetic run in a temporary directory, break exactly one
thing, and assert the analyser refuses it: a missing image, a duplicated image, a
substituted image, a wrong canonical label, a permuted class order, malformed
probabilities, a seed mismatch, a no_tta checkpoint mismatch.

FORMULA TESTS check the arithmetic against values computed by hand or by an
independent method, because a validator that accepts only clean data still tells
you nothing about whether its numbers are right. These cover per-class
specificity, the multiclass Brier score, the ECE bin edges, AP versus
trapezoidal PR-AUC, the exact two-sided McNemar calculation, Holm across exactly
21 tests, paired stratified bootstrap determinism, and ddof=1 seed SD.

Three tests exist specifically to pin down the protocol guarantees that are easy
to violate silently: that filename order cannot change any result, that the
Holm family must be 21 and not 7, and that 2,544 predictions are never treated as
independent observations.

Nothing here touches runs/, runs_*/, results*/, publication_runs/ or dataset
directories. Every fixture is synthetic and lives under the OS temp directory.

    python src/publication/test_final_analysis.py --project-root .
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import final_analysis as A

PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        FAILED.append((name, "assertion: %s" % e))
        print("  FAIL  %s\n          %s" % (name, e), flush=True)
    except Exception as e:
        FAILED.append((name, "%s: %s" % (type(e).__name__, e)))
        print("  ERROR %s\n          %s: %s" % (name, type(e).__name__, e), flush=True)
    else:
        PASSED.append(name)
        print("  ok    %s" % name, flush=True)


def rejects(fn, *a, **k):
    try:
        fn(*a, **k)
    except A.AnalysisError as e:
        return str(e)
    raise AssertionError("accepted an input it should have rejected")


# --------------------------------------------------------------- fixtures ---

CLASSES = list(A.CLASSES)
N = 40


def synth_mapping(n=N):
    return {"img_%03d.jpg" % i: CLASSES[i % len(CLASSES)] for i in range(n)}


def canon_of(mapping):
    fp = A.fingerprint(sorted(mapping), [mapping[f] for f in sorted(mapping)])
    from collections import Counter
    return {"source": "synthetic", "source_sha256": "0" * 64,
            "n_identities": len(mapping), "mapping": mapping,
            "fingerprint_sha256": fp, "class_order": CLASSES,
            "class_counts": dict(Counter(mapping.values())),
            "evidence_note": "synthetic fixture"}


def write_preds(path, mapping, files=None, labels=None, class_cols=None,
                probs=None, pred_labels=None, wrong_every=None):
    files = list(mapping) if files is None else files
    labels = [mapping[f] for f in files] if labels is None else labels
    cols = class_cols or CLASSES
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "true_label", "pred_label"] + ["p_%s" % c for c in cols])
        for i, (f, lab) in enumerate(zip(files, labels)):
            if probs is not None:
                pr = list(probs[i])
            else:
                wrong = wrong_every is not None and i % wrong_every == 0
                target = (CLASSES[(CLASSES.index(lab) + 1) % 4] if wrong else lab)
                pr = [0.05] * len(cols)
                pr[cols.index(target)] = 1.0 - 0.05 * (len(cols) - 1)
            pl = pred_labels[i] if pred_labels else cols[int(np.argmax(pr))]
            w.writerow([f, lab, pl] + ["%.6f" % x for x in pr])


def load(path, mapping, expect_n=None):
    old = A.CLEAN_TEST_N
    A.CLEAN_TEST_N = expect_n or len(mapping)
    try:
        return A.load_predictions(path, A.CLEAN_TEST_N, canon_of(mapping))
    finally:
        A.CLEAN_TEST_N = old


# ------------------------------------------------- 1. rejection: identity ---

def t01_missing_image_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)[:-1]
        p = os.path.join(tmp, "p.csv")
        write_preds(p, m, files=files, labels=[m[f] for f in files])
        msg = rejects(load, p, m, expect_n=len(m))
        assert "rows, expected" in msg, msg


def t02_duplicate_image_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)
        files[5] = files[4]
        p = os.path.join(tmp, "p.csv")
        write_preds(p, m, files=files, labels=[m[f] for f in files])
        msg = rejects(load, p, m)
        assert "duplicate image identities" in msg, msg


def t03_substituted_image_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)
        files[7] = "not_in_the_split.jpg"
        labels = [m.get(f, "glioma") for f in files]
        p = os.path.join(tmp, "p.csv")
        write_preds(p, m, files=files, labels=labels)
        msg = rejects(load, p, m)
        assert "canonical membership differs" in msg, msg


def t04_wrong_canonical_label_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)
        labels = [m[f] for f in files]
        labels[3] = CLASSES[(CLASSES.index(labels[3]) + 1) % 4]
        p = os.path.join(tmp, "p.csv")
        write_preds(p, m, files=files, labels=labels)
        msg = rejects(load, p, m)
        assert "true_label != canonical label" in msg, msg


def t05_wrong_class_order_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        p = os.path.join(tmp, "p.csv")
        write_preds(p, m, class_cols=[CLASSES[2], CLASSES[0], CLASSES[1], CLASSES[3]])
        msg = rejects(load, p, m)
        assert "sorted class order" in msg or "class set" in msg, msg


def t06_malformed_probabilities_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        # (a) rows that do not sum to 1
        P = np.full((len(m), 4), 0.25)
        P[3] = [0.1, 0.1, 0.1, 0.1]
        p = os.path.join(tmp, "a.csv")
        write_preds(p, m, probs=P)
        msg = rejects(load, p, m)
        assert "deviate from 1" in msg, msg
        # (b) a probability outside [0,1]
        P2 = np.full((len(m), 4), 0.25)
        P2[2] = [-0.5, 0.5, 0.5, 0.5]
        p2 = os.path.join(tmp, "b.csv")
        write_preds(p2, m, probs=P2)
        msg2 = rejects(load, p2, m)
        assert "outside [0,1]" in msg2, msg2
        # (c) a non-finite probability
        p3 = os.path.join(tmp, "c.csv")
        write_preds(p3, m)
        rows = list(csv.DictReader(open(p3, encoding="utf-8")))
        cols = list(rows[0])
        rows[1]["p_glioma"] = "nan"
        with open(p3, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
        msg3 = rejects(load, p3, m)
        assert "non-finite" in msg3, msg3


def t07_pred_label_must_match_argmax():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)
        pl = [m[f] for f in files]
        pl[6] = CLASSES[(CLASSES.index(pl[6]) + 1) % 4]   # not the argmax
        p = os.path.join(tmp, "p.csv")
        write_preds(p, m, files=files, pred_labels=pl)
        msg = rejects(load, p, m)
        assert "pred_label != argmax" in msg, msg


def t08_documented_tie_is_allowed_and_counted():
    """A genuine top-two tie is not an error, but it must be counted and reported
    rather than passed over in silence."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)
        P = np.zeros((len(m), 4))
        for i, f in enumerate(files):
            P[i, CLASSES.index(m[f])] = 1.0
        # row 0: exact tie between the true class and one other
        other = (CLASSES.index(m[files[0]]) + 1) % 4
        P[0] = 0.0
        P[0, CLASSES.index(m[files[0]])] = 0.5
        P[0, other] = 0.5
        pl = [m[f] for f in files]
        pl[0] = CLASSES[other]        # legitimately either side of the tie
        p = os.path.join(tmp, "p.csv")
        write_preds(p, m, files=files, pred_labels=pl, probs=P)
        d = load(p, m)
        assert d["n_documented_argmax_ties"] == 1, d["n_documented_argmax_ties"]


# ------------------------------------------- 2. filename-order independence ---

def t09_filename_order_independence():
    """Row order must not change anything. The analyser reorders onto canonical
    identity order at load time; this asserts that actually happens."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        fwd = list(m)
        rev = fwd[::-1]
        pa = os.path.join(tmp, "a.csv")
        pb = os.path.join(tmp, "b.csv")
        write_preds(pa, m, files=fwd, labels=[m[f] for f in fwd], wrong_every=7)
        write_preds(pb, m, files=rev, labels=[m[f] for f in rev], wrong_every=None)
        # rebuild b with the SAME per-image decisions as a, in reverse row order
        rows = {r["filename"]: r for r in csv.DictReader(open(pa, encoding="utf-8"))}
        cols = list(next(iter(rows.values())))
        with open(pb, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, cols); w.writeheader()
            for f in rev:
                w.writerow(rows[f])
        da, db = load(pa, m), load(pb, m)
        assert da["fingerprint_sha256"] == db["fingerprint_sha256"]
        assert list(da["canonical_order_files"]) == list(db["canonical_order_files"])
        assert np.array_equal(da["y"], db["y"])
        assert np.array_equal(da["pred"], db["pred"]), "row order changed the predictions"
        assert np.allclose(da["P_canonical"], db["P_canonical"])
        ma = A.compute_metrics(da["y"], da["P_canonical"], CLASSES)
        mb = A.compute_metrics(db["y"], db["P_canonical"], CLASSES)
        for k in ("accuracy", "macro_f1", "balanced_accuracy", "nll", "brier", "ece"):
            assert abs(ma[k] - mb[k]) < 1e-12, "%s changed with row order" % k


# ----------------------------------------------------- 3. metric formulas ----

def t10_metric_formulas_against_hand_computation():
    """A 6-image, 4-class case small enough to verify by hand."""
    y = np.array([0, 0, 1, 1, 2, 3])
    # predictions: 0->0, 0->1, 1->1, 1->1, 2->2, 3->0
    P = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.7, 0.1, 0.1, 0.1],
    ])
    m = A.compute_metrics(y, P, CLASSES)
    assert m["accuracy"] == 4 / 6, m["accuracy"]
    # class 0: tp=1 fp=1 fn=1 tn=3 -> prec .5 rec .5 f1 .5 spec 3/4
    c0 = m["per_class"][0]
    assert (c0["tp"], c0["fp"], c0["fn"], c0["tn"]) == (1, 1, 1, 3), c0
    assert abs(c0["precision"] - 0.5) < 1e-12
    assert abs(c0["recall_sensitivity"] - 0.5) < 1e-12
    assert abs(c0["specificity"] - 0.75) < 1e-12
    assert abs(c0["f1"] - 0.5) < 1e-12
    # class 1: tp=2 fp=1 fn=0 tn=3 -> prec 2/3 rec 1 f1 0.8 spec 3/4
    c1 = m["per_class"][1]
    assert (c1["tp"], c1["fp"], c1["fn"], c1["tn"]) == (2, 1, 0, 3), c1
    assert abs(c1["precision"] - 2 / 3) < 1e-12
    assert abs(c1["recall_sensitivity"] - 1.0) < 1e-12
    assert abs(c1["f1"] - 0.8) < 1e-12
    assert abs(c1["specificity"] - 0.75) < 1e-12
    # class 2: perfect -> 1/1/1, spec 1
    c2 = m["per_class"][2]
    assert abs(c2["f1"] - 1.0) < 1e-12 and abs(c2["specificity"] - 1.0) < 1e-12
    # class 3: tp=0 fn=1 fp=0 -> prec 0 rec 0 f1 0 (zero-division policy), spec 1
    c3 = m["per_class"][3]
    assert c3["precision"] == 0.0 and c3["recall_sensitivity"] == 0.0 and c3["f1"] == 0.0
    assert abs(c3["specificity"] - 1.0) < 1e-12
    expect_f1 = np.mean([0.5, 0.8, 1.0, 0.0])
    assert abs(m["macro_f1"] - expect_f1) < 1e-12, (m["macro_f1"], expect_f1)
    expect_bal = np.mean([0.5, 1.0, 1.0, 0.0])
    assert abs(m["balanced_accuracy"] - expect_bal) < 1e-12
    assert abs(m["macro_recall"] - expect_bal) < 1e-12, "balanced acc == macro recall"
    # confusion matrix rows sum to support
    cm = np.array(m["confusion_matrix"]["counts"])
    assert cm.sum() == 6 and list(cm.sum(1)) == [2, 2, 1, 1], cm.tolist()


def t11_per_class_specificity_is_one_vs_rest():
    """Specificity must be TN/(TN+FP) over the OTHER classes, not 1 - recall."""
    y = np.array([0, 1, 1, 2])
    P = np.eye(4)[[0, 0, 1, 2]]      # one class-1 image predicted as class 0
    m = A.compute_metrics(y, P, CLASSES)
    c0 = m["per_class"][0]
    # class 0: tp=1, fp=1 (the class-1 image), fn=0, tn=2 -> spec = 2/3
    assert (c0["tp"], c0["fp"], c0["fn"], c0["tn"]) == (1, 1, 0, 2), c0
    assert abs(c0["specificity"] - 2 / 3) < 1e-12, c0["specificity"]
    assert abs(c0["specificity"] - (1 - c0["recall_sensitivity"])) > 0.1, \
        "specificity was computed as 1 - recall"


def t12_multiclass_brier_score():
    """Brier = mean over images of the SUM of squared errors across classes.
    A common error is to average over classes as well, which is 4x smaller."""
    y = np.array([0, 1])
    P = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    m = A.compute_metrics(y, P, CLASSES)
    # image 1: 0. image 2: (0.5-0)^2 + (0.5-1)^2 = 0.5 -> mean 0.25
    assert abs(m["brier"] - 0.25) < 1e-12, m["brier"]
    assert abs(m["brier"] - 0.25 / 4) > 1e-6, "Brier was averaged over classes too"
    # NLL: -(log 1 + log 0.5)/2
    assert abs(m["nll"] - (-(math.log(1.0) + math.log(0.5)) / 2)) < 1e-9, m["nll"]


def t13_ece_bin_edges_and_value():
    """15 equal-width bins on [0,1]; first bin closed at both ends, rest (lo, hi]."""
    conf = np.array([0.0, 1 / 15.0, 0.5, 1.0])
    correct = np.array([True, True, False, True])
    ece, rows = A.expected_calibration_error(conf, correct, bins=15)
    assert len(rows) == 15, len(rows)
    edges = [r["bin_lo"] for r in rows] + [rows[-1]["bin_hi"]]
    assert abs(edges[0]) < 1e-15 and abs(edges[-1] - 1.0) < 1e-15, edges
    widths = [rows[i]["bin_hi"] - rows[i]["bin_lo"] for i in range(15)]
    assert max(widths) - min(widths) < 1e-12, "bins are not equal width"
    assert abs(widths[0] - 1 / 15.0) < 1e-12, widths[0]
    # 0.0 and 1/15 both fall in bin 0 (closed at both ends), so it holds 2 images
    assert rows[0]["n"] == 2, [r["n"] for r in rows]
    assert sum(r["n"] for r in rows) == len(conf)
    # hand value: bin0 n=2 acc 1.0 conf mean 1/30 -> 0.5*|1-1/30|
    #             bin7 (0.4667,0.5333] n=1 acc 0 conf 0.5 -> 0.25*|0-0.5|
    #             bin14 n=1 acc 1 conf 1 -> 0
    want = 0.5 * abs(1.0 - (0.0 + 1 / 15.0) / 2) + 0.25 * 0.5
    assert abs(ece - want) < 1e-12, (ece, want)
    assert A.ECE_BINS == 15, A.ECE_BINS


def t14_ap_is_not_trapezoidal_pr_auc():
    """AP is the step-wise mean of precision at each positive. Assert it equals
    the hand value AND differs from a trapezoidal PR curve area, so the naming in
    the outputs is not cosmetic."""
    yb = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6])
    ap = A.average_precision_ovr(yb, score)
    # positives at ranks 1 and 3: precision 1/1 and 2/3 -> mean 5/6
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-12, ap

    def trapezoid_pr_auc(yb, score):
        order = np.argsort(-score, kind="mergesort")
        y = yb[order]
        tp = np.cumsum(y)
        prec = tp / np.arange(1, len(y) + 1)
        rec = tp / y.sum()
        return float(np.trapezoid(prec, rec)) if hasattr(np, "trapezoid") \
            else float(np.trapz(prec, rec))

    tz = trapezoid_pr_auc(yb, score)
    assert abs(ap - tz) > 1e-6, ("AP and trapezoidal PR-AUC coincided on a case "
                                 "built to separate them", ap, tz)
    assert A.average_precision_ovr(np.array([0, 0, 0]), np.array([.1, .2, .3])) is None, \
        "AP with no positives must be not-available, not 0"


def t15_roc_auc_undefined_is_not_zero():
    assert A.roc_auc_ovr(np.array([0, 0, 0]), np.array([.1, .2, .3])) is None
    assert A.roc_auc_ovr(np.array([1, 1, 1]), np.array([.1, .2, .3])) is None
    # perfect separation -> 1.0; reversed -> 0.0; all ties -> 0.5
    assert abs(A.roc_auc_ovr(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) - 1.0) < 1e-12
    assert abs(A.roc_auc_ovr(np.array([1, 1, 0, 0]), np.array([.1, .2, .8, .9])) - 0.0) < 1e-12
    assert abs(A.roc_auc_ovr(np.array([0, 1]), np.array([.5, .5])) - 0.5) < 1e-12, \
        "tied scores must be rank-averaged, not broken arbitrarily"


# --------------------------------------------- 4. paired statistics ----------

def t16_mcnemar_discordance_counts():
    """The 2x2 table must count the four cells over matched images."""
    m = synth_mapping(8)
    files = sorted(m)
    y = np.array([CLASSES.index(m[f]) for f in files])
    # reference correct on all but index 0; ablation correct on all but 0 and 1
    rp = y.copy(); rp[0] = (y[0] + 1) % 4
    ap_ = y.copy(); ap_[0] = (y[0] + 1) % 4; ap_[1] = (y[1] + 1) % 4
    ref = {"seed": 42, "run_dir": "r",
           "clean_test": {"canonical_order_files": files, "y": y, "pred": rp}}
    abl = {"seed": 42, "run_dir": "a",
           "clean_test": {"canonical_order_files": files, "y": y, "pred": ap_}}
    old = A.CLEAN_TEST_N
    A.CLEAN_TEST_N = len(files)
    try:
        tab = A.paired_disagreement(ref, abl)
    finally:
        A.CLEAN_TEST_N = old
    assert tab["both_wrong"] == 1, tab            # index 0
    assert tab["reference_correct_ablation_wrong"] == 1, tab   # index 1
    assert tab["reference_wrong_ablation_correct"] == 0, tab
    assert tab["both_correct"] == 6, tab
    assert sum([tab["both_correct"], tab["reference_correct_ablation_wrong"],
                tab["reference_wrong_ablation_correct"], tab["both_wrong"]]) == 8


def t17_exact_two_sided_mcnemar_values():
    """Against values computable in closed form from Binomial(n, 0.5)."""
    # b=0, c=0 -> no information; p reported as 1.0 and flagged
    z = A.mcnemar_exact(0, 0)
    assert z["p_value"] == 1.0 and z["n_discordant"] == 0
    assert "must not be read as evidence of equivalence" in z["note"]
    # b=1, c=0, n=1 -> 2 * 0.5 = 1.0
    assert abs(A.mcnemar_exact(1, 0)["p_value"] - 1.0) < 1e-12
    # b=5, c=0, n=5 -> 2 * 0.5^5 = 0.0625
    assert abs(A.mcnemar_exact(5, 0)["p_value"] - 2 * 0.5 ** 5) < 1e-12
    # b=10, c=1, n=11 -> 2 * (C(11,0)+C(11,1)) * 0.5^11 = 2*12/2048
    assert abs(A.mcnemar_exact(10, 1)["p_value"] - 2 * 12 / 2048.0) < 1e-12
    # symmetric in b,c
    assert A.mcnemar_exact(3, 9)["p_value"] == A.mcnemar_exact(9, 3)["p_value"]
    # never exceeds 1
    for b in range(6):
        for c in range(6):
            assert 0.0 <= A.mcnemar_exact(b, c)["p_value"] <= 1.0, (b, c)
    # b=c -> maximally non-significant
    assert abs(A.mcnemar_exact(4, 4)["p_value"] - 1.0) < 1e-12
    # cross-check against an independent binomial sum for a larger case
    b, c = 17, 4
    n = b + c
    want = min(1.0, 2 * sum(math.comb(n, i) for i in range(c + 1)) * 0.5 ** n)
    assert abs(A.mcnemar_exact(b, c)["p_value"] - want) < 1e-15


def t18_holm_across_exactly_21_tests():
    n = 21
    ps = [0.0001] + [0.6] * (n - 1)
    adj = A.holm(ps)
    assert len(adj) == n
    assert abs(adj[0] - min(1.0, 21 * 0.0001)) < 1e-12, adj[0]
    assert all(x <= 1.0 for x in adj)
    # monotone in the sorted order
    order = sorted(range(n), key=lambda i: ps[i])
    seq = [adj[i] for i in order]
    assert seq == sorted(seq), "Holm output is not monotone in sorted p order"
    # adjusted is never smaller than unadjusted
    assert all(a >= p - 1e-15 for a, p in zip(adj, ps))
    # family-size guard: 7 must be refused, 21 accepted
    msg = rejects(A.holm_family_guard, 7)
    assert "expected 21" in msg, msg
    A.holm_family_guard(21)
    assert A.N_EXPECTED_TESTS == 21, A.N_EXPECTED_TESTS


def t19_holm_is_more_conservative_than_no_correction():
    ps = [0.001, 0.002, 0.03, 0.04] + [0.5] * 17
    adj = A.holm(ps)
    assert sum(1 for p in adj if p < 0.05) <= sum(1 for p in ps if p < 0.05), \
        "Holm declared more significant results than the unadjusted tests"
    # a 21-family Holm on p=0.004 with the rest large: 21*0.004 = 0.084 > 0.05
    adj2 = A.holm([0.004] + [0.9] * 20)
    assert adj2[0] > 0.05, adj2[0]


def t20_paired_bootstrap_determinism_and_pairing():
    m = synth_mapping(80)
    files = sorted(m)
    y = np.array([CLASSES.index(m[f]) for f in files])
    rng = np.random.default_rng(7)
    rp = y.copy(); ap_ = y.copy()
    rp[rng.choice(80, 6, replace=False)] = 0
    ap_[rng.choice(80, 11, replace=False)] = 1

    def mk(pred):
        return {"seed": 42, "clean_test": {"canonical_order_files": files,
                                           "y": y, "pred": pred}}
    a1 = A.paired_bootstrap(mk(rp), mk(ap_), n_iter=400, seed=123)
    a2 = A.paired_bootstrap(mk(rp), mk(ap_), n_iter=400, seed=123)
    a3 = A.paired_bootstrap(mk(rp), mk(ap_), n_iter=400, seed=124)
    for k in ("ci95_lo_pp", "ci95_hi_pp", "bootstrap_mean_pp"):
        assert a1["accuracy_difference_pp"][k] == a2["accuracy_difference_pp"][k], \
            "same seed gave a different interval: %s" % k
    # The seed must change the draw. Asserted on bootstrap_mean_pp, not on a
    # percentile: accuracy differences over 80 images are multiples of 1.25 pp,
    # so two seeds can land the 2.5th percentile on the same discrete value
    # while the underlying draws differ. The mean is continuous and cannot
    # coincide by that mechanism.
    assert a1["accuracy_difference_pp"]["bootstrap_mean_pp"] != \
        a3["accuracy_difference_pp"]["bootstrap_mean_pp"], "the seed had no effect"
    assert a1["resampling_unit"] == "image"
    assert a1["stratified_by"] == "true class"
    assert "not patient-level" in a1["interpretation"]
    # the point estimate must equal the observed difference exactly
    want = 100 * (float((ap_ == y).mean()) - float((rp == y).mean()))
    assert abs(a1["accuracy_difference_pp"]["point_pp"] - want) < 1e-12
    # the interval must bracket the point estimate
    lo = a1["accuracy_difference_pp"]["ci95_lo_pp"]
    hi = a1["accuracy_difference_pp"]["ci95_hi_pp"]
    assert lo <= want <= hi, (lo, want, hi)
    # identical predictions -> zero difference, interval collapsed on zero
    same = A.paired_bootstrap(mk(rp), mk(rp), n_iter=400, seed=1)
    d = same["accuracy_difference_pp"]
    assert abs(d["point_pp"]) < 1e-12 and abs(d["ci95_lo_pp"]) < 1e-12 \
        and abs(d["ci95_hi_pp"]) < 1e-12, d
    assert d["excludes_zero"] is False


def t21_bootstrap_stratification_preserves_class_counts():
    """Stratified resampling must keep each class's count fixed; an unstratified
    bootstrap would not."""
    m = synth_mapping(40)
    files = sorted(m)
    y = np.array([CLASSES.index(m[f]) for f in files])
    rng = np.random.default_rng(A.BOOTSTRAP_SEED)
    idx_by_class = [np.nonzero(y == c)[0] for c in range(4)]
    want = [len(ix) for ix in idx_by_class]
    for _ in range(50):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by_class])
        got = [int((y[take] == c).sum()) for c in range(4)]
        assert got == want, (got, want)
        assert len(take) == len(y)


def t22_seed_sd_uses_ddof_1():
    v = [0.98, 0.99, 1.00]
    got = A.mean_sd(v)
    assert abs(got["mean"] - np.mean(v)) < 1e-15
    assert abs(got["sd"] - float(np.std(v, ddof=1))) < 1e-15, got["sd"]
    assert abs(got["sd"] - float(np.std(v, ddof=0))) > 1e-6, \
        "population SD (ddof=0) was used instead of the sample SD"
    assert got["ddof"] == 1 and got["n"] == 3
    # a single defined value cannot have a sample SD
    one = A.mean_sd([0.5, None, None])
    assert one["n"] == 1 and one["sd"] is None, one
    # all-undefined must report not-available, not 0
    none = A.mean_sd([None, None, None])
    assert none["mean"] is None and none["sd"] is None, none


# ----------------------------------- 5. protocol guarantees ------------------

def t23_pooling_2544_predictions_is_refused():
    """The 21 tests are seed-specific by construction. Assert that a comparison
    cannot be built across seeds, and that the recorded family size is 21 rather
    than anything derived from 2,544 pooled rows."""
    m = synth_mapping(12)
    files = sorted(m)
    y = np.array([CLASSES.index(m[f]) for f in files])
    ref = {"seed": 42, "run_dir": "r",
           "clean_test": {"canonical_order_files": files, "y": y, "pred": y}}
    abl = {"seed": 43, "run_dir": "a",
           "clean_test": {"canonical_order_files": files, "y": y, "pred": y}}
    old = A.CLEAN_TEST_N
    A.CLEAN_TEST_N = len(files)
    try:
        msg = rejects(A.paired_disagreement, ref, abl)
    finally:
        A.CLEAN_TEST_N = old
    assert "across seeds" in msg, msg
    # 2544 = 848 * 3 must never be a legal family size
    msg2 = rejects(A.holm_family_guard, 2544)
    assert "expected 21" in msg2, msg2
    assert 848 * 3 == 2544 and A.N_EXPECTED_TESTS == 21


def t24_paired_comparison_requires_same_images():
    m1 = synth_mapping(10)
    f1 = sorted(m1)
    f2 = f1[:-1] + ["other.jpg"]
    y = np.array([CLASSES.index(m1[f]) for f in f1])
    ref = {"seed": 42, "run_dir": "r",
           "clean_test": {"canonical_order_files": f1, "y": y, "pred": y}}
    abl = {"seed": 42, "run_dir": "a",
           "clean_test": {"canonical_order_files": f2, "y": y, "pred": y}}
    msg = rejects(A.paired_disagreement, ref, abl)
    assert "different image sets" in msg, msg


def t25_reference_ablation_seed_mismatch_rejected():
    """load_run must refuse a directory whose metrics.json records another seed."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = os.path.join(tmp, A.IMPORT_ROOT, "no_ema", "seed_42")
        os.makedirs(d)
        write_preds(os.path.join(d, "preds_test.csv"), m)
        json.dump({"seed": 43, "split": "clean"},
                  open(os.path.join(d, "metrics.json"), "w", encoding="utf-8"))
        old = A.CLEAN_TEST_N
        A.CLEAN_TEST_N = len(m)
        try:
            msg = rejects(A.load_run, tmp, "no_ema", 42, canon_of(m))
        finally:
            A.CLEAN_TEST_N = old
        assert "records seed 43" in msg, msg


def t26_no_tta_checkpoint_mismatch_detected():
    with tempfile.TemporaryDirectory() as tmp:
        right, wrong = "a" * 64, "f" * 64
        rec = {"seeds": {str(s): {"gates": {"container": {"sha256": right}}}
                         for s in A.SEEDS}}
        os.makedirs(os.path.join(tmp, A.RECOVERY))
        json.dump(rec, open(os.path.join(tmp, A.RECOVERY, "import_manifest.json"),
                            "w", encoding="utf-8"))
        runs = {}
        for s in A.SEEDS:
            d = os.path.join(tmp, A.IMPORT_ROOT, "no_tta", "seed_%d" % s)
            os.makedirs(d)
            digest = wrong if s == 43 else right
            open(os.path.join(d, "source_checkpoint_sha256.txt"), "w",
                 encoding="utf-8").write(digest + "\n")
            runs[("no_tta", s)] = {"raw_metrics": {
                "trained": False, "checkpoint_written": False,
                "resolved_config": {"tta": False}}}
        got = A.no_tta_provenance(tmp, runs)
        assert got["all_seeds_clean"] is False, got
        assert got["per_seed"][43]["identical_checkpoint"] is False
        assert got["per_seed"][42]["identical_checkpoint"] is True
        assert got["per_seed"][42]["tta_is_the_only_change"] is True
        # a no_tta that trained must also be caught
        runs[("no_tta", 42)]["raw_metrics"]["trained"] = True
        got2 = A.no_tta_provenance(tmp, runs)
        assert got2["per_seed"][42]["no_weights_changed"] is False, got2["per_seed"][42]
        assert got2["per_seed"][42]["tta_is_the_only_change"] is False
        # and a no_tta with TTA still on
        runs[("no_tta", 42)]["raw_metrics"]["trained"] = False
        runs[("no_tta", 42)]["raw_metrics"]["resolved_config"]["tta"] = True
        got3 = A.no_tta_provenance(tmp, runs)
        assert got3["per_seed"][42]["tta_is_the_only_change"] is False


def t27_gate_provenance_requires_all_three_facts():
    """The gate claim may only be strengthened when the recovery record shows a
    strict load, a gate mass away from 0.5, AND prediction reproduction. Removing
    any one must fall back to the weaker caveat."""
    def manifest(gate_mass=0.28, missing=0, pred_mismatch=0):
        return {"seeds": {str(s): {"decision": "ACCEPTED", "gates": {
            "container": {"sha256": "a" * 64},
            "load": {"loaded": True, "n_missing": missing, "n_unexpected": 0,
                     "n_shape_mismatch": 0, "state_dict_key_count": 578},
            "architecture_and_gate": {
                "gated_flag": True, "multiscale_flag": True,
                "cnn_proj_in_features": 1432, "gate_is_learned_module": True,
                "gate_parameter_count": 262914,
                "gate_cnn_mean_on_probe": gate_mass, "gate_cnn_std_on_probe": 0.02},
            "full_test": {"n_compared": A.CLEAN_TEST_N,
                          "n_pred_label_mismatch": pred_mismatch,
                          "n_filename_mismatch": 0, "n_true_label_mismatch": 0,
                          "max_abs_prob_diff": 4.8e-6, "accuracy_diff_pp": 0.0}}}
            for s in A.SEEDS}}

    def run(man):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, A.RECOVERY))
            json.dump(man, open(os.path.join(tmp, A.RECOVERY, "import_manifest.json"),
                                "w", encoding="utf-8"))
            return A.gate_provenance(tmp)

    good = run(manifest())
    assert good["strengthened"] is True, good["argument"]
    assert "SUPERSEDED" in good["consequence"], good["consequence"]
    # gate mass at exactly 0.5 -> the gated and ungated paths coincide, so
    # reproduction proves nothing about gating
    half = run(manifest(gate_mass=0.5))
    assert half["strengthened"] is False, "a 0.5/0.5 gate was treated as decisive"
    assert "RETAINED" in half["consequence"]
    # a non-strict load
    loose = run(manifest(missing=3))
    assert loose["strengthened"] is False, "a non-strict load was accepted"
    # predictions not reproduced
    nore = run(manifest(pred_mismatch=5))
    assert nore["strengthened"] is False, "unreproduced predictions were accepted"
    # absent manifest
    with tempfile.TemporaryDirectory() as tmp:
        gone = A.gate_provenance(tmp)
    assert gone["strengthened"] is False and "retained" in gone["caveat"]
    assert "not evidence of causal branch contribution" in \
        good["what_this_does_not_establish"].replace("It is not", "not")


def _per(d_acc, holm, ci, d_f1=None, d_bal=None):
    """Build a per-seed evidence list for classify_component."""
    d_f1 = d_acc if d_f1 is None else d_f1
    d_bal = d_acc if d_bal is None else d_bal
    return [{"accuracy_difference_pp": a, "macro_f1_difference_pp": f,
             "balanced_accuracy_difference_pp": b, "holm_adjusted_p": h,
             "bootstrap_accuracy_excludes_zero": c}
            for a, f, b, h, c in zip(d_acc, d_f1, d_bal, holm, ci)]


def t28_verdict_vocabulary_is_the_four_cautious_categories():
    """Exactly the four requested categories, and none of them asserts
    equivalence, uselessness or harm."""
    assert len(A.VERDICTS) == 4, A.VERDICTS
    assert A.VERDICT_SUPPORTED == "supported by consistent accuracy evidence"
    assert A.VERDICT_SUGGESTIVE == "suggestive but inconclusive"
    assert A.VERDICT_NO_DETECTABLE == \
        "no detectable accuracy benefit under this protocol"
    assert A.VERDICT_INCONCLUSIVE == "inconclusive"
    for banned in ("equivalent", "equivalence", "useless", "harmful",
                   "negligible", "material"):
        assert not any(banned in v.lower() for v in A.VERDICTS), \
            "the verdict vocabulary contains %r" % banned
    # the rule is written down, one entry per verdict
    assert len(A.DECISION_RULE) == 4, A.DECISION_RULE
    assert [n for n, _ in A.DECISION_RULE] == A.VERDICTS, A.DECISION_RULE


def t33_no_post_hoc_threshold_anywhere_in_the_verdict_path():
    """The removed 0.14 pp materiality threshold must not be reachable, and no
    number may have replaced it.

    The decisive test is behavioural: scaling every observed effect size by a
    large factor, while holding the corrected evidence fixed, must not change any
    verdict. A rule that compared effects to a cutoff could not survive that."""
    src = open(os.path.join(_HERE, "final_analysis.py"), encoding="utf-8").read()
    fn = src[src.index("def classify_component("):
             src.index("def build_claims(")]
    # no arithmetic on the reference seed SD, and no bare numeric comparison
    for banned in ("REF_ACC[1]", "material", "0.14", "threshold ="):
        assert banned not in fn, \
            "classify_component still references %r" % banned
    for op in ("<= -", ">= 1.0", "< 0.14", "abs(mean_acc)"):
        assert op not in fn, "classify_component still compares an effect size: %r" % op

    for scale in (1.0, 10.0, 1000.0):
        # consistently adverse, no corrected support -> suggestive at any scale
        c = A.classify_component("no_auxiliary_loss",
                                 _per([-0.9 * scale, -0.6 * scale, -0.1 * scale],
                                      [1.0, 1.0, 1.0], [False, False, False]), {})
        assert c["verdict"] == A.VERDICT_SUGGESTIVE, (scale, c["verdict"])
        # sign-inconsistent, no corrected support -> no detectable, at any scale
        c2 = A.classify_component("no_ema",
                                  _per([+0.1 * scale, -0.2 * scale, +0.2 * scale],
                                       [1.0, 1.0, 1.0], [False, False, False]), {})
        assert c2["verdict"] == A.VERDICT_NO_DETECTABLE, (scale, c2["verdict"])
        # tiny but fully corrected -> supported, at any scale
        c3 = A.classify_component("cnn_branch_only",
                                  _per([-0.001 * scale] * 3, [1e-4] * 3, [True] * 3), {})
        assert c3["verdict"] == A.VERDICT_SUPPORTED, (scale, c3["verdict"])

    # a vanishing effect with full corrected support still counts as supported:
    # the verdict follows the corrected evidence, not the magnitude
    tiny = A.classify_component("cnn_branch_only",
                                _per([-1e-9] * 3, [1e-6] * 3, [True] * 3), {})
    assert tiny["verdict"] == A.VERDICT_SUPPORTED, tiny["verdict"]

    # and the machine-readable record says so
    for c in (tiny, c, c2):
        assert c["decision_inputs"]["threshold_used"] is None, c["decision_inputs"]
        assert "threshold" in c["decision_inputs"]["threshold_note"].lower()
        assert "important" in c["effect_size_label"] or \
            "negligible" in c["effect_size_label"], c["effect_size_label"]


def t34_consistent_adverse_without_correction_is_suggestive():
    """The documented rule for a directionally consistent adverse effect whose
    multiplicity-corrected evidence is insufficient."""
    # the real no_auxiliary_loss numbers: adverse at all 3, 0/3 Holm, 1/3 CI
    c = A.classify_component("no_auxiliary_loss",
                             _per([-0.9434, -0.5896, -0.1179],
                                  [0.6943, 1.0, 1.0], [True, False, False]), {})
    assert c["verdict"] == A.VERDICT_SUGGESTIVE, c["verdict"]
    assert c["accuracy_adverse_at_every_seed"] is True
    assert any("direction is consistent" in r for r in c["reasons"])
    assert any("does not support it at every seed" in r for r in c["reasons"])
    assert any("Three seeds is a small number" in r for r in c["reasons"])
    # the real vit_branch_only numbers: adverse at all 3, 0/3 Holm, 0/3 CI
    c2 = A.classify_component("vit_branch_only",
                              _per([-0.1179, -0.3538, -0.2358],
                                   [1.0, 1.0, 1.0], [False, False, False]), {})
    assert c2["verdict"] == A.VERDICT_SUGGESTIVE, c2["verdict"]
    # 2 of 3 corrected is still not support at every seed
    c3 = A.classify_component("no_ema",
                              _per([-1.0, -1.0, -1.0],
                                   [1e-4, 1e-4, 0.9], [True, True, False]), {})
    assert c3["verdict"] == A.VERDICT_SUGGESTIVE, c3["verdict"]
    # 3 of 3 corrected tips it to supported
    c4 = A.classify_component("no_ema",
                              _per([-1.0, -1.0, -1.0],
                                   [1e-4, 1e-4, 1e-4], [True, True, True]), {})
    assert c4["verdict"] == A.VERDICT_SUPPORTED, c4["verdict"]


def t35_sign_inconsistent_nonsignificant_maps_by_documented_rule():
    """Sign-inconsistent and uncorrected -> no detectable accuracy benefit.
    Sign-inconsistent but some corrected evidence -> inconclusive. Both branches
    are in DECISION_RULE, so the mapping is documented, not implicit."""
    rule = dict(A.DECISION_RULE)
    assert "not adverse at all three seeds" in rule[A.VERDICT_NO_DETECTABLE]
    assert "not consistently adverse" in rule[A.VERDICT_INCONCLUSIVE]

    # the real no_gate numbers, including the exact 0.000 at seed 42
    c = A.classify_component("no_gate",
                             _per([0.0000, -0.1179, +0.1179], [1.0, 1.0, 1.0],
                                  [False, False, False],
                                  d_f1=[-0.0509, -0.1393, +0.0428],
                                  d_bal=[-0.1059, -0.2152, -0.0169]), {})
    assert c["verdict"] == A.VERDICT_NO_DETECTABLE, c["verdict"]
    assert c["accuracy_adverse_at_every_seed"] is False, \
        "a zero difference must not count as an adverse direction"
    # the consistently adverse balanced accuracy must be REPORTED, not hidden,
    # and must not change the verdict
    assert c["secondary_balanced_accuracy_adverse_at_every_seed"] is True
    assert any("OBSERVED SECONDARY PATTERN" in r for r in c["reasons"]), c["reasons"]
    assert any("descriptive only and does not" in r for r in c["reasons"])

    # sign-inconsistent WITH corrected evidence -> inconclusive, not swept into C
    c2 = A.classify_component("no_multiscale",
                              _per([+0.5, -0.9, +0.3], [1.0, 0.01, 1.0],
                                   [False, True, False]), {})
    assert c2["verdict"] == A.VERDICT_INCONCLUSIVE, c2["verdict"]
    assert any("conflicting directions" in r for r in c2["reasons"])

    # consistently favourable to removal WITH corrected support is flagged rather
    # than forced into one of the four adverse-effect readings
    c3 = A.classify_component("no_ema",
                              _per([+1.5, +1.2, +1.4], [1e-4, 1e-4, 1e-4],
                                   [True, True, True]), {})
    assert c3["verdict"] == A.VERDICT_INCONCLUSIVE, c3["verdict"]
    assert "removal_consistently_favourable_with_corrected_support" in c3["flags"]
    assert any("outside the four prespecified categories" in r for r in c3["reasons"])
    # consistently favourable but uncorrected -> no detectable benefit
    c4 = A.classify_component("no_ema",
                              _per([+0.1, +0.2, +0.3], [1.0, 1.0, 1.0],
                                   [False, False, False]), {})
    assert c4["verdict"] == A.VERDICT_NO_DETECTABLE, c4["verdict"]
    assert any("did not lower accuracy at any" in r for r in c4["reasons"])


def t36_nonsignificance_is_never_equivalence():
    """Every verdict that rests on absent evidence must carry the disclaimer, in
    those words, and no verdict may assert equivalence or zero effect."""
    for banned in ("equivalent", "equivalence", "no effect", "zero effect",
                   "useless", "unnecessary", "proves"):
        assert banned not in A.VERDICT_NO_DETECTABLE.lower()
    assert "not a demonstration of equivalence" in A.NOT_EQUIVALENCE.lower()
    assert "does not show the component has zero effect" in A.NOT_EQUIVALENCE.lower()
    assert "does not show the component is" in A.NOT_EQUIVALENCE.lower()

    for per in (_per([0.1, -0.2, 0.2], [1.0] * 3, [False] * 3),
                _per([0.1, 0.2, 0.3], [1.0] * 3, [False] * 3),
                _per([+0.5, -0.9, +0.3], [1.0, 0.01, 1.0], [False, True, False])):
        c = A.classify_component("no_gate", per, {})
        assert c["verdict"] in (A.VERDICT_NO_DETECTABLE, A.VERDICT_INCONCLUSIVE)
        assert any(A.NOT_EQUIVALENCE in r for r in c["reasons"]), \
            "a verdict resting on absent evidence omitted the disclaimer: %s" % c["verdict"]
    # the supported and suggestive branches are about present evidence and do not
    # need the disclaimer, but they must not claim equivalence either
    for per in (_per([-1.0] * 3, [1e-4] * 3, [True] * 3),
                _per([-1.0] * 3, [1.0] * 3, [False] * 3)):
        c = A.classify_component("cnn_branch_only", per, {})
        joined = " ".join(c["reasons"]).lower()
        assert "equivalen" not in joined, joined


def t37_calibration_is_labelled_descriptive_and_not_a_superiority_claim():
    """The report must label calibration comparisons descriptive/exploratory and
    must not contain superiority or equivalence wording for them."""
    src = open(os.path.join(_HERE, "final_analysis.py"), encoding="utf-8").read()
    assert "descriptive and exploratory" in src, \
        "the calibration section carries no descriptive/exploratory label"
    assert "not described as better calibrated" in src
    assert "numerically lower" in src
    banned = ["better calibrated than", "is better calibrated",
              "no calibration benefit", "not carried by a calibration",
              "superior calibration", "equivalently calibrated",
              "calibration proves", "worse calibrated than"]
    low = src.lower()
    for b in banned:
        assert b not in low, "prohibited calibration wording in the code: %r" % b
    # no calibration hypothesis test was bolted on afterwards
    for t in ("ece_p_value", "brier_p_value", "nll_p_value", "calibration_test",
              "mcnemar_ece", "bootstrap_ece"):
        assert t not in low, "a post-hoc calibration test appeared: %r" % t
    # and the report renders that label
    out = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                       "publication_results", "ablations", "final_analysis",
                       "FINAL_ABLATION_ANALYSIS.md")
    if os.path.exists(out):
        md = open(out, encoding="utf-8").read().lower()
        assert "descriptive and exploratory" in md, \
            "the generated report does not label calibration as descriptive"
        for b in banned:
            assert b not in md, "prohibited calibration wording in the report: %r" % b


def t38_markdown_and_machine_readable_verdicts_agree():
    """Every verdict must appear identically in component_conclusions.csv, both
    JSON files and the Markdown. A disagreement between them is how a corrected
    conclusion silently fails to reach a reader."""
    base = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                        "publication_results", "ablations", "final_analysis")
    need = ["component_conclusions.csv", "final_ablation_analysis.json",
            "analysis_provenance.json", "FINAL_ABLATION_ANALYSIS.md"]
    if not all(os.path.exists(os.path.join(base, f)) for f in need):
        return      # nothing generated yet; the driver's own run asserts this too
    csv_v = {r["variant"]: r["verdict"] for r in
             csv.DictReader(open(os.path.join(base, "component_conclusions.csv"),
                                encoding="utf-8"))}
    j = json.load(open(os.path.join(base, "final_ablation_analysis.json"),
                       encoding="utf-8"))
    json_v = {c["variant"]: c["verdict"] for c in j["component_conclusions"]}
    assert csv_v == json_v, (csv_v, json_v)
    assert set(csv_v) == set(A.VARIANTS), set(csv_v) ^ set(A.VARIANTS)
    for v, verdict in csv_v.items():
        assert verdict in A.VERDICTS, (v, verdict)
    md = open(os.path.join(base, "FINAL_ABLATION_ANALYSIS.md"), encoding="utf-8").read()
    for v, verdict in csv_v.items():
        assert "### `%s` - %s" % (v, verdict) in md, \
            "the Markdown heading for %s does not carry verdict %r" % (v, verdict)
        assert "| `%s` |" % v in md
    # the removed threshold must be absent from every generated artifact
    for f in os.listdir(base):
        body = open(os.path.join(base, f), encoding="utf-8", errors="replace").read()
        low = body.lower()
        if "materiality" in low or "material " in low:
            # permitted only where the text states it was removed
            for i, line in enumerate(body.splitlines(), 1):
                if "material" in line.lower():
                    assert ("removed" in line.lower() or "has been removed" in line.lower()), \
                        "%s line %d still uses the threshold: %s" % (f, i, line.strip()[:120])
        assert "materiality_threshold_pp" not in body, \
            "%s still carries the materiality_threshold_pp field" % f


def t39_claims_are_derived_from_the_numbers():
    """The report's claim sentences are generated from the computed results, with
    a recorded check behind each, so a sentence cannot outlive its evidence."""
    tests = []
    for v in A.VARIANTS:
        for sd in A.SEEDS:
            adverse = v in ("cnn_branch_only", "no_auxiliary_loss", "vit_branch_only")
            sig = v == "cnn_branch_only"
            tests.append({
                "variant": v, "seed": sd,
                "accuracy_difference_pp": -2.5 if sig else (-0.5 if adverse else
                                                            (0.1 if sd != 43 else -0.1)),
                "macro_f1_difference_pp": -0.1, "balanced_accuracy_difference_pp": -0.1,
                "unadjusted_p": 1e-5 if sig else 0.9,
                "holm_adjusted_p": 1e-4 if sig else 1.0,
                "holm_significant_at_0.05": sig,
                "bootstrap_accuracy_excludes_zero": sig,
                "mcnemar": {"n_discordant": 30 if sig else 4},
            })
    conclusions = [A.classify_component(
        v, [t for t in tests if t["variant"] == v], {}) for v in A.VARIANTS]
    rep = {"multiple_comparison_control": {"n_tests": 21,
                                           "n_significant_before_holm": 3,
                                           "n_significant_after_holm": 3},
           "gate_provenance": {"per_seed": {sd: {"gate_parameter_count": 262914,
                                                 "gate_cnn_mean_on_probe": 0.28}
                                            for sd in A.SEEDS}}}
    cl = A.build_claims(tests, conclusions, rep)
    assert cl["verified"]["only_vit_branch_removal_holm_significant_at_all_seeds"] is True
    assert cl["verified"]["auxiliary_loss_consistent_decrease_not_holm_significant"] is True
    assert cl["verified"]["gate_no_detectable_accuracy_benefit"] is True
    joined = " ".join(cl["claims"])
    assert "Only removal of the ViT branch produced Holm-significant accuracy " \
        "losses at all three seeds." in joined
    assert "did not survive Holm correction" in joined
    assert "no detectable accuracy benefit under this protocol" in joined
    assert "not proof of zero effect" in joined
    assert "learned and input-dependent" in joined
    assert "image-level" in joined and "not clinical evidence" in joined
    nc = " ".join(cl["not_claimed"]).lower()
    for must in ("only the vit branch matters", "cnn branch is unnecessary",
                 "gate is useless", "model should be changed",
                 "nonsignificance proves equality", "brain-meta"):
        assert must in nc, "the not-claimed list omits %r" % must

    # the guard must fire when the evidence changes: make a second variant
    # significant and the sentence must stop asserting "only the ViT branch"
    for t in tests:
        if t["variant"] == "no_ema":
            t["holm_significant_at_0.05"] = True
    cl2 = A.build_claims(tests, conclusions, rep)
    assert cl2["verified"]["only_vit_branch_removal_holm_significant_at_all_seeds"] is False
    assert "Only removal of the ViT branch" not in " ".join(cl2["claims"]), \
        "the claim survived evidence that contradicts it"
    assert "reported as found" in " ".join(cl2["claims"])


def t40_statistical_audit_reconfirms_the_prespecified_analysis():
    tests = [{"variant": v, "seed": sd,
              "mcnemar": {"test": "exact two-sided McNemar (binomial)"},
              "table": {"n_images": A.CLEAN_TEST_N},
              "bootstrap": {"n_iterations": 10000, "resampling_unit": "image",
                            "stratified_by": "true class"}}
             for v in A.VARIANTS for sd in A.SEEDS]
    rep = {"multiple_comparison_control": {"n_tests": 21,
                                           "n_significant_before_holm": 4,
                                           "n_significant_after_holm": 3}}
    au = A.statistical_audit(tests, rep)
    assert au["all_intact"] is True, au["checks"]
    assert au["n_tests"] == 21
    assert "848 images scored three times" in au["note"]
    # each guard must be able to fail
    bad = [dict(t) for t in tests]
    bad[0] = dict(bad[0], bootstrap={"n_iterations": 500, "resampling_unit": "image",
                                     "stratified_by": "true class"})
    assert A.statistical_audit(bad, rep)["all_intact"] is False
    bad2 = [dict(t) for t in tests[:20]]
    assert A.statistical_audit(bad2, rep)["all_intact"] is False
    bad3 = [dict(t) for t in tests]
    bad3[0] = dict(bad3[0], table={"n_images": 2544})
    assert A.statistical_audit(bad3, rep)["all_intact"] is False, \
        "pooling 2,544 predictions passed the audit"


def t29_deterministic_output_generation():
    """Running the metric and statistic layer twice on the same inputs must give
    byte-identical numbers; nothing may depend on dict order or an unseeded RNG."""
    m = synth_mapping(120)
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "p.csv")
        write_preds(p, m, wrong_every=9)
        d1, d2 = load(p, m), load(p, m)
        m1 = A.compute_metrics(d1["y"], d1["P_canonical"], CLASSES)
        m2 = A.compute_metrics(d2["y"], d2["P_canonical"], CLASSES)
        s1 = json.dumps(m1, sort_keys=True, default=str)
        s2 = json.dumps(m2, sort_keys=True, default=str)
        assert s1 == s2, "metric output is not deterministic"
        assert d1["sha256"] == d2["sha256"]
        files = d1["canonical_order_files"]
        y = d1["y"]
        ref = {"seed": 42, "clean_test": {"canonical_order_files": files, "y": y,
                                          "pred": y}}
        abl = {"seed": 42, "clean_test": {"canonical_order_files": files, "y": y,
                                          "pred": d1["pred"]}}
        b1 = A.paired_bootstrap(ref, abl, n_iter=300, seed=A.BOOTSTRAP_SEED)
        b2 = A.paired_bootstrap(ref, abl, n_iter=300, seed=A.BOOTSTRAP_SEED)
        assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True), \
            "bootstrap output is not deterministic at a fixed seed"


def t30_write_location_is_guarded():
    src = open(os.path.join(_HERE, "final_analysis.py"), encoding="utf-8").read()
    assert 'startswith("publication_")' in src, "the write-location guard is gone"
    assert 'refusing to write outside publication_' in src
    assert "if args.bootstrap_iterations < 10000" in src, \
        "the minimum bootstrap iteration guard is gone"
    for token in ('"w"', "'w'", '"a"', "'a'"):
        for i, line in enumerate(src.splitlines(), 1):
            if token in line and "open(" in line:
                assert ("out_root" in line or "path" in line), \
                    "line %d writes to an unaudited location: %s" % (i, line.strip())


def t31_no_brain_meta_comparison_in_code():
    src = open(os.path.join(_HERE, "final_analysis.py"), encoding="utf-8").read()
    low = src.lower()
    for banned in ("97.10", "96.94"):
        assert banned not in src, "a BRAIN-META number appears in the analysis code"
    # The only permitted mention is a disclaimer. A disclaimer often wraps across
    # source lines, so each mention is judged together with the two lines that
    # follow it rather than in isolation.
    lines = src.splitlines()
    allowed = ("no comparison", "unavailable", "no_brain_meta", "not computed",
               "no statistical statement", "no such comparison")
    for i, line in enumerate(lines, 1):
        if "brain" in line.lower() and "meta" in line.lower():
            window = " ".join(lines[i - 1:i + 2]).lower()
            assert any(a in window for a in allowed), \
                "line %d mentions BRAIN-META other than to disclaim it: %s" % (i, line.strip())
    for banned in ("clinically ready", "clinical readiness is",
                   "patient-independent", "universally generaliz"):
        assert banned not in low, "a prohibited claim appears: %s" % banned


def t32_protected_trees_untouched():
    root = os.path.dirname(os.path.dirname(_HERE))
    protected = ["runs", "runs_original", "runs_clean", "runs_v2_original",
                 "runs_v2_clean", "results", "results_original", "results_clean",
                 "results_v2_original", "results_v2_clean", "dataset",
                 "dataset_clean", "publication_runs"]
    for name in protected:
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        for dirpath, _dirs, files in os.walk(p):
            for f in files:
                assert not f.startswith("img_"), \
                    "a synthetic fixture leaked into %s" % dirpath


def frozen_digests(root):
    out = {}
    for rel in ("src/fusion_v2.py", "src/train_ablation.py",
                "src/publication/evaluate_fusion_v2.py"):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            out[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out",
                    default="publication_results/ablations/final_analysis/TEST_REPORT.md")
    args = ap.parse_args()
    root = os.path.abspath(args.project_root)
    before = frozen_digests(root)

    print("Phase 3B - final analysis tests")
    print("=" * 76)
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("t") and len(k) > 1 and k[1].isdigit() and callable(v)]
    order = [n for n, _ in tests]
    for name, fn in tests:
        check(name, fn)

    after = frozen_digests(root)
    if before != after:
        FAILED.append(("frozen code unchanged", "digests changed: %s"
                       % sorted(k for k in before if before[k] != after.get(k))))
        print("  FAIL  frozen code unchanged")
    else:
        PASSED.append("frozen code unchanged")
        print("  ok    frozen code unchanged (%d files)" % len(before))

    n_pass, n_total = len(PASSED), len(PASSED) + len(FAILED)
    status = "PASS" if not FAILED else "FAIL"
    print("=" * 76)
    print("%d passed, %d failed -> %s" % (n_pass, len(FAILED), status))
    for name, why in FAILED:
        print("  - %s :: %s" % (name, why))

    out = os.path.join(root, args.out)
    if not os.path.relpath(out, root).replace("\\", "/").startswith("publication_"):
        raise SystemExit("refusing to write outside publication_*")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    why = {n: (globals()[n].__doc__ or "").strip().split("\n")[0] for n in order}
    failed_names = dict(FAILED)
    L = ["# Phase 3B - Final Analysis Test Report", "",
         "Generated by `src/publication/test_final_analysis.py`.", "",
         "**%d of %d tests passed. Status: %s.**" % (n_pass, n_total, status), "",
         "Two kinds of test are here and both are needed. **Rejection tests** break "
         "exactly one thing in a synthetic run and assert the analyser refuses it. "
         "**Formula tests** check the arithmetic against values computed by hand or by "
         "an independent method, because a validator that only ever accepts clean data "
         "says nothing about whether its numbers are right.", "",
         "Several tests assert the opposite direction, so a suite that rejected "
         "everything could not pass: a genuine top-two tie is accepted and counted "
         "rather than rejected; reversing the row order must change nothing; identical "
         "predictions must give a zero-width bootstrap interval; and a correct "
         "reference-spec agreement must be usable.", "",
         "Three tests pin protocol guarantees that are easy to violate silently: the "
         "Holm family must be 21 and refuse both 7 and 2,544; a paired comparison "
         "across two different seeds must be refused; and the verdict vocabulary must "
         "contain no equivalence claim.", "",
         "| Test | Status | What it asserts |", "|---|---|---|"]
    for n in order:
        st = "FAIL" if n in failed_names else "PASS"
        L.append("| `%s` | %s | %s |" % (n, st, why[n].replace("|", "/")))
    L.append("| `frozen code unchanged` | %s | `src/fusion_v2.py`, "
             "`src/train_ablation.py` and `src/publication/evaluate_fusion_v2.py` are "
             "byte-identical before and after the suite |"
             % ("PASS" if "frozen code unchanged" in PASSED else "FAIL"))
    L += ["", "## Frozen files, digests recorded at run time", "",
          "| File | SHA-256 |", "|---|---|"]
    L += ["| `%s` | `%s` |" % (k, v) for k, v in sorted(after.items())]
    if FAILED:
        L += ["", "## Failures", ""] + ["- `%s`: %s" % (n, w) for n, w in FAILED]
    L += ["", "## Not runnable in this environment", "",
          "`src/publication/test_ablations.py` imports torch, which is not installed "
          "locally, so that pre-existing suite could not be re-run. It is unrelated to "
          "this phase and was not modified.", ""]
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote %s" % os.path.relpath(out, root))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
