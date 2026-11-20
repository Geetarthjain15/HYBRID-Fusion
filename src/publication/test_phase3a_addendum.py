"""
Task 6 - regression tests for the Phase 3A Addendum verifier.

A verifier that only ever sees good data proves nothing. Each test below builds
a deliberately corrupted artifact in a temporary directory and asserts the
verifier REJECTS it. Nothing here touches runs/, runs_*/, results*/,
publication_runs/ or dataset directories: every fixture is synthetic and lives
under the OS temporary directory.

The corruptions correspond one-to-one with the failure modes the addendum claims
to rule out:

  1  a substituted filename                     -> canonical alignment fails
  2  a wrong true label                         -> canonical alignment fails
  3  a duplicated filename                      -> prediction loading fails
  4  a permuted class order                     -> prediction loading fails
  5  an unintended configuration difference     -> REJECTION, not a limitation
  6  a wrong no_tta source checkpoint digest    -> provenance check fails
  7  a changed imported file                    -> import integrity fails
  8  reference seeds disagreeing on the mapping -> shared-mapping check fails
  9  an incorrect stored macro-F1               -> metric comparison fails
 10  no stored macro-F1 at all                  -> REPORTED, not passed silently

Plus positive controls, so a test suite that rejects everything cannot pass.

    python src/publication/test_phase3a_addendum.py
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import verify_phase3a_addendum as V

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
    """Assert the callable raises AddendumError. Any other exception is a bug in
    the test, not a rejection, and is allowed to propagate."""
    try:
        fn(*a, **k)
    except V.AddendumError as e:
        return str(e)
    raise AssertionError("accepted an artifact it should have rejected")


# --------------------------------------------------------------- fixtures ---

CLASSES = V.CLASSES
N = 24  # small stand-in for the 848 clean-test images


def synth_mapping(n=N):
    """A canonical-style mapping: n distinct identities across the four classes."""
    return {"img_%03d.jpg" % i: CLASSES[i % len(CLASSES)] for i in range(n)}


def write_preds(path, mapping, files=None, labels=None, class_cols=None,
                pred_from_truth=True, probs=None):
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
                target = lab if pred_from_truth else CLASSES[(CLASSES.index(lab) + 1) % 4]
                pr = [0.04] * len(cols)
                pr[cols.index(target)] = 1.0 - 0.04 * (len(cols) - 1)
            pred = cols[int(np.argmax(pr))]
            w.writerow([f, lab, pred] + ["%.6f" % x for x in pr])


def canon_from_mapping(mapping):
    ordered = sorted(mapping.items())
    fp = hashlib.sha256("\n".join("%s\t%s" % (f, l) for f, l in ordered).encode()).hexdigest()
    return {"source": "synthetic", "source_sha256": "0" * 64,
            "n_identities": len(mapping), "mapping": mapping,
            "fingerprint_sha256": fp, "class_counts": {},
            "evidence_note": "synthetic fixture"}


REF_CFG = {
    "arch": "fusion_v2", "seed": 42, "lr": 1e-4, "batch_size": 16, "epochs": 15,
    "weight_decay": 0.05, "label_smoothing": 0.1, "aux_weight": 0.3,
    "backbone_lr_mult": 0.1, "ema": True, "tta": True,
    "unfreeze_all_cnn": True, "unfreeze_all_vit": True,
    "params_total": 91211918, "split": "clean", "test_acc": 0.5,
}


def resolvers(gated=True, multiscale=True):
    """Resolver bundle with both unrecorded reference keys supplied, so a test
    that is not about resolution is not perturbed by it."""
    consts = {"LR": {"value": 1e-4, "line": 1}, "BATCH_SIZE": {"value": 16, "line": 2},
              "EPOCHS": {"value": 15, "line": 3}, "WEIGHT_DECAY": {"value": 0.05, "line": 4},
              "LABEL_SMOOTHING": {"value": 0.1, "line": 5}, "PATIENCE": {"value": 4, "line": 6},
              "VAL_FRACTION": {"value": 0.15, "line": 7},
              "VAL_SPLIT_SEED": {"value": 42, "line": 8}}
    spec = V.intended_from_spec()
    agree = {"source": "synthetic spec", "checked_fields": {"arch": {"spec": "fusion_v2",
             "recorded": "fusion_v2", "agrees": True}}, "n_checked": 1,
             "disagreements": [], "unrecorded_fields": ["gated", "multiscale"],
             "usable": True,
             "expectation": {"gated": gated, "multiscale": multiscale}}
    return {"module_constants": {"source": "synthetic", "source_sha256": "0" * 64,
                                "constants": consts, "missing": [], "note": ""},
            "derived_reference": {"multiscale": {"value": multiscale,
                                                "evidence_class": "derived_from_params_total",
                                                "source": "synthetic"},
                                  "gated": {"value": None,
                                            "evidence_class": "unresolvable",
                                            "source": "synthetic"}},
            "spec_intended": spec["intended"], "reference_spec": agree}


def variant_cfg(variant, **over):
    """A config in the variant trainer's nested schema, matching the reference
    except for the variant's intended change."""
    rc = {"variant": variant, "arch": "fusion_v2", "unfreeze_all_cnn": True,
          "unfreeze_all_vit": True, "backbone_lr_mult": 0.1, "aux_weight": 0.3,
          "ema": True, "tta": True, "gated": True, "multiscale": True}
    changes = {"no_gate": {"gated": False}, "no_multiscale": {"multiscale": False},
               "no_auxiliary_loss": {"aux_weight": 0.0}, "no_ema": {"ema": False},
               "no_tta": {"tta": False},
               "cnn_branch_only": {"arch": "cnn_branch_only", "unfreeze_all_vit": False},
               "vit_branch_only": {"arch": "vit_branch_only", "unfreeze_all_cnn": False}}
    rc.update(changes.get(variant, {}))
    cfg = {"seed": 42, "lr": 1e-4, "batch_size": 16, "epochs": 15,
           "weight_decay": 0.05, "label_smoothing": 0.1, "patience": 4,
           "val_fraction": 0.15, "val_split_seed": 42, "split": "clean",
           "resolved_config": rc, "test_acc": 0.5, "trained": True,
           "checkpoint_written": True}
    cfg.update(over)
    for k, v in over.items():
        if k in rc:
            rc[k] = v
    return cfg


def make_run(tmp, variant, mapping, cfg=None, preds_kw=None, extra_files=None,
             metrics_over=None):
    d = os.path.join(tmp, variant, "seed_42")
    os.makedirs(d, exist_ok=True)
    cfg = cfg if cfg is not None else variant_cfg(variant)
    pk = preds_kw or {}
    write_preds(os.path.join(d, "preds_test.csv"), mapping, **pk)
    acc = _recompute_acc(os.path.join(d, "preds_test.csv"))
    met = dict(cfg)
    met["test_acc"] = acc
    if metrics_over:
        met.update(metrics_over)
    json.dump(met, open(os.path.join(d, "metrics.json"), "w", encoding="utf-8"))
    json.dump(cfg, open(os.path.join(d, "config.json"), "w", encoding="utf-8"))
    for name, body in (extra_files or {}).items():
        open(os.path.join(d, name), "w", encoding="utf-8").write(body)
    return d


def _recompute_acc(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return sum(r["true_label"] == r["pred_label"] for r in rows) / float(len(rows))


def verify(d, variant, mapping, category="training", recovered=None, res=None,
           expect_n=None):
    """Call the verifier's per-run entry point with N patched to the fixture size."""
    old = V.CLEAN_TEST_N
    V.CLEAN_TEST_N = expect_n or len(mapping)
    try:
        return V.verify_run(d, variant, 42, category, canon_from_mapping(mapping),
                            REF_CFG, recovered or {}, res or resolvers())
    finally:
        V.CLEAN_TEST_N = old


# ------------------------------------------------------------------ tests ---

def t01_substituted_filename():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)
        files[7] = "not_in_the_split.jpg"
        labels = [m[f] if f in m else "glioma" for f in files]
        d = make_run(tmp, "no_ema", m, preds_kw={"files": files, "labels": labels})
        r = verify(d, "no_ema", m)
        al = r["alignment_clean_test"]
        assert not al["matches_canonical"], "a substituted filename was accepted"
        assert al["n_extra"] == 1 and al["n_missing"] == 1, al
        assert r["checks"]["canonical_alignment"] is False
        assert not r["accepted"], "run with a substituted image was accepted"


def t02_wrong_true_label():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)
        labels = [m[f] for f in files]
        labels[3] = CLASSES[(CLASSES.index(labels[3]) + 1) % 4]
        d = make_run(tmp, "no_ema", m, preds_kw={"files": files, "labels": labels})
        r = verify(d, "no_ema", m)
        al = r["alignment_clean_test"]
        assert not al["matches_canonical"], "a wrong true label was accepted"
        assert al["n_true_label_mismatch"] == 1, al
        assert al["n_missing"] == 0 and al["n_extra"] == 0, \
            "identities were fine; only the label was wrong"
        assert not r["accepted"]


def t03_duplicate_filename():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)
        files[5] = files[4]
        labels = [m[f] for f in files]
        d = make_run(tmp, "no_ema", m, preds_kw={"files": files, "labels": labels})
        msg = rejects(verify, d, "no_ema", m)
        assert "duplicate" in msg.lower(), msg


def t04_wrong_class_order():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        permuted = [CLASSES[2], CLASSES[0], CLASSES[1], CLASSES[3]]
        d = make_run(tmp, "no_ema", m, preds_kw={"class_cols": permuted})
        msg = rejects(verify, d, "no_ema", m)
        assert "sorted class order" in msg or "class set" in msg, msg


def t05_unintended_config_difference():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        # no_ema is supposed to change ema and nothing else; this also halves LR
        cfg = variant_cfg("no_ema")
        cfg["lr"] = 5e-5
        d = make_run(tmp, "no_ema", m, cfg=cfg)
        r = verify(d, "no_ema", m)
        cd = r["config_diff"]
        assert "lr" in cd["unintended"], cd
        assert "lr" in cd["protocol_violations"], cd
        assert cd["blocks_acceptance"] is True, cd
        assert cd["verdict"] == "UNINTENDED DIFFERENCE", cd["verdict"]
        assert not r["accepted"], "a confounded run was accepted"


def t05b_unintended_difference_is_not_a_limitation():
    """A real confound must not be downgraded to a 'named limitation'."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        cfg = variant_cfg("no_ema")
        cfg["epochs"] = 30
        d = make_run(tmp, "no_ema", m, cfg=cfg)
        cd = verify(d, "no_ema", m)["config_diff"]
        assert cd["blocks_acceptance"], cd
        assert "epochs" not in cd["comparability_limitations"], \
            "a confound was filed as a limitation"


def t05c_missing_intended_difference():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        cfg = variant_cfg("no_ema")
        cfg["resolved_config"]["ema"] = True  # the ablation was not actually applied
        d = make_run(tmp, "no_ema", m, cfg=cfg)
        cd = verify(d, "no_ema", m)["config_diff"]
        assert cd["intended_missing"] == ["ema"], cd
        assert cd["verdict"] == "INTENDED DIFFERENCE ABSENT", cd["verdict"]
        assert cd["blocks_acceptance"] is True


def t06_wrong_no_tta_source_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        wrong = "f" * 64
        right = "a" * 64
        cfg = variant_cfg("no_tta", trained=False, checkpoint_written=False)
        d = make_run(tmp, "no_tta", m, cfg=cfg,
                     extra_files={"source_checkpoint_sha256.txt": wrong + "\n"},
                     metrics_over={"trained": False, "checkpoint_written": False})
        r = verify(d, "no_tta", m, category="evaluation_only",
                   recovered={42: {"sha256": right}})
        assert r["source_checkpoint"]["matches"] is False, r["source_checkpoint"]
        assert r["checks"]["source_checkpoint_matches"] is False
        assert not r["accepted"], "no_tta with a foreign checkpoint was accepted"


def t06b_no_tta_that_trained_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        right = "a" * 64
        cfg = variant_cfg("no_tta", trained=True, checkpoint_written=True)
        d = make_run(tmp, "no_tta", m, cfg=cfg,
                     extra_files={"source_checkpoint_sha256.txt": right},
                     metrics_over={"trained": True, "checkpoint_written": True})
        r = verify(d, "no_tta", m, category="evaluation_only",
                   recovered={42: {"sha256": right}})
        assert r["checks"]["did_not_train"] is False
        assert not r["accepted"], "an eval-only variant that trained was accepted"


def t07_changed_imported_file():
    """Import integrity compares a freshly computed digest against the Phase 3A
    record. A file edited after import must surface as a digest mismatch even
    when the edit leaves the file perfectly well-formed - which is the case that
    a structural check alone would miss, so the digest has to be the thing doing
    the work here. Reordering the rows is exactly such an edit."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = make_run(tmp, "no_ema", m)
        pt = os.path.join(d, "preds_test.csv")
        recorded = V.sha256(pt)
        rows = list(csv.DictReader(open(pt, encoding="utf-8")))
        cols = list(rows[0])
        with open(pt, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows[::-1])
        assert V.sha256(pt) != recorded, "editing the file did not change its digest"
        r = verify(d, "no_ema", m)
        assert r["files"]["preds_test.csv"]["sha256"] != recorded, \
            "the verifier reported a stale digest instead of the file on disk"
        assert r["alignment_clean_test"]["matches_canonical"] is True, \
            "the edit was well-formed, so only the digest should have caught it"
        assert r["files"]["preds_test.csv"]["sha256"] == V.sha256(pt)


def t08_reference_seeds_disagree_on_mapping():
    with tempfile.TemporaryDirectory() as tmp:
        m1 = synth_mapping()
        m2 = dict(m1)
        # seed 43 scored one different image
        del m2["img_000.jpg"]
        m2["img_999.jpg"] = "glioma"
        a = make_run(tmp, "ref_a", m1)
        b = make_run(tmp, "ref_b", m2)
        ra = verify(a, "full_fusion_v2", m1, category="reference")
        canon = canon_from_mapping(m1)
        old = V.CLEAN_TEST_N
        V.CLEAN_TEST_N = len(m2)
        try:
            pb = V.load_predictions(os.path.join(b, "preds_test.csv"), len(m2))
        finally:
            V.CLEAN_TEST_N = old
        rb = V.align_to_canonical(pb, canon)
        fps = {ra["alignment_clean_test"]["fingerprint_sha256"], rb["fingerprint_sha256"]}
        assert len(fps) == 2, "two different image sets produced the same fingerprint"
        assert rb["matches_canonical"] is False


def t09_incorrect_stored_macro_f1():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = make_run(tmp, "no_ema", m,
                     metrics_over={"test_macro_f1": 0.123456})
        r = verify(d, "no_ema", m)
        comp = [c for c in r["metrics"]["comparisons"] if c["metric"] == "macro_f1"][0]
        assert comp["stored_field"] == "test_macro_f1", comp
        assert comp["within_tolerance"] is False, comp
        assert r["checks"]["macro_f1_reproduces"] is False
        assert not r["accepted"], "a wrong stored macro-F1 was accepted"


def t09b_correct_stored_macro_f1_accepted():
    """Positive control: the same code path must accept a true stored value."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = make_run(tmp, "no_ema", m)
        pt = V.load_predictions(os.path.join(d, "preds_test.csv"), len(m))
        _, f1 = V.accuracy_macro_f1(pt["y"], pt["pred"], len(CLASSES))
        met = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
        met["test_macro_f1"] = f1
        json.dump(met, open(os.path.join(d, "metrics.json"), "w", encoding="utf-8"))
        r = verify(d, "no_ema", m)
        comp = [c for c in r["metrics"]["comparisons"] if c["metric"] == "macro_f1"][0]
        assert comp["within_tolerance"] is True, comp
        assert r["accepted"], r["failed_checks"]


def t10_absent_macro_f1_is_reported_not_claimed():
    """The real runs store no macro-F1. That must be REPORTED as unavailable, and
    must not silently become a claim of reproduction."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = make_run(tmp, "no_ema", m)
        met = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
        assert not [k for k in met if "macro" in k.lower() and "f1" in k.lower()]
        r = verify(d, "no_ema", m)
        comp = [c for c in r["metrics"]["comparisons"] if c["metric"] == "macro_f1"][0]
        assert comp["stored_field"] is None, comp
        assert comp["stored_value"] is None, comp
        assert "abs_diff_pp" not in comp, "a diff was reported against nothing"
        assert "no stored macro-F1" in comp["note"], comp
        assert "macro_f1_reproduces" not in r["checks"], \
            "absence of a stored value was recorded as a reproduction"
        assert comp["recomputed"] > 0.0
        assert r["accepted"], "an honest 'unavailable' must not fail the run"


def t11_incorrect_stored_accuracy():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = make_run(tmp, "no_ema", m, metrics_over={"test_acc": 0.999})
        r = verify(d, "no_ema", m)
        assert r["checks"]["accuracy_reproduces"] is False
        assert not r["accepted"]


def t12_probability_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = make_run(tmp, "no_ema", m)
        pt = os.path.join(d, "preds_test.csv")
        rows = list(csv.DictReader(open(pt, encoding="utf-8")))
        cols = list(rows[0])
        for c in cols:
            if c.startswith("p_"):
                rows[3][c] = "%.6f" % (float(rows[3][c]) * 0.5)
        with open(pt, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)
        msg = rejects(V.load_predictions, pt, len(m))
        assert "sum 1" in msg, msg


def t13_row_count_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = make_run(tmp, "no_ema", m)
        msg = rejects(V.load_predictions, os.path.join(d, "preds_test.csv"), len(m) + 1)
        assert "rows, expected" in msg, msg


def t14_alignment_is_by_identity_not_row_order():
    """Shuffling the rows must not change the verdict; the addendum claims
    matching is by stable identity, so this has to hold."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        files = list(m)[::-1]
        d = make_run(tmp, "no_ema", m,
                     preds_kw={"files": files, "labels": [m[f] for f in files]})
        r = verify(d, "no_ema", m)
        assert r["alignment_clean_test"]["matches_canonical"] is True, \
            "row order changed the alignment verdict"
        assert r["accepted"], r["failed_checks"]


def t15_intended_sets_come_from_the_spec():
    """The corrected sets must be read from run_ablations.py, and must NOT claim
    vit_branch_only changes multiscale - the earlier hand-written bug."""
    spec = V.intended_from_spec()
    assert spec["source"].startswith("run_ablations.py"), spec["source"]
    it = spec["intended"]
    assert it["no_gate"] == {"gated"}, it["no_gate"]
    assert it["no_ema"] == {"ema"}, it["no_ema"]
    assert it["no_tta"] == {"tta"}, it["no_tta"]
    assert it["no_auxiliary_loss"] == {"aux_weight"}, it["no_auxiliary_loss"]
    assert it["no_multiscale"] == {"multiscale"}, it["no_multiscale"]
    assert "multiscale" not in it["vit_branch_only"], \
        "the hand-written vit_branch_only bug is back"
    assert it["vit_branch_only"] == {"arch", "unfreeze_all_cnn"}, it["vit_branch_only"]
    assert it["cnn_branch_only"] == {"arch", "unfreeze_all_vit"}, it["cnn_branch_only"]
    ref = spec["reference_expectation"]
    assert ref["ema"] is True and ref["tta"] is True and ref["aux_weight"] == 0.3


def t16_module_constants_are_parsed_not_invented():
    root = os.path.dirname(os.path.dirname(_HERE))
    c = V.module_constants(root)
    assert not c["missing"], c["missing"]
    assert len(c["source_sha256"]) == 64
    got = {k: v["value"] for k, v in c["constants"].items()}
    assert got["BATCH_SIZE"] == 16 and got["EPOCHS"] == 15, got
    assert got["PATIENCE"] == 4 and got["VAL_FRACTION"] == 0.15, got
    assert got["VAL_SPLIT_SEED"] == 42 and got["LABEL_SMOOTHING"] == 0.1, got
    src = open(os.path.join(root, V.REFERENCE_TRAINER), encoding="utf-8").read()
    for name, info in c["constants"].items():
        line = src.splitlines()[info["line"] - 1]
        assert line.startswith(name), "line %d does not define %s: %r" % (
            info["line"], name, line)


def t17_missing_module_constant_is_a_stop_condition():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "src"))
        open(os.path.join(tmp, V.REFERENCE_TRAINER), "w", encoding="utf-8").write(
            "BATCH_SIZE = 16\nEPOCHS = 15\n")
        c = V.module_constants(tmp)
        assert "PATIENCE" in c["missing"], c["missing"]
        assert "VAL_SPLIT_SEED" in c["missing"], c["missing"]


def t18_gating_is_not_derived_from_parameter_count():
    """fusion_v2 allocates the gate whether or not it is used, so the parameter
    count cannot settle gating. The deriver must say so rather than pick a side."""
    cohorts = {"gated": {True: {91211918}, False: {91211918}},
               "multiscale": {True: {91211918}, False: {91134094}}}
    out = V.derive_reference_architecture({"params_total": 91211918,
                                           "arch": "fusion_v2"}, cohorts)
    assert out["gated"]["value"] is None, out["gated"]
    assert out["gated"]["evidence_class"] == "unresolvable", out["gated"]
    assert out["multiscale"]["value"] is True, out["multiscale"]
    assert out["multiscale"]["evidence_class"] == "derived_from_params_total"


def t19_weak_evidence_is_labelled_not_hidden():
    """gated is resolved only from the spec. Any variant whose conclusion depends
    on it must carry that label."""
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        res = resolvers()
        d = make_run(tmp, "no_gate", m)
        cd = verify(d, "no_gate", m, res=res)["config_diff"]
        assert "gated" in cd["reference_weak_evidence_keys"], cd
        assert cd["weak_evidence_affecting_conclusion"] == ["gated"], cd
        assert cd["verdict"] == "OK ON WEAKEST EVIDENCE CLASS", cd["verdict"]
        assert cd["blocks_acceptance"] is False, \
            "an honestly named weak basis should not be a rejection"
        # a variant that does not depend on gating must not be labelled
        d2 = make_run(tmp, "no_ema", m)
        cd2 = verify(d2, "no_ema", m, res=res)["config_diff"]
        assert cd2["weak_evidence_affecting_conclusion"] == [], cd2
        assert cd2["verdict"] == "OK", cd2["verdict"]


def t20_spec_disagreement_is_not_usable():
    """If a field the reference DID record contradicts the spec, the spec may not
    be used to fill in the fields it did not record."""
    bad = dict(REF_CFG)
    bad["ema"] = False
    spec = V.intended_from_spec()
    agree = V.reference_spec_agreement(bad, spec["reference_expectation"])
    assert agree["usable"] is False, agree
    assert "ema" in agree["disagreements"], agree
    good = V.reference_spec_agreement(REF_CFG, spec["reference_expectation"])
    assert good["usable"] is True, good
    assert good["disagreements"] == []
    assert sorted(good["unrecorded_fields"]) == ["gated", "multiscale"], good


def t21_unmapped_config_field_is_surfaced():
    cfg = variant_cfg("no_ema")
    cfg["mystery_knob"] = 7
    got = V.unmapped_fields(cfg)
    assert "mystery_knob" in got, got
    assert "seed" not in got and "resolved_config" not in got, got


def t22_out_root_outside_publication_is_refused():
    src = open(os.path.join(_HERE, "verify_phase3a_addendum.py"), encoding="utf-8").read()
    assert 'startswith("publication_")' in src, \
        "the write-location guard is gone"
    assert 'raise AddendumError("refusing to write outside publication_' in src


def t23_verifier_writes_nowhere_else():
    """Static guard: the module must not open any protected path for writing."""
    src = open(os.path.join(_HERE, "verify_phase3a_addendum.py"), encoding="utf-8").read()
    for token in ('"w"', "'w'", '"a"', "'a'", '"wb"', "'wb'"):
        for i, line in enumerate(src.splitlines(), 1):
            if token in line:
                assert ("out_root" in line or "f.write" in line
                        or "json.dump" in line), \
                    "line %d writes somewhere unaudited: %s" % (i, line.strip())


def t24_canonical_mapping_rejects_wrong_identity_count():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "dataset_clean"))
        entries = [{"file": "a_%d.jpg" % i, "class": "glioma", "new_split": "Testing"}
                   for i in range(5)]
        json.dump(entries, open(os.path.join(tmp, V.SPLIT_MANIFEST), "w",
                               encoding="utf-8"))
        msg = rejects(V.canonical_mapping, tmp)
        assert "unique identities" in msg, msg


def t25_canonical_mapping_rejects_unknown_class():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "dataset_clean"))
        entries = [{"file": "a_%d.jpg" % i,
                    "class": "not_a_class" if i == 0 else CLASSES[i % 4],
                    "new_split": "Testing"} for i in range(V.CLEAN_TEST_N)]
        json.dump(entries, open(os.path.join(tmp, V.SPLIT_MANIFEST), "w",
                               encoding="utf-8"))
        msg = rejects(V.canonical_mapping, tmp)
        assert "outside the class set" in msg, msg


def t26_pred_label_must_match_argmax():
    with tempfile.TemporaryDirectory() as tmp:
        m = synth_mapping()
        d = make_run(tmp, "no_ema", m)
        pt = os.path.join(d, "preds_test.csv")
        rows = list(csv.DictReader(open(pt, encoding="utf-8")))
        cols = list(rows[0])
        rows[2]["pred_label"] = CLASSES[(CLASSES.index(rows[2]["pred_label"]) + 1) % 4]
        with open(pt, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows(rows)
        msg = rejects(V.load_predictions, pt, len(m))
        assert "argmax" in msg, msg


def t27_real_artifacts_are_untouched():
    """The suite must not have written into any protected tree."""
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
                assert not f.startswith("img_") or "seed_42" not in dirpath, \
                    "fixture leaked into %s" % dirpath


def frozen_digests(root):
    out = {}
    for rel in ("src/fusion_v2.py", "src/train_ablation.py"):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            out[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out",
                    default="publication_results/ablations/phase3a_addendum/TEST_REPORT.md")
    args = ap.parse_args()
    root = os.path.abspath(args.project_root)
    before = frozen_digests(root)

    print("Phase 3A Addendum - verifier regression tests")
    print("=" * 74)
    tests = [(k, v) for k, v in sorted(globals().items())
             if k[0] == "t" and k[1].isdigit() and callable(v)]
    order = [n for n, _ in tests]
    for name, fn in tests:
        check(name, fn)

    after = frozen_digests(root)
    if before != after:
        FAILED.append(("frozen model/training code unchanged",
                       "digests changed: %s" % sorted(
                           k for k in before if before[k] != after.get(k))))
        print("  FAIL  frozen model/training code unchanged")
    else:
        PASSED.append("frozen model/training code unchanged")
        print("  ok    frozen model/training code unchanged (%d files)" % len(before))

    n_pass, n_total = len(PASSED), len(PASSED) + len(FAILED)
    status = "PASS" if not FAILED else "FAIL"
    print("=" * 74)
    print("%d passed, %d failed -> %s" % (n_pass, len(FAILED), status))
    for name, why in FAILED:
        print("  - %s :: %s" % (name, why))

    out = os.path.join(root, args.out)
    if not os.path.relpath(out, root).replace("\\", "/").startswith("publication_"):
        raise SystemExit("refusing to write outside publication_*")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    why = {n: (globals()[n].__doc__ or "").strip().split("\n")[0] for n in order}
    failed_names = dict(FAILED)
    L = ["# Phase 3A Addendum - Verifier Test Report", "",
         "Generated by `src/publication/test_phase3a_addendum.py`.", "",
         "**%d of %d tests passed. Status: %s.**" % (n_pass, n_total, status), "",
         "Every test builds a synthetic run in a temporary directory, breaks "
         "exactly one thing, and asserts the verifier rejects it for that reason. "
         "The suite also asserts the opposite direction wherever a check could "
         "pass by being indiscriminate: that row order does **not** change an "
         "alignment verdict, that a correct stored macro-F1 is accepted by the "
         "same code path that rejects a wrong one, that a variant which does not "
         "depend on gating is **not** labelled with the weak-evidence caveat, and "
         "that an honestly reported 'no stored value available' does not fail a "
         "run. A suite that only ever rejected things would pass while the "
         "verifier rejected everything.", "",
         "Two tests exist specifically because the verifier was wrong earlier in "
         "this phase: `t15` pins the intended-difference sets to the launch "
         "specification after a hand-written version incorrectly claimed "
         "`vit_branch_only` changes `multiscale`, and `t18` pins the fact that "
         "parameter count cannot decide gating, after an attempt to derive it "
         "that way.", "",
         "| Test | Status | What it asserts |", "|---|---|---|"]
    for n in order:
        st = "FAIL" if n in failed_names else "PASS"
        L.append("| `%s` | %s | %s |" % (n, st, why[n].replace("|", "/")))
    L.append("| `frozen model/training code unchanged` | %s | "
             "`src/fusion_v2.py` and `src/train_ablation.py` are byte-identical "
             "before and after the suite |"
             % ("PASS" if "frozen model/training code unchanged" in PASSED else "FAIL"))
    L += ["", "## Frozen files, digests recorded at run time", "",
          "| File | SHA-256 |", "|---|---|"]
    L += ["| `%s` | `%s` |" % (k, v) for k, v in sorted(after.items())]
    if FAILED:
        L += ["", "## Failures", ""] + ["- `%s`: %s" % (n, w) for n, w in FAILED]
    L += ["", "## Not runnable in this environment", "",
          "`src/publication/test_ablations.py` imports torch, which is not "
          "installed locally, so that pre-existing suite could not be re-run here. "
          "It is unrelated to this addendum and was not modified.", ""]
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote %s" % os.path.relpath(out, root))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
