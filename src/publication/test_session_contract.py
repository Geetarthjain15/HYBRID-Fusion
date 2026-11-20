"""
Phase 3A regression tests for the artifact contract and run validator.

Each test builds a synthetic run directory, breaks exactly one thing, and
asserts the validator rejects it for that reason - and, where the design says a
file should be absent, that a correct run is NOT rejected for lacking it.

That second direction is the one that matters here: the "3/12 complete" failure
was a validator demanding gate CSVs from models that have no gate. A suite that
only tested "missing file -> reject" would have passed while the contract was
wrong.

    python src/publication/test_session_contract.py --project-root .
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from session_artifact_contract import (required_artifacts, forbidden_artifacts,
                                       design, CLASSES, CLEAN_TEST_N, EXTERNAL_N,
                                       PROB_SUM_TOL, METRIC_TOL_PP)
import verify_session_archives as VS

RESULTS = []
RECOVERED = {42: {"seed": 42, "sha256": "a" * 64},
             43: {"seed": 43, "sha256": "b" * 64},
             44: {"seed": 44, "sha256": "c" * 64}}


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append({"test": name, "status": "PASS", "detail": detail or ""})
        print("  PASS  %-56s %s" % (name, detail or ""), flush=True)
    except AssertionError as e:
        RESULTS.append({"test": name, "status": "FAIL", "detail": str(e)})
        print("  FAIL  %-56s %s" % (name, e), flush=True)
    except Exception as e:
        RESULTS.append({"test": name, "status": "ERROR",
                        "detail": "%s: %s" % (type(e).__name__, e),
                        "traceback": traceback.format_exc()})
        print("  ERROR %-56s %s: %s" % (name, type(e).__name__, e), flush=True)


# ------------------------------------------------------------- synthesis ---

def write_preds(path, n, classes=CLASSES, seed=0, bad=None, n_override=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_override if n_override is not None else n):
        p = rng.dirichlet(np.ones(len(classes)))
        if bad == "sum":
            p = p * 0.5          # stays inside [0,1]; sums to 0.5, not 1
        if bad == "range":
            p = p - 0.6
        if bad == "nan":
            p = p.astype(float); p[0] = float("nan")
        k = int(np.argmax(p)) if bad not in ("nan",) else 0
        rows.append({"filename": "img_%05d.jpg" % i,
                     "true_label": classes[i % len(classes)],
                     "pred_label": classes[k],
                     **{"p_" + c: "%.6f" % p[j] for j, c in enumerate(classes)}})
    if bad == "dupe" and len(rows) > 1:
        rows[1]["filename"] = rows[0]["filename"]
    if bad == "label":
        rows[0]["true_label"] = "notaclass"
    if bad == "argmax":
        cur = rows[0]["pred_label"]
        rows[0]["pred_label"] = next(c for c in classes if c != cur)
        for j, c in enumerate(classes):
            rows[0]["p_" + c] = "%.6f" % (0.97 if c == cur else 0.01)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    correct = sum(r["true_label"] == r["pred_label"] for r in rows)
    return correct / len(rows)


def write_gates(path, n):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "true_label", "gate_cnn", "gate_vit"])
        for i in range(n):
            w.writerow(["img_%05d.jpg" % i, CLASSES[i % 4], "0.200000", "0.800000"])


def make_run(d, variant, seed=42, *, omit=(), add=(), bad_pred=None,
             pred_n=None, stored_acc=None, cfg_over=None, met_over=None,
             classes=CLASSES):
    os.makedirs(d, exist_ok=True)
    des = design(variant)
    acc = write_preds(os.path.join(d, "preds_test.csv"), CLEAN_TEST_N,
                      classes=classes, seed=seed, bad=bad_pred, n_override=pred_n)
    write_preds(os.path.join(d, "preds_dataset_ood.csv"), EXTERNAL_N,
                classes=classes, seed=seed + 1)
    cfg = {"variant": variant, "seed": seed, "split": "clean",
           "resolved_config": {"ema": variant != "no_ema", "tta": variant != "no_tta",
                               "gated": des["has_gate"] or variant == "no_tta",
                               "multiscale": variant != "no_multiscale",
                               "aux_weight": 0.0 if variant == "no_auxiliary_loss" else 0.3}}
    cfg.update(cfg_over or {})
    met = {"variant": variant, "seed": seed, "split": "clean",
           "test_acc": stored_acc if stored_acc is not None else acc,
           "overfitting": {"checkpoint_is_best_epoch": True}}
    if variant == "no_tta":
        met.update({"tta_used": False, "trained": False, "weights_updated": False,
                    "checkpoint_written": False,
                    "source_checkpoint_sha256": RECOVERED[seed]["sha256"]})
    met.update(met_over or {})
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    json.dump(met, open(os.path.join(d, "metrics.json"), "w"))
    json.dump({"python": "3.12"}, open(os.path.join(d, "environment.json"), "w"))
    open(os.path.join(d, "run.log"), "w").write("log\n")
    if des["mode"] == "train":
        json.dump([{"epoch": 1, "train_acc": 0.9, "val_acc": 0.95}],
                  open(os.path.join(d, "history.json"), "w"))
        open(os.path.join(d, "checkpoint.pth"), "wb").write(b"\x00" * 16)
    else:
        open(os.path.join(d, "source_checkpoint_sha256.txt"), "w").write(
            "%s  /path/checkpoint.pth\n" % RECOVERED[seed]["sha256"])
    if des["has_gate"]:
        write_gates(os.path.join(d, "gates_test.csv"), CLEAN_TEST_N)
        write_gates(os.path.join(d, "gates_dataset_ood.csv"), EXTERNAL_N)
    for name in omit:
        p = os.path.join(d, name)
        if os.path.exists(p):
            os.remove(p)
    for name in add:
        open(os.path.join(d, name), "wb").write(b"\x00" * 8)
    return d


def run(variant, seed=42, **kw):
    tmp = tempfile.mkdtemp()
    try:
        d = make_run(os.path.join(tmp, "r"), variant, seed, **kw)
        return VS.validate_run(d, variant, seed, RECOVERED)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------- tests ---

def t_contract_table():
    assert required_artifacts("no_ema").count("gates_test.csv") == 1
    assert "gates_test.csv" not in required_artifacts("cnn_branch_only")
    assert "gates_test.csv" not in required_artifacts("vit_branch_only")
    assert "checkpoint.pth" not in required_artifacts("no_tta")
    assert "history.json" not in required_artifacts("no_tta")
    assert "source_checkpoint_sha256.txt" in required_artifacts("no_tta")
    assert "checkpoint.pth" in forbidden_artifacts("no_tta")
    assert "history.json" in forbidden_artifacts("no_tta")
    assert "gates_test.csv" in forbidden_artifacts("cnn_branch_only")
    return "requirements follow architecture and mode"


def t_gated_missing_gates_fails():
    r = run("no_ema", omit=["gates_test.csv"])
    assert not r["accepted"] and "gates_test.csv" in r["reason"], r["reason"]
    return "no_ema without gate CSVs -> REJECT"


def t_branchonly_missing_gates_passes():
    for v in ("cnn_branch_only", "vit_branch_only"):
        r = run(v)
        assert r["accepted"], "%s rejected: %s" % (v, r["reason"])
        assert "gates_test.csv" not in r["required"]
    return "cnn/vit_branch_only accepted with no gate CSVs (the 3/12 defect)"


def t_branchonly_with_gates_rejected():
    r = run("cnn_branch_only", add=["gates_test.csv"])
    assert not r["accepted"] and "must not exist" in r["reason"], r["reason"]
    return "gate file present on a gateless model -> REJECT"


def t_no_tta_no_checkpoint_or_history():
    r = run("no_tta")
    assert r["accepted"], r["reason"]
    r2 = run("no_tta", add=["checkpoint.pth"])
    assert not r2["accepted"] and "checkpoint.pth" in r2["reason"], r2["reason"]
    r3 = run("no_tta", add=["history.json"])
    assert not r3["accepted"] and "history.json" in r3["reason"], r3["reason"]
    return "no_tta needs neither; a produced checkpoint or history -> REJECT"


def t_no_tta_source_checkpoint_must_match():
    r = run("no_tta", met_over={"source_checkpoint_sha256": "f" * 64})
    assert not r["accepted"] and "source_checkpoint_matches_verified" in r["reason"], r["reason"]
    return "wrong source checkpoint digest -> REJECT"


def t_no_tta_must_not_have_trained():
    r = run("no_tta", met_over={"trained": True})
    assert not r["accepted"] and "did_not_train" in r["reason"], r["reason"]
    r2 = run("no_tta", met_over={"tta_used": True})
    assert not r2["accepted"] and "tta_disabled" in r2["reason"], r2["reason"]
    return "trained=True or tta_used=True -> REJECT"


def t_no_ema_differs_only_in_ema():
    r = run("no_ema")
    assert r["accepted"], r["reason"]
    assert r["checks"]["ema_disabled"] and r["checks"]["tta_unchanged"]
    r2 = run("no_ema", cfg_over={"resolved_config": {
        "ema": False, "tta": False, "gated": True, "multiscale": True, "aux_weight": 0.3}})
    assert not r2["accepted"] and "tta_unchanged" in r2["reason"], r2["reason"]
    r3 = run("no_ema", cfg_over={"resolved_config": {
        "ema": True, "tta": True, "gated": True, "multiscale": True, "aux_weight": 0.3}})
    assert not r3["accepted"] and "ema_disabled" in r3["reason"], r3["reason"]
    return "no_ema must change EMA and only EMA"


def t_wrong_seed_fails():
    tmp = tempfile.mkdtemp()
    try:
        d = make_run(os.path.join(tmp, "r"), "no_ema", 42)
        r = VS.validate_run(d, "no_ema", 43, RECOVERED)   # path says 43, config says 42
        assert not r["accepted"] and "config disagrees with path" in r["reason"], r["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "seed in config not matching the path -> REJECT"


def t_wrong_class_order_fails():
    r = run("cnn_branch_only", classes=["pituitary", "notumor", "meningioma", "glioma"])
    assert not r["accepted"] and "sorted class order" in r["reason"], r["reason"]
    return "unsorted class columns -> REJECT"


def t_wrong_sample_count_fails():
    r = run("cnn_branch_only", pred_n=CLEAN_TEST_N - 1)
    assert not r["accepted"] and "contract requires 848" in r["reason"], r["reason"]
    return "847 clean-test rows -> REJECT"


def t_malformed_probabilities_fail():
    for bad, frag in (("sum", "deviate from 1"), ("range", "outside [0,1]"),
                      ("nan", "non-finite")):
        r = run("cnn_branch_only", bad_pred=bad)
        assert not r["accepted"] and frag in r["reason"], "%s: %s" % (bad, r["reason"])
    return "sum, range and NaN violations each -> REJECT"


def t_filename_misalignment_fails():
    r = run("cnn_branch_only", bad_pred="dupe")
    assert not r["accepted"] and "duplicate filenames" in r["reason"], r["reason"]
    r2 = run("cnn_branch_only", bad_pred="label")
    assert not r2["accepted"] and "outside the class set" in r2["reason"], r2["reason"]
    return "duplicate filenames and unknown labels -> REJECT"


def t_argmax_mismatch_fails():
    r = run("cnn_branch_only", bad_pred="argmax")
    assert not r["accepted"] and "argmax" in r["reason"], r["reason"]
    return "pred_label not equal to argmax -> REJECT"


def t_metric_disagreement_fails():
    r = run("cnn_branch_only", stored_acc=0.5)
    assert not r["accepted"] and "differs from stored" in r["reason"], r["reason"]
    return "stored accuracy disagreeing with recomputation -> REJECT"


def t_metrics_within_tolerance_pass():
    tmp = tempfile.mkdtemp()
    try:
        d = make_run(os.path.join(tmp, "r"), "cnn_branch_only", 42)
        met = json.load(open(os.path.join(d, "metrics.json")))
        met["test_acc"] = met["test_acc"] + 0.00005      # 0.005 pp, inside 0.01
        json.dump(met, open(os.path.join(d, "metrics.json"), "w"))
        r = VS.validate_run(d, "cnn_branch_only", 42, RECOVERED)
        assert r["accepted"], r["reason"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "a 0.005 pp difference is inside the 0.01 pp tolerance"


def t_unsafe_archive_members_rejected():
    tmp = tempfile.mkdtemp()
    try:
        import zipfile
        for name, frag in (("/etc/passwd", "absolute"),
                           ("../escape.txt", "traversal")):
            p = os.path.join(tmp, "bad.zip")
            with zipfile.ZipFile(p, "w") as z:
                z.writestr(name, "x")
            info = VS.inspect_members(p)
            assert not info["safe"] and any(frag in s for s in info["problems"]), \
                "%s not caught: %s" % (name, info["problems"])
        p = os.path.join(tmp, "ok.zip")
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("clean/no_ema/seed_42/metrics.json", "{}")
        assert VS.inspect_members(p)["safe"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "absolute paths and traversal rejected; a normal member accepted"


def t_sidecar_parsed_as_data():
    tmp = tempfile.mkdtemp()
    try:
        z = "a.zip"
        open(os.path.join(tmp, z), "wb").write(b"x")
        open(os.path.join(tmp, z + ".sha256"), "w").write("%s  %s\n" % ("d" * 64, z))
        s = VS.read_sidecar(tmp, z)
        assert s["recorded_sha256"] == "d" * 64
        open(os.path.join(tmp, z + ".sha256"), "w").write("not a digest\n")
        try:
            VS.read_sidecar(tmp, z)
            raise AssertionError("malformed sidecar accepted")
        except VS.VerifyError:
            pass
        open(os.path.join(tmp, z + ".sha256"), "w").write("%s  other.zip\n" % ("d" * 64))
        try:
            VS.read_sidecar(tmp, z)
            raise AssertionError("sidecar naming the wrong file accepted")
        except VS.VerifyError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "malformed or mis-named sidecars rejected"


def t_expected_matrix_categories():
    e = VS.expected_cells()
    assert len(e["session_1_training"]) == 9
    assert len(e["session_2_training"]) == 9
    assert len(e["session_2_eval_only"]) == 3
    assert len(e["reference_reused"]) == 3
    assert set(v for v, _ in e["reference_reused"]) == {"full_fusion_v2"}
    return "9 + 9 + 3 newly trained/evaluated, 3 reference kept separate"


def t_protected_unchanged(root, baseline):
    def inner():
        changed = [p for p, h in baseline.items()
                   if not os.path.exists(os.path.join(root, p))
                   or hashlib.sha256(open(os.path.join(root, p), "rb").read()).hexdigest() != h]
        assert not changed, "modified: %s" % changed[:5]
        return "%d protected/frozen files unchanged" % len(baseline)
    return inner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out", default="publication_results/ablations/phase3a/TEST_REPORT.md")
    args = ap.parse_args()
    root = os.path.abspath(args.project_root)

    baseline = {}
    for rel in ("src/fusion_v2.py", "src/train_ablation.py"):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            baseline[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()

    print("Phase 3A contract tests\n" + "=" * 72, flush=True)
    check("contract table follows architecture and mode", t_contract_table)
    check("gated variant missing gate CSVs -> reject", t_gated_missing_gates_fails)
    check("branch-only missing gate CSVs -> ACCEPT", t_branchonly_missing_gates_passes)
    check("gate file on a gateless model -> reject", t_branchonly_with_gates_rejected)
    check("no_tta needs no checkpoint or history", t_no_tta_no_checkpoint_or_history)
    check("no_tta source checkpoint must match", t_no_tta_source_checkpoint_must_match)
    check("no_tta must not have trained", t_no_tta_must_not_have_trained)
    check("no_ema differs only in EMA", t_no_ema_differs_only_in_ema)
    check("wrong seed -> reject", t_wrong_seed_fails)
    check("wrong class order -> reject", t_wrong_class_order_fails)
    check("wrong sample count -> reject", t_wrong_sample_count_fails)
    check("malformed probabilities -> reject", t_malformed_probabilities_fail)
    check("filename misalignment -> reject", t_filename_misalignment_fails)
    check("argmax mismatch -> reject", t_argmax_mismatch_fails)
    check("metric disagreement -> reject", t_metric_disagreement_fails)
    check("metric within tolerance -> accept", t_metrics_within_tolerance_pass)
    check("unsafe archive members rejected", t_unsafe_archive_members_rejected)
    check("checksum sidecar parsed as data", t_sidecar_parsed_as_data)
    check("expected matrix categories", t_expected_matrix_categories)
    check("protected artifacts unchanged", t_protected_unchanged(root, baseline))

    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    status = "PASS" if n_pass == len(RESULTS) else "FAIL"
    print("=" * 72)
    print("%d/%d passed -> %s" % (n_pass, len(RESULTS), status), flush=True)

    out = os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L = ["# Phase 3A - Contract Test Report", "",
         "Generated by `src/publication/test_session_contract.py`.", "",
         "**%d of %d tests passed. Status: %s.**" % (n_pass, len(RESULTS), status), "",
         "Each test builds a synthetic run, breaks exactly one thing, and asserts "
         "the validator rejects it for that reason. Crucially it also asserts the "
         "opposite direction: that a gateless model is NOT rejected for lacking "
         "gate files. A suite testing only 'missing file -> reject' would have "
         "passed while the contract was wrong - which is how the 3/12 result "
         "arose.", "", "| Test | Status | Detail |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % (r["test"], r["status"], r["detail"]) for r in RESULTS]
    L.append("")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote %s" % os.path.relpath(out, root))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
