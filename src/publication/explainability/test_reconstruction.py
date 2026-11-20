"""
Phase X1 tests.

Each test breaks exactly one thing and asserts the module refuses it, plus
positive controls so a suite that rejected everything could not pass. Nothing
here touches dataset/, dataset_clean/, runs*, results* or the checkpoints; the
fixtures are synthetic and live in a temporary directory.

    env_win/Scripts/python.exe src/publication/explainability/test_reconstruction.py --project-root .
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import reconstruct_clean_test as X1

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
    except X1.X1Error as e:
        return str(e)
    raise AssertionError("accepted an input it should have rejected")


def make_manifest(tmp, entries):
    d = Path(tmp) / "dataset_clean"
    d.mkdir(parents=True, exist_ok=True)
    (d / "split_manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return Path(tmp)


def full_entries():
    """848 synthetic Testing entries with the real class counts."""
    out, n = [], 0
    for cls, cnt in X1.EXPECTED_COUNTS.items():
        for i in range(cnt):
            out.append({"file": "%s_%05d.jpg" % (cls, i), "class": cls,
                        "new_split": "Testing", "original_split": "Training",
                        "cluster_size": 1, "cluster_members": []})
            n += 1
    assert n == 848
    return out


# ---------------------------------------------------------------- tests -----

def t01_manifest_wrong_count_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_manifest(tmp, full_entries()[:-1])
        msg = rejects(X1.read_manifest, root)
        assert "847 Testing entries" in msg or "expected 848" in msg, msg


def t02_manifest_duplicate_filename_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        e = full_entries()
        e[5]["file"] = e[4]["file"]
        root = make_manifest(tmp, e)
        msg = rejects(X1.read_manifest, root)
        assert "not unique" in msg, msg


def t03_manifest_unknown_class_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        e = full_entries()
        e[0]["class"] = "not_a_class"
        root = make_manifest(tmp, e)
        msg = rejects(X1.read_manifest, root)
        assert "outside the documented set" in msg or "class counts" in msg, msg


def t04_manifest_wrong_class_counts_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        e = full_entries()
        e[0]["class"] = "pituitary"      # shifts glioma 243->242, pituitary 261->262
        root = make_manifest(tmp, e)
        msg = rejects(X1.read_manifest, root)
        assert "class counts" in msg, msg


def t05_valid_manifest_accepted():
    """Positive control: a correct manifest must pass."""
    with tempfile.TemporaryDirectory() as tmp:
        root = make_manifest(tmp, full_entries())
        testing, sha = X1.read_manifest(root)
        assert len(testing) == 848
        assert len(sha) == 64


def t06_missing_source_file_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        e = full_entries()[:8]
        index = {x["file"]: str(root / "dataset" / x["file"]) for x in e[:-1]}
        msg = rejects(X1.reconstruct, root, e, index, {})
        assert "not found under dataset/" in msg, msg


def t07_ambiguous_source_filename_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        e = full_entries()[:8]
        index = {x["file"]: "a/%s" % x["file"] for x in e}
        dupes = {e[3]["file"]: ["a/x.jpg", "b/x.jpg"]}
        msg = rejects(X1.reconstruct, root, e, index, dupes)
        assert "more than one source path" in msg, msg


def t08_refuses_to_overwrite_destination():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dest = root / X1.RECON_ROOT
        dest.mkdir(parents=True)
        (dest / "sentinel.txt").write_text("pre-existing", encoding="utf-8")
        e = full_entries()[:4]
        index = {x["file"]: str(root / x["file"]) for x in e}
        msg = rejects(X1.reconstruct, root, e, index, {})
        assert "refusing to overwrite" in msg, msg
        assert (dest / "sentinel.txt").read_text(encoding="utf-8") == "pre-existing"


def t09_copy_uses_manifest_class_not_source_dir():
    """The repartition moved images between splits, so the source directory is
    NOT the label. Copying by source directory would mislabel images."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "dataset" / "Training" / "meningioma"
        src.mkdir(parents=True)
        f = src / "glioma_00001.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"x" * 40)
        entry = [{"file": "glioma_00001.jpg", "class": "glioma",
                  "new_split": "Testing", "original_split": "Training"}]
        rec = X1.reconstruct(root, entry, {"glioma_00001.jpg": str(f)}, {})
        dest = root / X1.RECON_ROOT / "glioma" / "glioma_00001.jpg"
        assert dest.is_file(), "file was not placed under its MANIFEST class"
        assert not (root / X1.RECON_ROOT / "meningioma").exists(), \
            "file was placed under its SOURCE directory instead of its manifest class"
        assert rec["files"][0]["manifest_class"] == "glioma"


def t10_records_hash_and_source_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "dataset" / "Testing" / "glioma"
        src.mkdir(parents=True)
        f = src / "glioma_00002.jpg"
        payload = b"\xff\xd8\xff" + b"y" * 100
        f.write_bytes(payload)
        entry = [{"file": "glioma_00002.jpg", "class": "glioma", "new_split": "Testing"}]
        rec = X1.reconstruct(root, entry, {"glioma_00002.jpg": str(f)}, {})
        r = rec["files"][0]
        assert r["sha256"] == hashlib.sha256(payload).hexdigest()
        assert r["bytes"] == len(payload)
        assert r["source_path"].endswith("glioma_00002.jpg")
        assert "dataset/" in r["source_path"]
        assert r["destination_path"].startswith(X1.RECON_ROOT)


def t11_content_duplicates_are_detected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "dataset" / "Testing" / "glioma"
        src.mkdir(parents=True)
        same = b"\xff\xd8\xff" + b"z" * 60
        for n in ("a.jpg", "b.jpg"):
            (src / n).write_bytes(same)
        entries = [{"file": "a.jpg", "class": "glioma", "new_split": "Testing"},
                   {"file": "b.jpg", "class": "glioma", "new_split": "Testing"}]
        idx = {n: str(src / n) for n in ("a.jpg", "b.jpg")}
        rec = X1.reconstruct(root, entries, idx, {})
        assert rec["n_content_duplicate_groups"] == 1, rec["n_content_duplicate_groups"]


def t12_compare_detects_pred_label_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ref.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_label", "pred_label"] + ["p_" + c for c in X1.CLASSES])
            w.writerow(["a.jpg", "glioma", "glioma", 0.9, 0.04, 0.03, 0.03])
        rows = [{"filename": "a.jpg", "true_label": "glioma", "pred_label": "meningioma",
                 "probs": np.array([0.04, 0.9, 0.03, 0.03])}]
        c = X1.compare(rows, p)
        assert c["n_pred_label_mismatch"] == 1, c
        assert c["pred_label_mismatch_examples"] == ["a.jpg"]


def t13_compare_detects_true_label_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ref.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_label", "pred_label"] + ["p_" + c for c in X1.CLASSES])
            w.writerow(["a.jpg", "glioma", "glioma", 0.9, 0.04, 0.03, 0.03])
        rows = [{"filename": "a.jpg", "true_label": "pituitary", "pred_label": "glioma",
                 "probs": np.array([0.9, 0.04, 0.03, 0.03])}]
        c = X1.compare(rows, p)
        assert c["n_true_label_mismatch"] == 1, c


def t14_compare_detects_filename_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ref.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_label", "pred_label"] + ["p_" + c for c in X1.CLASSES])
            w.writerow(["a.jpg", "glioma", "glioma", 0.9, 0.04, 0.03, 0.03])
        rows = [{"filename": "b.jpg", "true_label": "glioma", "pred_label": "glioma",
                 "probs": np.array([0.9, 0.04, 0.03, 0.03])}]
        c = X1.compare(rows, p)
        assert c["n_filename_mismatch"] == 2, c      # one each way
        assert c["n_compared"] == 0


def t15_compare_exact_match_is_clean():
    """Positive control: identical predictions must report zero of everything."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ref.csv"
        probs = [0.9, 0.04, 0.03, 0.03]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_label", "pred_label"] + ["p_" + c for c in X1.CLASSES])
            w.writerow(["a.jpg", "glioma", "glioma"] + probs)
        rows = [{"filename": "a.jpg", "true_label": "glioma", "pred_label": "glioma",
                 "probs": np.array(probs)}]
        c = X1.compare(rows, p)
        assert c["n_filename_mismatch"] == 0 and c["n_true_label_mismatch"] == 0
        assert c["n_pred_label_mismatch"] == 0
        assert c["max_abs_prob_diff"] == 0.0
        assert c["accuracy_diff_pp"] == 0.0


def t16_compare_reports_probability_difference():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ref.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "true_label", "pred_label"] + ["p_" + c for c in X1.CLASSES])
            w.writerow(["a.jpg", "glioma", "glioma", 0.9, 0.04, 0.03, 0.03])
        rows = [{"filename": "a.jpg", "true_label": "glioma", "pred_label": "glioma",
                 "probs": np.array([0.9 + 3e-6, 0.04, 0.03, 0.03])}]
        c = X1.compare(rows, p)
        assert abs(c["max_abs_prob_diff"] - 3e-6) < 1e-9, c["max_abs_prob_diff"]
        assert c["n_pred_label_mismatch"] == 0, "a tiny probability shift must not flip a label"


def t17_smoke_sample_is_deterministic_and_stratified():
    e = full_entries()
    a = X1.limit_sample(e, 4)
    b = X1.limit_sample(e, 4)
    assert [x["file"] for x in a] == [x["file"] for x in b], "sample is not deterministic"
    from collections import Counter
    assert dict(Counter(x["class"] for x in a)) == {c: 4 for c in X1.CLASSES}


def t18_tolerances_are_prespecified():
    assert X1.PROB_TOL == 1e-4, X1.PROB_TOL
    assert X1.ACC_TOL_PP == 0.0, X1.ACC_TOL_PP
    assert X1.EXPECTED_N == 848
    assert X1.EXPECTED_COUNTS == {"glioma": 243, "meningioma": 228,
                                  "notumor": 116, "pituitary": 261}
    assert X1.CLASSES == sorted(X1.CLASSES), "class order must be alphabetical"


def t19_uses_authoritative_transform_and_tta():
    """The transform and TTA must be IMPORTED from the frozen modules, not
    re-typed here - a re-typed copy could drift silently."""
    src = open(HERE / "reconstruct_clean_test.py", encoding="utf-8").read()
    assert "import train_ablation as TA" in src
    assert "TA.eval_transform" in src
    assert "from fusion_v2 import tta_logits" in src
    assert "tta_logits(model, x)" in src
    assert "transforms.Compose" not in src, "the transform appears re-implemented"


def t20_writes_only_under_permitted_roots():
    src = open(HERE / "reconstruct_clean_test.py", encoding="utf-8").read()
    assert X1.RECON_ROOT.startswith("publication_runs/explainability")
    assert X1.OUT_ROOT.startswith("publication_results/explainability")
    for line in src.splitlines():
        if ('open(' in line and ('"w"' in line or "'w'" in line)) or "shutil.copy2" in line:
            assert ("out /" in line or "dest" in line or "dst" in line
                    or "man_path" in line or "scratch" in line or "d / e[" in line), \
                "unaudited write: %s" % line.strip()


def t21_protected_trees_untouched():
    root = HERE.parent.parent.parent
    for name in ("dataset", "dataset_clean", "runs_v2_clean", "results"):
        p = root / name
        if not p.is_dir():
            continue
        for dp, _d, fs in os.walk(p):
            for f in fs:
                assert not f.startswith("glioma_000") or "dataset" in dp, \
                    "fixture leaked into %s" % dp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    frozen = {}
    for rel in ("src/fusion_v2.py", "src/train_ablation.py"):
        p = root / rel
        if p.is_file():
            frozen[rel] = hashlib.sha256(p.read_bytes()).hexdigest()

    print("Phase X1 - reconstruction tests")
    print("=" * 74)
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("t") and len(k) > 1 and k[1].isdigit() and callable(v)]
    for name, fn in tests:
        check(name, fn)

    after = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in frozen}
    if after == frozen:
        PASSED.append("frozen model/training code unchanged")
        print("  ok    frozen model/training code unchanged (%d files)" % len(frozen))
    else:
        FAILED.append(("frozen code unchanged", "digests changed"))
        print("  FAIL  frozen model/training code unchanged")

    print("=" * 74)
    print("%d passed, %d failed" % (len(PASSED), len(FAILED)))
    for n, w in FAILED:
        print("  - %s :: %s" % (n, w))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
