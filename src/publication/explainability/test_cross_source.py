"""Cheap structural tests for Phase X6."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
import cross_source as X6  # noqa: E402
import faithfulness as FA  # noqa: E402

RESULTS = []


def check(name, value, detail=""):
    RESULTS.append(bool(value)); print("  [%s] %-48s %s" %
                                       ("PASS" if value else "FAIL", name, detail))


def main():
    selector, digest = X6.load_selector(ROOT)
    check("t01_protocol_constant", FA.PROTOCOL_LOCK_SHA256 ==
          "a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5")
    check("t02_protocol_file_hash", FA.verify_lock(ROOT)["protocol_lock_sha256"] == FA.PROTOCOL_LOCK_SHA256)
    check("t03_selector_count", len(selector) == 373, str(len(selector)))
    check("t04_selector_unique", len({r['filename'] for r in selector}) == 373)
    check("t05_selector_disjoint_only", all(r["kind"] == "disjoint" for r in selector))
    check("t06_selector_class_counts", dict(__import__('collections').Counter(
          r['true_label'] for r in selector)) == X6.EXPECTED_COUNTS)
    check("t07_selector_digest", digest == hashlib.sha256(
          (ROOT / 'results/cross_source.json').read_bytes()).hexdigest())
    if (ROOT / "dataset_ood").is_dir():
        idx, dup = X6.index_external(ROOT)
        check("t08_all_selector_images_resolve", all(r['filename'] in idx for r in selector))
        check("t09_no_selector_ambiguity", not any(r['filename'] in dup for r in selector))
    else:
        # Detached compute hosts receive the reconstruction, not the immutable
        # source dataset. load_reconstruction verifies exact selector membership
        # and the SHA-256 of every copied image, so this is not a weaker silent
        # fallback: absence of either evidence source remains a hard failure.
        reconstructed = X6.load_reconstruction(ROOT, selector)
        check("t08_reconstruction_exact_membership", len(reconstructed) == 373)
        check("t09_reconstruction_all_hashes_verified", all(
            X6.sha256(ROOT / r["destination_path"]) == r["sha256"]
            for r in reconstructed))
    for seed in X6.SEEDS:
        rec = X6.load_recorded(ROOT, seed, selector)
        check("t10_seed%d_exact_membership" % seed, len(rec) == 373)
        check("t11_seed%d_label_alignment" % seed,
              all(rec[r['filename']]['true_label'] == r['true_label'] for r in selector))
    sample = X6.smoke_sample(selector)
    check("t12_smoke_8", len(sample) == 8)
    check("t13_smoke_2_per_class", all(sum(r['true_label'] == c for r in sample) == 2
                                        for c in X6.CLASSES))
    p = np.array([0.7, 0.1, 0.1, 0.1])
    check("t14_entropy_finite", np.isfinite(X6.entropy(p)))
    ext = [{"filename": "ext_%d" % i, "true_label": c, "m": float(i + 1)}
           for i, c in enumerate(X6.CLASSES)]
    clean = [{"filename": "clean_%d" % i, "true_label": c, "m": float(i)}
             for i, c in enumerate(X6.CLASSES)]
    effect = X6.independent_stratified_shift(ext, clean, "m", 20, 7)
    check("t15_shift_direction", effect["difference_external_minus_clean"] == 1.0)
    check("t16_shift_sets_independent", effect["sets"] == "independent")
    # The implementation must not contain an alternate protocol or selector.
    source = (HERE / "cross_source.py").read_text(encoding="utf-8")
    check("t17_calls_x5_lock", "FA.verify_lock(root)" in source)
    check("t18_no_threshold_tuning", "external_labels_used_for_selection_or_tuning\": False" in source)
    check("t19_no_eager_attention", "set_attn_implementation" not in source and
          "output_attentions" not in source)
    check("t20_frozen_model_hash", hashlib.sha256((ROOT/'src/fusion_v2.py').read_bytes()).hexdigest() ==
          "dd1a1180459b9eb1a3e1a02716d0406722f97d2e0eb1b9922ac551d0cdba1292")
    failures = len(RESULTS) - sum(RESULTS)
    print("\n%d tests, %d passed, %d failed -> %s" %
          (len(RESULTS), sum(RESULTS), failures, "PASS" if failures == 0 else "FAIL"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
