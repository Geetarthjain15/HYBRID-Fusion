"""Cheap tests for the cross-source ablation analysis helpers."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_cross_source_ablations as A


def test_mean_sd_uses_sample_sd():
    mean, sd = A.mean_sd([1.0, 2.0, 3.0])
    assert mean == 2.0 and sd == 1.0


def test_macro_f1_perfect():
    y = np.asarray([0, 1, 2, 3] * 2)
    assert A.macro_f1(y, y) == 1.0


def test_paired_counts():
    base = {"names": ["a", "b", "c", "d"], "y": np.asarray([0, 0, 1, 1]),
            "pred": np.asarray([0, 0, 0, 1])}
    other = {"names": ["a", "b", "c", "d"], "y": np.asarray([0, 0, 1, 1]),
             "pred": np.asarray([0, 1, 1, 1])}
    result = A.paired_counts(base, other)
    assert result == {"both_correct": 2, "reference_correct_ablation_wrong": 1,
                      "reference_wrong_ablation_correct": 1, "both_wrong": 0}


def test_fixed_family_size():
    assert len(A.VARIANTS) * len(A.SEEDS) == 21


def test_external_labels_are_evaluation_only():
    source = Path(A.__file__).read_text(encoding="utf-8")
    assert "no tuning, selection, weighting, or thresholding" in source
    assert "EXPECTED_DISJOINT = 373" in source


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test(); print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")

