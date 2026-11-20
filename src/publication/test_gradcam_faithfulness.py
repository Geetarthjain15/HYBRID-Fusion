"""Cheap unit tests for Phase 6 faithfulness helpers."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit_gradcam_faithfulness as A


def test_normalized_auc_constant_curve():
    assert abs(A.normalized_auc([0.5, 0.5, 0.5]) - 0.5) < 1e-12


def test_coarse_order_is_complete_permutation():
    heatmap = np.arange(224 * 224, dtype=np.float32).reshape(224, 224)
    order = A.coarse_order(heatmap)
    assert order.shape == (196,)
    assert sorted(order.tolist()) == list(range(196))


def test_block_mask_endpoints():
    order = np.arange(196)
    assert float(A.block_mask(order, 0.0).sum()) == 0.0
    assert float(A.block_mask(order, 1.0).sum()) == 224.0 * 224.0


def test_flip_correlation():
    data = np.arange(64, dtype=np.float64).reshape(8, 8)
    assert abs(A.finite_correlation(data, data) - 1.0) < 1e-12


def test_no_tumor_semantics_are_explicit():
    source = Path(A.__file__).read_text(encoding="utf-8")
    assert "support for the no-tumor class" in source
    assert "not evidence that a tumor exists" in source
    assert "Localization accuracy: **not available**" in source


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")

