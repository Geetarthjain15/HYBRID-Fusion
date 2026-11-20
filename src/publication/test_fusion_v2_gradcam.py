"""Cheap tests for FusionV2 Grad-CAM++ demo helpers; no checkpoint download."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fusion_v2_gradcam as G


def test_class_order():
    assert G.CLASS_NAMES == ("glioma", "meningioma", "notumor", "pituitary")


def test_prepare_image_shapes_and_finiteness():
    im = Image.new("L", (91, 73), color=127)
    display, tensor, rgb = G.prepare_image(im)
    assert display.mode == "RGB" and display.size == (224, 224)
    assert tuple(tensor.shape) == (1, 3, 224, 224)
    assert rgb.shape == (224, 224, 3)
    assert np.isfinite(tensor.numpy()).all() and np.isfinite(rgb).all()


def test_overlay_shape_for_every_class_target():
    rgb = np.full((224, 224, 3), 0.5, dtype=np.float32)
    heat = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(224, 224)
    for target in range(len(G.CLASS_NAMES)):
        assert target in range(4)
        overlay = G.overlay_heatmap(rgb, heat)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8


def test_invalid_shapes_refused():
    try:
        G.overlay_heatmap(np.zeros((10, 10, 3), np.float32), np.zeros((224, 224), np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid display image shape accepted")


def test_app_source_has_requested_compact_ui_and_on_demand_heatmap():
    source = (HERE / "streamlit_fusion_v2.py").read_text(encoding="utf-8")
    assert "Generate Grad-CAM++ heatmap" in source
    assert "cross_source_disjoint" in source
    assert "Interpretation & limitations" not in source
    assert "Research benchmark only" not in source
    assert "Cross-source accuracy falls" not in source
    assert "not the probability that a medical diagnosis is correct" not in source


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} PASS")
