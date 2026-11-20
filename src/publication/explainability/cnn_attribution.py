"""Phase X3 - CNN-path Grad-CAM++ at native stage-8 resolution (7x7).

The verified helper `fusion_v2_gradcam.gradcam_plus_plus` goes through
`pytorch_grad_cam`, whose `BaseCAM.compute_cam_per_layer` always upsamples to
the input size before returning, so the native grid is never exposed. This
module computes the same quantity at native resolution:

    weights_c = sum_ij  relu(g_cij) * a_cij,
    a_cij     = g^2 / (2 g^2 + (sum_ij A_cij) g^3 + eps),  a_cij = 0 where g == 0
    cam       = relu( sum_c weights_c * A_c )

which is Eq. 19 of Chattopadhay et al. (arXiv:1710.11063) exactly as
`pytorch_grad_cam.GradCAMPlusPlus.get_cam_weights` implements it, evaluated in
numpy float32 like the library. Target layer and target are identical to the
helper: `model.cnn_stages[-1]` and the complete fused-model logit.

`rescale_like_library` reproduces the library's post-processing so the native
map can be checked against the helper (amendment §3, 1e-5).
"""

from __future__ import annotations

import numpy as np

EPS = 0.000001            # identical to grad_cam_plusplus.py
CNN_GRID = 7


def gradcam_pp_weights(activations: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """[B,C,H,W] float32 activations/grads -> [B,C] Grad-CAM++ weights."""
    g2 = grads ** 2
    g3 = g2 * grads
    sum_a = activations.sum(axis=(2, 3), keepdims=True)
    aij = g2 / (2 * g2 + sum_a * g3 + EPS)
    aij = np.where(grads != 0, aij, 0)
    return (np.maximum(grads, 0) * aij).sum(axis=(2, 3))


def gradcam_pp_native(activations: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """[B,C,H,W] -> [B,H,W] native-resolution, ReLU'd, UNnormalised CAM."""
    a = np.asarray(activations, dtype=np.float32)
    g = np.asarray(grads, dtype=np.float32)
    w = gradcam_pp_weights(a, g)
    cam = (w[:, :, None, None] * a).sum(axis=1)
    return np.maximum(cam, 0).astype(np.float32)


def rescale_like_library(cam_native: np.ndarray, size: int = 224) -> np.ndarray:
    """Replicate the helper's full post-processing on one [H,W] native map.

    library compute_cam_per_layer : min-max (1e-7) -> cv2 bilinear resize
    library aggregate_multi_layers: relu -> mean over 1 layer -> min-max (1e-7)
    helper gradcam_plus_plus      : min-max (exact) or zeros if constant
    """
    import cv2
    img = cam_native.astype(np.float32)
    img = img - img.min()
    img = img / (1e-7 + img.max())
    img = cv2.resize(np.float32(img), (size, size))
    img = np.maximum(img, 0)
    img = img - img.min()
    img = img / (1e-7 + img.max())
    img = np.float32(img)
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
