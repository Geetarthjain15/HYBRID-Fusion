"""Phase X5 - composite map construction, per the locked protocol.

Two composites, both formed on the locked 14x14 common grid from maps that have
already been resampled and min-max normalised:

    composite_equal          0.5 * cnn + 0.5 * vit
    composite_gate_weighted  gate[:,0] * cnn + gate[:,1] * vit

The gate values come from the SAME forward pass that produced the maps and the
target class, so the weighting describes that one decision.

Gate mass measures branch utilisation. It is not causal importance and it is not
evidence that either branch improves the prediction. X3 deliberately formed no
composite; composites exist only from X5 onward, and only as evaluated methods.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

COMMON_GRID = (14, 14)


def to_common(m: torch.Tensor) -> torch.Tensor:
    """Resample to 14x14 (bilinear, align_corners=False), then min-max to [0,1].

    Locked order: resample FIRST, normalise SECOND.
    """
    t = m if m.dim() == 4 else m.reshape(1, 1, *m.shape[-2:])
    t = t.float()
    if tuple(t.shape[-2:]) != COMMON_GRID:
        t = F.interpolate(t, size=COMMON_GRID, mode="bilinear", align_corners=False)
    t = t[0, 0]
    lo, hi = t.min(), t.max()
    return (t - lo) / (hi - lo) if float(hi) > float(lo) else torch.zeros_like(t)


def equal_weight(cnn14: torch.Tensor, vit14: torch.Tensor) -> torch.Tensor:
    return to_common(0.5 * cnn14 + 0.5 * vit14)


def gate_weighted(cnn14: torch.Tensor, vit14: torch.Tensor,
                  g_cnn: float, g_vit: float) -> torch.Tensor:
    return to_common(g_cnn * cnn14 + g_vit * vit14)


def centre_prior(grid=COMMON_GRID) -> torch.Tensor:
    """Deterministic isotropic Gaussian, sigma = grid/4. Null baseline."""
    h, w = grid
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sig = max(h, w) / 4.0
    return to_common(torch.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))))


def random_map(generator: torch.Generator, grid=COMMON_GRID) -> torch.Tensor:
    """Deterministic uniform random map. Null baseline."""
    return to_common(torch.rand(grid, generator=generator, dtype=torch.float32))
