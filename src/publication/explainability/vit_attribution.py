"""Phase X3 - ViT patch-embedding Gradient x Input (the verified X2b method).

No new attribution mathematics. This is the X2b relevance expression factored
out so X3 can apply it to a gradient obtained from the SAME synchronized forward
pass as the CNN map, instead of a second forward. `test_attribution.py` asserts
it is bit-identical to `sdpa_token_attribution.TokenAttributor.attribute`.
"""

from __future__ import annotations

import torch

VIT_SITE = "vit.vit.embeddings"
N_PATCH = 196
VIT_GRID = 14


def patch_relevance(grad: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """[B,197,768] grad and hidden -> [B,14,14] relevance; CLS (index 0) dropped."""
    if hidden.dim() != 3 or hidden.shape[1] != N_PATCH + 1:
        raise ValueError("unexpected hidden shape %s" % (tuple(hidden.shape),))
    rel = (grad[:, 1:] * hidden[:, 1:]).sum(-1)
    return rel.detach().reshape(-1, VIT_GRID, VIT_GRID)
