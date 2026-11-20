"""
Branch-only controls for the FusionV2 ablation (Phase 3).

FusionV2 already exposes gate, multi-scale, auxiliary-loss, EMA and TTA as
switches, so those five ablations need no new model code. The two branch-only
controls do need code, because FusionV2 has no single-branch mode.

The point of these classes is a FAIR control, not merely "a CNN" or "a ViT".
Each keeps everything about FusionV2 except the removed branch:

  - the same backbone and the same trainability policy;
  - the same projection head (Linear -> LayerNorm -> GELU to proj_dim), so the
    representation entering the classifier has the same width and
    normalisation as the corresponding half of the fusion;
  - the same classifier shape, narrowed from 2*proj_dim to proj_dim because
    only one branch feeds it;
  - the same discriminative-learning-rate split between pretrained backbone and
    freshly initialised parts.

The removed branch is NOT constructed. Zeroing a gate would leave the other
backbone in the graph, still consuming compute and still contributing through
BatchNorm/LayerNorm statistics; a reviewer could then reasonably object that
"CNN-only" still ran a ViT. Absence is verifiable, masking is not, and
test_ablations.py asserts the absence.

Nothing here modifies src/fusion_v2.py - constants are imported from it so the
two cannot drift apart.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import ViTForImageClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_v2 import TAPS, FINAL_CH, VIT_DIM  # noqa: E402  single source of truth


class CNNBranchOnly(nn.Module):
    """FusionV2 with the ViT branch removed entirely.

    Keeps the multi-scale taps and the projection head so the only difference
    from the fusion's CNN half is that nothing is concatenated to it."""

    branch = "cnn"
    removed_branch = "vit"

    def __init__(self, num_classes: int = 4, proj_dim: int = 512,
                 multiscale: bool = True, dropout: float = 0.2):
        super().__init__()
        self.multiscale = multiscale
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.cnn_stages = base.features
        cnn_dim = FINAL_CH + (sum(TAPS.values()) if multiscale else 0)
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim), nn.Dropout(dropout), nn.Linear(proj_dim, num_classes))
        self.aux_cnn = nn.Linear(proj_dim, num_classes)

    def branch_features(self, x):
        feats, h = [], x
        for i, stage in enumerate(self.cnn_stages):
            h = stage(h)
            if self.multiscale and i in TAPS:
                feats.append(h.mean((2, 3)))
        feats.append(h.mean((2, 3)))
        return self.cnn_proj(torch.cat(feats, 1))

    def forward(self, x, return_aux: bool = False):
        c = self.branch_features(x)
        logits = self.head(c)
        if return_aux:
            # No gate exists with one branch; the aux head mirrors the main one
            # so the training loop's loss composition stays structurally identical.
            return logits, self.aux_cnn(c), None, None
        return logits

    def param_groups(self, lr: float, backbone_mult: float = 0.1):
        backbone, fresh = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if n.startswith("cnn_stages") else fresh).append(p)
        return [{"params": backbone, "lr": lr * backbone_mult},
                {"params": fresh, "lr": lr}]


class ViTBranchOnly(nn.Module):
    """FusionV2 with the CNN branch removed entirely."""

    branch = "vit"
    removed_branch = "cnn"

    def __init__(self, num_classes: int = 4, proj_dim: int = 512,
                 vit_source: str = "google/vit-base-patch16-224", dropout: float = 0.2):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            vit_source, num_labels=num_classes, ignore_mismatched_sizes=True)
        self.vit_proj = nn.Sequential(
            nn.Linear(VIT_DIM, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim), nn.Dropout(dropout), nn.Linear(proj_dim, num_classes))
        self.aux_vit = nn.Linear(proj_dim, num_classes)

    def branch_features(self, x):
        cls = self.vit.vit(pixel_values=x).last_hidden_state[:, 0]
        return self.vit_proj(cls)

    def forward(self, x, return_aux: bool = False):
        v = self.branch_features(x)
        logits = self.head(v)
        if return_aux:
            return logits, self.aux_vit(v), None, None
        return logits

    def param_groups(self, lr: float, backbone_mult: float = 0.1):
        backbone, fresh = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if n.startswith("vit.") else fresh).append(p)
        return [{"params": backbone, "lr": lr * backbone_mult},
                {"params": fresh, "lr": lr}]


def build_branch_only(arch: str, unfreeze_all_cnn: bool, unfreeze_all_vit: bool,
                      vit_source: str = "google/vit-base-patch16-224",
                      multiscale: bool = True):
    """Apply the SAME freezing policy FusionV2 uses, to the surviving branch."""
    if arch == "cnn_branch_only":
        m = CNNBranchOnly(multiscale=multiscale)
        for p in m.cnn_stages.parameters():
            p.requires_grad = bool(unfreeze_all_cnn)
        for p in m.cnn_stages[-1].parameters():
            p.requires_grad = True
        trainable_heads = (m.cnn_proj, m.head, m.aux_cnn)
    elif arch == "vit_branch_only":
        m = ViTBranchOnly(vit_source=vit_source)
        for p in m.vit.parameters():
            p.requires_grad = bool(unfreeze_all_vit)
        for layer in m.vit.vit.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        trainable_heads = (m.vit_proj, m.head, m.aux_vit)
    else:
        raise ValueError("not a branch-only arch: %s" % arch)
    for mod in trainable_heads:
        for p in mod.parameters():
            p.requires_grad = True
    return m
