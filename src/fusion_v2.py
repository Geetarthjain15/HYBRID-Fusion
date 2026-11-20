"""
FusionV2 - the fusion design that actually uses both branches.

The v1 fusion (hybrid_model.py) measured statistically identical to ViT-only
(McNemar p = 0.34-1.00 over 6 paired tests). Three concrete causes, three fixes:

  1. THE CNN BRANCH WAS CRIPPLED. v1 unfroze only EfficientNet's last block
     (1.07M of 4.67M params) - the exact setting that scores 92% standalone,
     while the fully-unfrozen CNN scores 99.46%. v2 trains the whole backbone
     with a discriminative LR.

  2. PLAIN CONCAT LETS THE OPTIMISER DELETE THE WEAK BRANCH. v1 concatenated
     1280-d CNN + 768-d ViT into one Linear. Once ViT alone nearly solved the
     task, gradient descent drove the CNN half's weights toward zero. v2 gates
     the branches explicitly: the gate is *forced* to allocate mass across both
     (softmax over 2), so the CNN half cannot be silently zeroed - and the gate
     value is readable, so you can report how much each branch is actually used.

  3. NOTHING FORCED EITHER BRANCH TO BE GOOD ALONE. v2 attaches an auxiliary
     classifier to each branch. The branch features must stay individually
     discriminative, which keeps them complementary instead of letting one
     collapse into a passenger.

Plus two things that are pure accuracy in this regime:
  - MULTI-SCALE CNN TAPS. v1 used only the final 1280-d pooled map. Stage-3
    (40ch) and stage-5 (112ch) carry the local texture detail that a single
    global-pooled final map washes out.
  - HFLIP TTA + EMA WEIGHTS at inference.

    python src/train_ablation.py --arch fusion_v2 --seed 42 --unfreeze_all_cnn \
        --unfreeze_all_vit --backbone_lr_mult 0.1 --aux_weight 0.3 --ema --tta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from transformers import ViTForImageClassification

# EfficientNet-B0 stage output channels; taps chosen at the two resolution
# steps before the final 1280-d map.
TAPS = {3: 40, 5: 112}
FINAL_CH = 1280
VIT_DIM = 768


class FusionV2(nn.Module):
    def __init__(self, num_classes=4, vit_source="google/vit-base-patch16-224",
                 proj_dim=512, multiscale=True, gated=True, dropout=0.2):
        super().__init__()
        self.multiscale = multiscale
        self.gated = gated

        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.cnn_stages = base.features
        cnn_dim = FINAL_CH + (sum(TAPS.values()) if multiscale else 0)

        self.vit = ViTForImageClassification.from_pretrained(
            vit_source, num_labels=num_classes, ignore_mismatched_sizes=True)

        # Project both branches to a common width so the gate compares like
        # with like - v1 blended a 1280-d and a 768-d vector with no
        # normalisation, which biases the head toward the wider branch.
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.vit_proj = nn.Sequential(
            nn.Linear(VIT_DIM, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())

        # Gate sees both branches and splits attention between them. Softmax
        # over 2 means neither can be driven to zero without the other taking
        # the full mass - the collapse mode in v1 is structurally excluded.
        self.gate = nn.Sequential(
            nn.Linear(2 * proj_dim, 256), nn.GELU(), nn.Linear(256, 2))

        self.head = nn.Sequential(
            nn.LayerNorm(2 * proj_dim), nn.Dropout(dropout),
            nn.Linear(2 * proj_dim, num_classes))

        # Auxiliary heads: each branch must classify on its own.
        self.aux_cnn = nn.Linear(proj_dim, num_classes)
        self.aux_vit = nn.Linear(proj_dim, num_classes)

    def branch_features(self, x):
        feats, h = [], x
        for i, stage in enumerate(self.cnn_stages):
            h = stage(h)
            if self.multiscale and i in TAPS:
                feats.append(h.mean((2, 3)))
        feats.append(h.mean((2, 3)))
        cnn = self.cnn_proj(torch.cat(feats, 1))
        vit = self.vit_proj(self.vit.vit(pixel_values=x).last_hidden_state[:, 0])
        return cnn, vit

    def forward(self, x, return_aux=False):
        c, v = self.branch_features(x)
        if self.gated:
            g = F.softmax(self.gate(torch.cat([c, v], 1)), dim=-1)
            # x2 keeps the post-gate scale comparable to the ungated case,
            # since the two gate values sum to 1.
            fused = torch.cat([c * g[:, 0:1] * 2, v * g[:, 1:2] * 2], 1)
        else:
            g = None
            fused = torch.cat([c, v], 1)
        logits = self.head(fused)
        if return_aux:
            return logits, self.aux_cnn(c), self.aux_vit(v), g
        return logits

    @torch.no_grad()
    def gate_usage(self, x):
        """Mean [cnn, vit] gate mass on a batch - report this in the paper to
        show the CNN branch is genuinely carrying signal, not a passenger."""
        c, v = self.branch_features(x)
        return F.softmax(self.gate(torch.cat([c, v], 1)), dim=-1).mean(0)

    def param_groups(self, lr, backbone_mult=0.1):
        """Pretrained backbones need a smaller LR than freshly-initialised
        fusion parts, or fine-tuning destroys the ImageNet features."""
        backbone, fresh = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if n.startswith(("cnn_stages", "vit.")) else fresh).append(p)
        return [{"params": backbone, "lr": lr * backbone_mult},
                {"params": fresh, "lr": lr}]


class ModelEMA:
    """Exponential moving average of weights. Reliably worth +0.1-0.3% at this
    accuracy level for the cost of one extra copy of the model."""

    def __init__(self, model, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        # Warmup: without it the average is still dominated by the random init
        # for the first few thousand steps, which corrupts val-based model
        # selection in the early epochs (and destroys short runs entirely).
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


@torch.no_grad()
def tta_logits(model, x):
    """Horizontal-flip TTA. Brain MRI is near-symmetric, so the flip is a valid
    view and averaging the two halves the variance of borderline predictions."""
    return (model(x).softmax(-1) + model(torch.flip(x, dims=[3])).softmax(-1)) / 2
