"""
Central experiment configuration.

BASELINE is an exact restatement of the hyperparameters hardcoded in the
original train.py. Running with BASELINE must reproduce the original model's
behaviour (up to the seed noise that the original had and never controlled).

Every improvement is an explicit flag that DEFAULTS TO OFF, so:

    BASELINE          -> all flags off
    BASELINE + A      -> one flag on
    BASELINE + A + B  -> two flags on

Ablations are therefore configuration, never source edits.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- identity -------------------------------------------------------
    name: str = "baseline"

    # ---- reproducibility (NEW - baseline had none of this) --------------
    # The original seeded only the val-split permutation. Seeding the rest is
    # a pure addition: it fixes which random draws happen, not what the
    # experiment is.
    seed: int = 42
    deterministic: bool = False       # cudnn deterministic; slower, exact

    # ---- data (unchanged from original) ---------------------------------
    train_dir: str = "dataset/Training"
    test_dir: str = "dataset/Testing"
    val_fraction: float = 0.15
    val_split_seed: int = 42          # original: torch.Generator().manual_seed(42)
    image_size: int = 224
    norm_mean: tuple = (0.5, 0.5, 0.5)
    norm_std: tuple = (0.5, 0.5, 0.5)

    # ---- augmentation (unchanged from original) -------------------------
    rrc_scale_min: float = 0.8
    rotation_degrees: int = 15
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2

    # ---- model (unchanged from original) --------------------------------
    num_classes: int = 4
    vit_weights_path: str = "src/vit_brain_tumor_5class.pth"
    # Set False to start the ViT branch from ImageNet weights instead of the
    # warm-start checkpoint. Relevant because that checkpoint currently
    # measures at chance (macro ROC-AUC 0.503) - see the audit.
    use_vit_warm_start: bool = True

    # ---- freezing (unchanged from original) ------------------------------
    unfreeze_cnn_last_block: bool = True
    unfreeze_vit_last_n_layers: int = 2

    # ---- optimisation (unchanged from original) --------------------------
    epochs: int = 15
    patience: int = 4
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    scheduler: str = "cosine"

    # ---- improvement flags (ALL OFF = baseline) --------------------------
    # A: per-class prior correction fitted on validation, applied at eval.
    enable_prior_correction: bool = False
    # B: class-weighted loss to counter class imbalance.
    enable_class_weights: bool = False

    # ---- output ----------------------------------------------------------
    out_checkpoint: str = "src/hybrid_brain_tumor.pth"
    out_metrics: str = "src/metrics.json"
    out_history: str = "src/history.json"

    def to_dict(self):
        return asdict(self)


BASELINE = Config(name="baseline")

# Example ablation arms - add here, never by editing training source.
PRIOR_CORRECTION = Config(name="prior_correction", enable_prior_correction=True)
NO_WARM_START = Config(name="no_warm_start", use_vit_warm_start=False)

REGISTRY = {c.name: c for c in [BASELINE, PRIOR_CORRECTION, NO_WARM_START]}


def get(name: str) -> Config:
    if name not in REGISTRY:
        raise KeyError(f"unknown config '{name}'. available: {sorted(REGISTRY)}")
    return REGISTRY[name]
