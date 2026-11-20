"""
Session 1 scope and pre-flight contract.

Session 1 trains exactly nine clean-split cells: three variants x three seeds.
Everything else in the Phase 3 matrix is explicitly out of scope and is listed
here so the driver can refuse it rather than rely on a human reading a plan.

The pre-flight constants are the fingerprint of the duplicate-clean split as it
was verified locally. The driver rebuilds the split on Kaggle and must match
every one of them before a single gradient step is taken. If the rebuild
diverges - a different source dataset, a changed hashing threshold, a partial
upload - the numbers will differ and the session stops. That check is the only
thing standing between "we retrained on the same split" and "we think we did".
"""

from __future__ import annotations

# ---- scope ---------------------------------------------------------------
SESSION = "session_1"
SPLIT = "clean"
SEEDS = [42, 43, 44]

APPROVED_VARIANTS = ["no_gate", "no_multiscale", "no_auxiliary_loss"]

# Named explicitly so the driver can give a precise refusal, not a generic one.
OUT_OF_SCOPE = {
    "no_ema": "Session 2 - training-time ablation, pre-EMA weights were never saved",
    "cnn_branch_only": "Session 2 - branch-only control, needs training",
    "vit_branch_only": "Session 2 - branch-only control, needs training",
    "no_tta": "evaluation-only, uses the recovered checkpoints; not a training cell",
    "full_fusion_v2": "reused - predictions and verified checkpoints already exist",
}

# ---- pre-flight fingerprint of the duplicate-clean split -----------------
# Source: dataset_clean/split_report.json and split_manifest.json, verified
# locally in the Phase 3 addendum when the recovered checkpoints reproduced
# their stored predictions on exactly this split.
EXPECTED = {
    "n_pooled": 7023,
    "n_clusters": 5648,
    "n_redundant_dropped": 1375,
    "n_train": 4800,
    "n_test": 848,
    "hamming_threshold": 8,
    "split_seed": 42,
    "class_counts": {
        "Training": {"glioma": 1377, "meningioma": 1290, "notumor": 655, "pituitary": 1478},
        "Testing": {"glioma": 243, "meningioma": 228, "notumor": 116, "pituitary": 261},
    },
    "split_manifest_sha256":
        "a02bcb62847ce1b9146a3e769f41f88ec1afce5a46d03fb5aaffb0ef43a0e6d2",
    "split_report_sha256":
        "8487d31c0f0b06a4afaf493f6a5597e8f8e45feafe7679b77a745b070e34abc0",
}

# ---- reference FusionV2 configuration ------------------------------------
# Transcribed from runs_v2_clean/fusion_v2_seed42/metrics.json, which is the run
# whose checkpoints were verified in the Phase 3 addendum (strict load, gate
# intact, predictions reproduced exactly on all 848 clean-test images).
#
# It is embedded because the run directory is deliberately NOT uploaded to
# Kaggle: shipping existing outputs risks their being mistaken for new results.
# The ablation still has to prove it inherits this configuration, so the check
# reads the file when present and falls back to this fingerprint when it is not,
# recording which source it used either way.
REFERENCE_CONFIG = {
    "arch": "fusion_v2",
    "unfreeze_all_cnn": True,
    "unfreeze_all_vit": True,
    "backbone_lr_mult": 0.1,
    "aux_weight": 0.3,
    "ema": True,
    "tta": True,
    "lr": 0.0001,
    "weight_decay": 0.05,
    "label_smoothing": 0.1,
    "batch_size": 16,
    "epochs": 15,
    "params_total": 91211918,
    "params_trainable": 91211918,
    "n_train": 4080, "n_val": 720, "n_test": 848,
}
REFERENCE_SOURCE = "runs_v2_clean/fusion_v2_seed42/metrics.json"

# ---- artifacts every completed run must contain --------------------------
# Presence of metrics.json alone is NOT a completion signal: a run killed
# between writing metrics and writing predictions would look finished. The
# validator requires the whole set, and checks each file is parseable and has
# the expected row count.
REQUIRED_ARTIFACTS = [
    "checkpoint.pth", "metrics.json", "history.json",
    "preds_test.csv", "preds_dataset_ood.csv",
    "config.json", "environment.json", "run.log",
]
# gates_*.csv are required only where a gate exists. no_gate has none by
# definition, so demanding them there would be wrong.
GATE_ARTIFACTS = ["gates_test.csv", "gates_dataset_ood.csv"]

EXPECTED_ROWS = {"preds_test.csv": 848, "preds_dataset_ood.csv": 499,
                 "gates_test.csv": 848, "gates_dataset_ood.csv": 499}

# ---- overfitting surveillance -------------------------------------------
# Session 1 must not change the training recipe: these are controlled
# ablations, and altering regularisation would confound the component being
# tested with a change in how hard the model is allowed to fit. So overfitting
# is MEASURED rather than newly defended against.
#
# Guards already in the reviewed recipe, unchanged:
#   - validation carved from the TRAINING partition only, seeded, 15%
#   - early stopping on validation accuracy, patience 4
#   - the checkpoint saved is the BEST-validation epoch, not the last
#   - label smoothing 0.1, weight decay 0.05
#   - RandomResizedCrop(0.8-1.0), horizontal flip, rotation 15, colour jitter
#   - EMA of weights (where the variant keeps it)
#   - the duplicate-clean split itself removes near-duplicate leakage
#
# Reference behaviour on this split (runs_v2_clean/fusion_v2_seed42): the
# train-validation gap peaks at +1.62 pp at epoch 12, early stopping fires at
# epoch 12 having selected epoch 8. A variant that departs markedly from that
# is flagged for inspection - not silently accepted, and not silently "fixed".
OVERFIT_REFERENCE = {
    "source": "runs_v2_clean/fusion_v2_seed42/history.json",
    "max_train_val_gap_pp": 1.62,
    "best_epoch": 8,
    "epochs_run": 12,
    "early_stopping_fired": True,
}
OVERFIT_FLAG_GAP_PP = 5.0        # flag for review, never auto-correct
OVERFIT_FLAG_NO_EARLY_STOP = True  # ran the full budget without early stopping


def is_in_scope(variant: str) -> tuple:
    """(allowed, reason). Refusal carries the reason, so a mistyped run name is
    distinguishable from a deliberately deferred one."""
    if variant in APPROVED_VARIANTS:
        return True, "approved for %s" % SESSION
    if variant in OUT_OF_SCOPE:
        return False, "out of scope for %s: %s" % (SESSION, OUT_OF_SCOPE[variant])
    return False, "unknown variant %r" % variant


def cells() -> list:
    return [{"variant": v, "seed": s, "split": SPLIT}
            for v in APPROVED_VARIANTS for s in SEEDS]
