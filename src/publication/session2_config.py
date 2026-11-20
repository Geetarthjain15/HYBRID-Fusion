"""
Session 2 scope and pre-flight contract.

Session 2 completes the Phase 3 ablation: three training variants x three seeds,
plus three evaluation-only `no_tta` cells that re-score the verified recovered
FusionV2 checkpoints without test-time augmentation.

`no_tta` is the reason this session needs a checkpoint input. It differs from
the reference only at inference, so retraining it would answer a different
question - it would measure "a model trained again, evaluated without TTA"
rather than "the reference model, evaluated without TTA". The distinction
matters because seed-to-seed training noise (s.d. 0.07 pp on this split) is the
same order as the effect being measured.

`no_ema` must train. The pre-EMA weights were never saved: the trainer
checkpoints `eval_model`, which IS the EMA copy when EMA is on, so the raw
weights were discarded at save time. Validation-based selection also used the
EMA copy, so a no_ema run would have selected a different epoch. Evaluation-only
is not available for it, and pretending otherwise would be wrong.
"""

from __future__ import annotations

# ---- scope ---------------------------------------------------------------
SESSION = "session_2"
SPLIT = "clean"
SEEDS = [42, 43, 44]

APPROVED_VARIANTS = ["no_ema", "cnn_branch_only", "vit_branch_only"]
EVAL_ONLY_VARIANTS = ["no_tta"]

OUT_OF_SCOPE = {
    "no_gate": "Session 1 - already trained there",
    "no_multiscale": "Session 1 - already trained there",
    "no_auxiliary_loss": "Session 1 - already trained there",
    "full_fusion_v2": "reused - predictions and verified checkpoints already exist; "
                      "retraining it would replace a verified reference with an "
                      "unverified one",
}

# ---- pre-flight fingerprint of the duplicate-clean split -----------------
EXPECTED = {
    "n_pooled": 7023,
    "n_clusters": 5648,
    "n_redundant_dropped": 1375,
    "n_train": 4800,          # images written to dataset_clean/Training
    "n_test": 848,
    "n_model_train": 4080,    # after the seeded 15% validation carve-out
    "n_val": 720,
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
# Transcribed from runs_v2_clean/fusion_v2_seed42/metrics.json. Embedded because
# the run directory is not uploaded: shipping existing outputs risks their being
# mistaken for new results. Checks read the file when present, this when not.
REFERENCE_CONFIG = {
    "arch": "fusion_v2",
    "unfreeze_all_cnn": True, "unfreeze_all_vit": True,
    "backbone_lr_mult": 0.1, "aux_weight": 0.3, "ema": True, "tta": True,
    "lr": 0.0001, "weight_decay": 0.05, "label_smoothing": 0.1,
    "batch_size": 16, "epochs": 15,
    "params_total": 91211918, "params_trainable": 91211918,
    "n_train": 4080, "n_val": 720, "n_test": 848,
}
REFERENCE_SOURCE = "runs_v2_clean/fusion_v2_seed42/metrics.json"

# ---- recovered checkpoints required by no_tta ----------------------------
# Verified in the Phase 3 addendum: each loaded strictly against the reference
# architecture (578/578 keys, 0 missing, 0 unexpected, 0 shape mismatches),
# retained a learned input-dependent gate, and reproduced its stored clean-test
# predictions exactly - 0.0000 pp accuracy difference, 0 predicted-label
# mismatches, max absolute probability difference ~4e-06 from float
# non-determinism between CUDA training and CPU re-scoring.
#
# The notebook re-checks every hash before no_tta runs. A checkpoint that
# differs by one byte is a different model, and re-scoring it would silently
# answer a question about weights nobody verified.
RECOVERED_CHECKPOINTS = {
    42: {"relative_path": "clean/full_fusion_v2/seed_42/checkpoint.pth",
         "seed": 42, "bytes": 365236755,
         "sha256": "9b328ba83df5d44b542e19faae68fd1493fee61e0c5b80e493c2f8a387ed592e",
         "architecture": "FusionV2 (gated, multi-scale, proj_dim 512)",
         "verification_status": "ACCEPTED - strict load, gate intact, predictions reproduced",
         "expected_clean_test_accuracy": 0.9823},
    43: {"relative_path": "clean/full_fusion_v2/seed_43/checkpoint.pth",
         "seed": 43, "bytes": 365236755,
         "sha256": "9f97b6aa0c9ec3f1124e593c8c29e11d31f4aef8bc08e20a40e7ed1f29e80f25",
         "architecture": "FusionV2 (gated, multi-scale, proj_dim 512)",
         "verification_status": "ACCEPTED - strict load, gate intact, predictions reproduced",
         "expected_clean_test_accuracy": 0.9823},
    44: {"relative_path": "clean/full_fusion_v2/seed_44/checkpoint.pth",
         "seed": 44, "bytes": 365236755,
         "sha256": "4155a2b71df99a6159e1e4a5dbe635f4e9ea7852a6704d804dd1fb5d43aca434",
         "architecture": "FusionV2 (gated, multi-scale, proj_dim 512)",
         "verification_status": "ACCEPTED - strict load, gate intact, predictions reproduced",
         "expected_clean_test_accuracy": 0.9835},
}

# ---- artifacts -----------------------------------------------------------
REQUIRED_ARTIFACTS = [
    "checkpoint.pth", "metrics.json", "history.json",
    "preds_test.csv", "preds_dataset_ood.csv",
    "config.json", "environment.json", "run.log",
]
# no_tta trains nothing, so it must NOT write a checkpoint or a history; it must
# record which checkpoint it scored.
EVAL_ONLY_ARTIFACTS = [
    "metrics.json", "preds_test.csv", "preds_dataset_ood.csv",
    "config.json", "environment.json", "run.log", "source_checkpoint_sha256.txt",
]
EVAL_ONLY_FORBIDDEN = ["checkpoint.pth"]

# Gate CSVs exist only where a gate runs in the forward pass. The branch-only
# variants have no gate at all; no_tta scores the gated reference, so it has one.
GATE_ARTIFACTS = ["gates_test.csv", "gates_dataset_ood.csv"]
VARIANTS_WITH_GATES = ["no_ema", "no_tta"]

EXPECTED_ROWS = {"preds_test.csv": 848, "preds_dataset_ood.csv": 499,
                 "gates_test.csv": 848, "gates_dataset_ood.csv": 499}

# ---- overfitting surveillance -------------------------------------------
# Unchanged from Session 1: the recipe is fixed so the ablation stays
# controlled, and overfitting is measured rather than newly defended against.
# no_ema is the variant to watch - removing weight averaging is exactly the kind
# of change that can widen the train-validation gap.
OVERFIT_REFERENCE = {
    "source": "runs_v2_clean/fusion_v2_seed42/history.json",
    "max_train_val_gap_pp": 1.62, "best_epoch": 8, "epochs_run": 12,
    "early_stopping_fired": True,
}
OVERFIT_FLAG_GAP_PP = 5.0
OVERFIT_FLAG_NO_EARLY_STOP = True

COST_MIN = {"no_ema": 45, "cnn_branch_only": 10, "vit_branch_only": 38, "no_tta": 3}


def is_in_scope(variant: str) -> tuple:
    if variant in APPROVED_VARIANTS:
        return True, "approved for %s (training)" % SESSION
    if variant in EVAL_ONLY_VARIANTS:
        return False, ("%s is evaluation-only in %s; it must not train. Use the "
                       "evaluation path." % (variant, SESSION))
    if variant in OUT_OF_SCOPE:
        return False, "out of scope for %s: %s" % (SESSION, OUT_OF_SCOPE[variant])
    return False, "unknown variant %r" % variant


def is_eval_in_scope(variant: str) -> tuple:
    if variant in EVAL_ONLY_VARIANTS:
        return True, "approved for %s (evaluation-only)" % SESSION
    return False, "%r is not an evaluation-only variant in %s" % (variant, SESSION)


def training_cells() -> list:
    return [{"variant": v, "seed": s, "split": SPLIT, "mode": "train"}
            for v in APPROVED_VARIANTS for s in SEEDS]


def eval_cells() -> list:
    return [{"variant": v, "seed": s, "split": SPLIT, "mode": "eval_only"}
            for v in EVAL_ONLY_VARIANTS for s in SEEDS]


def all_cells() -> list:
    return training_cells() + eval_cells()
