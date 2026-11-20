"""
Mode- and architecture-specific artifact requirements (Phase 3A).

The Session 2 run reported "3/12 complete". That was a validator-contract
defect, not a training failure: the older rule assumed every variant except
`no_gate` carries a learned fusion gate, which is false for the single-branch
controls. Demanding gate CSVs from a model that has no gate fails a correct run.

This module states the requirement per variant from the experimental design
rather than from what happens to be on disk. Two rules govern it:

  REQUIREMENTS FOLLOW ARCHITECTURE, NOT AVAILABILITY. A file is not optional
  because it is missing. `cnn_branch_only` has no gate in its forward pass, so
  gate CSVs are *expected to be absent*; `no_ema` is the full gated model, so
  their absence is a genuine failure.

  MODE DETERMINES THE CONTRACT. An evaluation-only run scores existing weights.
  It must NOT produce a checkpoint or a training history - their presence means
  something trained, which would silently answer a different question - and it
  MUST record which weights it scored.

Nothing here is loosened to make a run pass.
"""

from __future__ import annotations

# ---- numerical tolerances, stated once -----------------------------------
PROB_SUM_TOL = 1e-3        # |sum(p) - 1| per row
METRIC_TOL_PP = 0.01       # recomputed vs stored, percentage points
CLEAN_TEST_N = 848         # duplicate-clean test split
EXTERNAL_N = 499           # dataset_ood

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# ---- per-variant design facts --------------------------------------------
# has_gate: does a learned fusion gate run in the forward pass?
# mode:     "train" produces weights; "eval_only" scores existing weights.
# differs_from_reference_in: the single switch this ablation changes.
VARIANT_DESIGN = {
    # --- Session 1 ---
    "no_gate": {
        "session": "session_1", "mode": "train", "has_gate": False,
        "differs_from_reference_in": "gated",
        "note": "the gate is bypassed in forward(), so no gate mass is emitted",
    },
    "no_multiscale": {
        "session": "session_1", "mode": "train", "has_gate": True,
        "differs_from_reference_in": "multiscale",
    },
    "no_auxiliary_loss": {
        "session": "session_1", "mode": "train", "has_gate": True,
        "differs_from_reference_in": "aux_weight",
    },
    # --- Session 2 ---
    "no_ema": {
        "session": "session_2", "mode": "train", "has_gate": True,
        "differs_from_reference_in": "ema",
        "note": ("EMA disabled: validation selection and the saved checkpoint must "
                 "both use the raw live weights, not an EMA shadow"),
    },
    "cnn_branch_only": {
        "session": "session_2", "mode": "train", "has_gate": False,
        "differs_from_reference_in": "architecture (ViT branch removed)",
        "note": "single-branch model - no gate exists to dump",
    },
    "vit_branch_only": {
        "session": "session_2", "mode": "train", "has_gate": False,
        "differs_from_reference_in": "architecture (CNN branch removed)",
        "note": "single-branch model - no gate exists to dump",
    },
    "no_tta": {
        "session": "session_2", "mode": "eval_only", "has_gate": True,
        "differs_from_reference_in": "tta",
        "note": ("scores the verified recovered FusionV2 checkpoints; the model is "
                 "the gated reference, so gate telemetry is available"),
    },
    # --- reference, kept as its own category ---
    "full_fusion_v2": {
        "session": "reference", "mode": "train", "has_gate": True,
        "differs_from_reference_in": None,
        "note": "the reference itself - reused, never a newly trained ablation",
    },
}

TRAIN_REQUIRED = ["config.json", "environment.json", "run.log", "metrics.json",
                  "history.json", "preds_test.csv", "preds_dataset_ood.csv",
                  "checkpoint.pth"]
EVAL_REQUIRED = ["config.json", "environment.json", "run.log", "metrics.json",
                 "preds_test.csv", "preds_dataset_ood.csv",
                 "source_checkpoint_sha256.txt"]
# An evaluation-only run that produced these did something it must not do.
EVAL_FORBIDDEN = ["checkpoint.pth", "history.json"]
GATE_FILES = ["gates_test.csv", "gates_dataset_ood.csv"]

EXPECTED_ROWS = {"preds_test.csv": CLEAN_TEST_N,
                 "preds_dataset_ood.csv": EXTERNAL_N,
                 "gates_test.csv": CLEAN_TEST_N,
                 "gates_dataset_ood.csv": EXTERNAL_N}


def design(variant: str) -> dict:
    if variant not in VARIANT_DESIGN:
        raise KeyError("no documented design for variant %r; refusing to guess "
                       "its artifact contract" % variant)
    return VARIANT_DESIGN[variant]


def required_artifacts(variant: str) -> list:
    """What this variant MUST produce, from its design."""
    d = design(variant)
    req = list(TRAIN_REQUIRED) if d["mode"] == "train" else list(EVAL_REQUIRED)
    if d["has_gate"]:
        req += GATE_FILES
    return req


def forbidden_artifacts(variant: str) -> list:
    """What this variant must NOT produce. For a gateless model the gate files
    are not merely unnecessary - their presence would mean a gate ran."""
    d = design(variant)
    forbidden = list(EVAL_FORBIDDEN) if d["mode"] == "eval_only" else []
    if not d["has_gate"]:
        forbidden += GATE_FILES
    return forbidden


def expected_cells() -> dict:
    """The authoritative matrix, by category. Reference runs are a separate
    category and must not be counted as newly trained ablations."""
    seeds = [42, 43, 44]
    return {
        "session_1_training": [(v, s) for v in
                               ("no_gate", "no_multiscale", "no_auxiliary_loss")
                               for s in seeds],
        "session_2_training": [(v, s) for v in
                               ("no_ema", "cnn_branch_only", "vit_branch_only")
                               for s in seeds],
        "session_2_eval_only": [("no_tta", s) for s in seeds],
        "reference_reused": [("full_fusion_v2", s) for s in seeds],
    }


def summarise() -> str:
    lines = ["variant              session      mode        gate  required  forbidden"]
    for v, d in VARIANT_DESIGN.items():
        lines.append("%-20s %-12s %-11s %-5s %-9d %d"
                     % (v, d["session"], d["mode"], d["has_gate"],
                        len(required_artifacts(v)), len(forbidden_artifacts(v))))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarise())
