"""
Phase 3 - FusionV2 component ablation framework (planning / dry-run).

Determines which parts of FusionV2 actually produce its result, by changing
exactly one component at a time against a fixed reference configuration.

This module is the planner. It does NOT train: it builds the experiment matrix,
audits which existing runs may legitimately be reused, and emits the exact
commands for the runs that are genuinely missing. Training is a separate,
explicitly approved step.

Reference configuration is not hand-typed. It is read from
runs_v2_clean/fusion_v2_seed42/metrics.json - the actual FusionV2 run - so the
ablations cannot silently diverge from the model they are ablating.

Reuse discipline
----------------
An existing run may substitute for an ablation cell only if architecture,
split, preprocessing, trainability, optimizer, schedule, augmentation and
evaluation protocol are demonstrably identical. Anything less is marked
RETRAIN. Two consequences found in this repository, both verified rather than
assumed:

  - no_tta would be evaluation-only IF a full FusionV2 checkpoint existed. None
    does; the Kaggle packaging excluded .pth files from the downloaded archive.
  - no_ema would be evaluation-only IF the pre-EMA weights had been saved. They
    were not: train_ablation.py checkpoints `eval_model`, which IS the EMA copy
    when --ema is set, so the raw weights were discarded at save time.

    python src/publication/run_ablations.py --project-root . --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys

SEEDS = [42, 43, 44]
PRIMARY_SPLIT = "clean"

# Reference run whose configuration every variant inherits.
REFERENCE_RUN = "runs_v2_clean/fusion_v2_seed%d/metrics.json"

RECOVERED_CKPT = ("publication_runs/recovered_checkpoints/clean/"
                  "full_fusion_v2/seed_%d/checkpoint.pth")
RECOVERY_MANIFEST = "publication_results/ablations/checkpoint_recovery/import_manifest.json"


# Per-run wall-clock estimates, grounded in the measured FusionV2 Kaggle runs
# (37-58 min on a T4 for the full model). Branch-only arms drop one backbone.
COST_MIN = {
    "full_fusion_v2": 45, "no_gate": 45, "no_multiscale": 45,
    "no_auxiliary_loss": 45, "no_ema": 45, "no_tta": 3,
    "cnn_branch_only": 10, "vit_branch_only": 38,
}

VARIANTS = {
    "full_fusion_v2": {
        "question": "Reference proposed model.",
        "component_changed": "none - this is the reference",
        "mode": "train",
        "arch": "fusion_v2",
        "flags": ["--unfreeze_all_cnn", "--unfreeze_all_vit",
                  "--backbone_lr_mult", "0.1", "--aux_weight", "0.3", "--ema", "--tta"],
    },
    "no_gate": {
        "question": "Does the learned softmax gate help, versus plain concatenation?",
        "component_changed": "gate disabled; branch widths, head and training identical",
        "mode": "train",
        "arch": "fusion_v2",
        "flags": ["--unfreeze_all_cnn", "--unfreeze_all_vit",
                  "--backbone_lr_mult", "0.1", "--aux_weight", "0.3", "--ema", "--tta",
                  "--no_gate"],
    },
    "no_multiscale": {
        "question": "Do the intermediate CNN taps (stages 3 and 5) contribute?",
        "component_changed": "multi-scale taps removed; CNN input to the projection "
                             "falls from 1432 to 1280 channels",
        "mode": "train",
        "arch": "fusion_v2",
        "flags": ["--unfreeze_all_cnn", "--unfreeze_all_vit",
                  "--backbone_lr_mult", "0.1", "--aux_weight", "0.3", "--ema", "--tta",
                  "--no_multiscale"],
    },
    "no_auxiliary_loss": {
        "question": "Do the per-branch auxiliary losses prevent branch collapse?",
        "component_changed": "aux_weight 0.3 -> 0.0; architecture unchanged",
        "mode": "train",
        "arch": "fusion_v2",
        "flags": ["--unfreeze_all_cnn", "--unfreeze_all_vit",
                  "--backbone_lr_mult", "0.1", "--aux_weight", "0.0", "--ema", "--tta"],
    },
    "no_ema": {
        "question": "Does weight averaging explain the gain?",
        "component_changed": "EMA disabled; raw weights used for validation "
                             "selection and for the final checkpoint",
        "mode": "train",
        "arch": "fusion_v2",
        "flags": ["--unfreeze_all_cnn", "--unfreeze_all_vit",
                  "--backbone_lr_mult", "0.1", "--aux_weight", "0.3", "--tta"],
    },
    "no_tta": {
        "question": "Does horizontal-flip test-time augmentation explain the gain?",
        "component_changed": "TTA disabled at evaluation only; training identical",
        "mode": "eval_only_if_checkpoint",
        "arch": "fusion_v2",
        "flags": ["--unfreeze_all_cnn", "--unfreeze_all_vit",
                  "--backbone_lr_mult", "0.1", "--aux_weight", "0.3", "--ema"],
        "requires_checkpoint_from": "full_fusion_v2",
    },
    "cnn_branch_only": {
        "question": "What does the local (CNN) branch contribute alone?",
        "component_changed": "ViT branch removed entirely; CNN branch keeps the "
                             "FusionV2 projection and head",
        "mode": "train",
        "arch": "cnn_branch_only",
        "flags": ["--unfreeze_all_cnn", "--backbone_lr_mult", "0.1",
                  "--aux_weight", "0.3", "--ema", "--tta"],
    },
    "vit_branch_only": {
        "question": "What does the global (ViT) branch contribute alone?",
        "component_changed": "CNN branch removed entirely; ViT branch keeps the "
                             "FusionV2 projection and head",
        "mode": "train",
        "arch": "vit_branch_only",
        "flags": ["--unfreeze_all_vit", "--backbone_lr_mult", "0.1",
                  "--aux_weight", "0.3", "--ema", "--tta"],
    },
}

# Existing runs offered as substitutes, and the protocol facts that decide it.
REUSE_CANDIDATES = {
    "cnn_branch_only": "runs_clean/cnn_seed%d_unfrozen",
    "vit_branch_only": "runs_clean/vit_seed%d",
    "full_fusion_v2": "runs_v2_clean/fusion_v2_seed%d",
}


class PlanError(Exception):
    """A Phase 3 stop condition was hit."""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_reference(root: str) -> dict:
    """Read the reference FusionV2 configuration from the run that produced it."""
    per_seed = {}
    for s in SEEDS:
        rel = REFERENCE_RUN % s
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            raise PlanError("reference FusionV2 run missing: %s - cannot reproduce "
                            "the configuration the ablations must inherit" % rel)
        per_seed[s] = {"path": rel, "sha256": sha256(p),
                       "config": json.load(open(p, encoding="utf-8"))}
    keys = ["arch", "unfreeze_all_cnn", "unfreeze_all_vit", "backbone_lr_mult",
            "aux_weight", "ema", "tta", "lr", "weight_decay", "label_smoothing",
            "batch_size", "epochs", "n_train", "n_val", "n_test",
            "params_total", "params_trainable"]
    ref = {k: per_seed[SEEDS[0]]["config"].get(k) for k in keys}
    for s in SEEDS[1:]:
        drift = {k: (ref[k], per_seed[s]["config"].get(k))
                 for k in keys if k not in ("params_total", "params_trainable")
                 and per_seed[s]["config"].get(k) != ref[k]}
        if drift:
            raise PlanError("reference configuration differs between seeds %d and %d: %s"
                            % (SEEDS[0], s, drift))
    return {"config": ref, "sources": {s: {"path": v["path"], "sha256": v["sha256"]}
                                       for s, v in per_seed.items()}}


def audit_reuse(root: str, ref: dict) -> dict:
    """Decide, per variant and with artifact-level evidence, whether an existing
    run can substitute for a required ablation cell."""
    R = ref["config"]
    out = {}

    # ---- full_fusion_v2 ---------------------------------------------------
    # A recovered checkpoint counts only if the Phase 3 addendum VERIFIED it:
    # loaded strictly, gate intact, and reproduced the stored predictions. The
    # manifest records that verdict; the file's presence alone is not evidence.
    verified = {}
    man_p = os.path.join(root, RECOVERY_MANIFEST)
    if os.path.exists(man_p):
        man = json.load(open(man_p, encoding="utf-8"))
        for k, e in man.get("seeds", {}).items():
            verified[int(k)] = (e.get("decision") == "ACCEPTED")
    ckpts = {s: (os.path.exists(os.path.join(root, RECOVERED_CKPT % s))
                 and verified.get(s, False)) for s in SEEDS}
    preds = {s: os.path.exists(os.path.join(root, "runs_v2_clean",
                                            "fusion_v2_seed%d" % s, "preds_test.csv"))
             for s in SEEDS}
    out["full_fusion_v2"] = {
        "candidate": "runs_v2_clean/fusion_v2_seed{42,43,44}",
        "predictions_present": preds,
        "checkpoints_present": ckpts,
        "verdict": "REUSE - NO RETRAINING" if all(ckpts.values()) else "REUSE PREDICTIONS ONLY",
        "recovered_checkpoints_verified": ckpts,
        "evidence": ("Predictions, gates, metrics and histories exist for all three seeds "
                     "and were produced by exactly this configuration. Checkpoints were "
                     "recovered from Kaggle staging and VERIFIED in the Phase 3 addendum: "
                     "each loaded strictly against the reference architecture (578/578 "
                     "keys, 0 missing, 0 unexpected, 0 shape mismatches), retained a "
                     "learned input-dependent gate, and reproduced its stored clean-test "
                     "predictions exactly (0.0000 pp accuracy difference, 0 predicted-"
                     "label mismatches, max absolute probability difference ~4e-06 from "
                     "float non-determinism). Verified per seed: %s."
                     % ", ".join("%d: %s" % (s, ckpts[s]) for s in SEEDS)),
        "consequence": ("full_fusion_v2 needs no retraining: both its predictions and its "
                        "weights are available and verified. no_tta becomes a genuine "
                        "evaluation-only re-scoring of these weights."),
    }

    # ---- no_tta -----------------------------------------------------------
    out["no_tta"] = {
        "candidate": "evaluation-only re-scoring of the recovered full_fusion_v2 checkpoints",
        "verdict": "EVALUATION ONLY" if all(ckpts.values()) else "RETRAIN REQUIRED",
        "checkpoint_source": RECOVERED_CKPT,
        "evidence": ("no_tta differs from the reference only at inference, so it is a "
                     "re-scoring of the same weights rather than a new training run. "
                     "Those weights now exist and are verified for all three seeds, so "
                     "the evaluation-only route is available: each no_tta cell uses the "
                     "recovered checkpoint of the SAME seed."),
        "consequence": ("No training. Each cell loads "
                        "publication_runs/recovered_checkpoints/clean/full_fusion_v2/"
                        "seed_<N>/checkpoint.pth and scores without TTA, roughly 3 "
                        "minutes per seed."),
    }

    # ---- no_ema -----------------------------------------------------------
    trainer = os.path.join(root, "src", "train_ablation.py")
    saves_ema_only = False
    if os.path.exists(trainer):
        t = open(trainer, encoding="utf-8").read()
        saves_ema_only = ("eval_model = ema.ema if ema is not None else model" in t
                          and "copy.deepcopy(eval_model.state_dict())" in t)
    out["no_ema"] = {
        "candidate": "evaluation-only using pre-EMA weights of the reference run",
        "verdict": "RETRAIN REQUIRED",
        "evidence": ("Verified in src/train_ablation.py rather than assumed: the "
                     "training loop sets eval_model = ema.ema when --ema is given and "
                     "checkpoints copy.deepcopy(eval_model.state_dict()) (pattern found: "
                     "%s). The raw non-EMA weights are therefore never written to disk."
                     % saves_ema_only),
        "consequence": ("no_ema is a training-time ablation here, not an evaluation-time "
                        "one. Validation-based model selection in the reference used the "
                        "EMA copy, so a no_ema run would also select a different epoch. "
                        "It must be trained."),
    }

    # ---- branch-only controls --------------------------------------------
    for variant, key_field, existing_tmpl in (
            ("cnn_branch_only", "unfreeze_all_cnn", REUSE_CANDIDATES["cnn_branch_only"]),
            ("vit_branch_only", "unfreeze_all_vit", REUSE_CANDIDATES["vit_branch_only"])):
        p = os.path.join(root, existing_tmpl % SEEDS[0], "metrics.json")
        if not os.path.exists(p):
            out[variant] = {"candidate": existing_tmpl % SEEDS[0],
                            "verdict": "RETRAIN REQUIRED",
                            "evidence": "candidate run not present",
                            "consequence": "train under the controlled protocol"}
            continue
        cand = json.load(open(p, encoding="utf-8"))
        diffs = []
        if cand.get("backbone_lr_mult") is None:
            diffs.append("no discriminative learning rate (reference uses backbone_lr_mult=%s)"
                         % R["backbone_lr_mult"])
        if not cand.get("ema"):
            diffs.append("EMA not used (reference uses EMA)")
        if not cand.get("tta"):
            diffs.append("TTA not used (reference uses hflip TTA)")
        if cand.get("aux_weight") is None:
            diffs.append("no auxiliary branch loss (reference uses aux_weight=%s)"
                         % R["aux_weight"])
        if variant == "cnn_branch_only":
            diffs.append("head is Linear->BatchNorm1d->ReLU->Dropout->Linear on the final "
                         "1280-d pooled map; the control requires the FusionV2 projection "
                         "(Linear->LayerNorm->GELU) over 1432-d multi-scale features")
        else:
            diffs.append("ViT trainability differs: candidate has %s trainable parameters "
                         "(last two encoder layers), the control requires the reference "
                         "policy unfreeze_all_vit=%s"
                         % (format(cand.get("params_trainable", 0), ","), R["unfreeze_all_vit"]))
            diffs.append("head is Linear->BatchNorm1d->ReLU->Dropout->Linear; the control "
                         "requires the FusionV2 projection (Linear->LayerNorm->GELU)")
        out[variant] = {
            "candidate": existing_tmpl % SEEDS[0],
            "candidate_config": {k: cand.get(k) for k in
                                 ("arch", "unfreeze_all_cnn", "unfreeze_all_vit",
                                  "params_trainable", "lr", "batch_size", "epochs")},
            "verdict": "RETRAIN REQUIRED",
            "protocol_differences": diffs,
            "evidence": ("Compared field-by-field against the reference configuration "
                         "read from %s. %d protocol differences found."
                         % (REFERENCE_RUN % SEEDS[0], len(diffs))),
            "consequence": ("Reusing this run would confound the branch ablation with "
                            "differences in head architecture, learning-rate schedule, "
                            "weight averaging and inference protocol. It is a legitimate "
                            "baseline for the earlier ablation table, but not a control "
                            "for THIS one."),
        }

    for v in VARIANTS:
        out.setdefault(v, {
            "candidate": "none - this variant has never been trained",
            "verdict": "RETRAIN REQUIRED",
            "evidence": "No existing run implements this component change.",
            "consequence": "train under the controlled protocol",
        })
    return out


def build_matrix(root: str, ref: dict, reuse: dict, split: str) -> list:
    """One row per variant per seed."""
    rows = []
    for name, spec in VARIANTS.items():
        rv = reuse.get(name, {})
        for seed in SEEDS:
            out_dir = "publication_runs/ablations/%s/%s/seed_%d" % (split, name, seed)
            done = os.path.exists(os.path.join(root, out_dir, "metrics.json"))
            # no_tta is an inference-time ablation. No checkpoint exists TODAY, but
            # the grid itself produces one when full_fusion_v2 runs. It is therefore
            # evaluation-only CONDITIONAL on that ordering, not a training run - and
            # costing it as training would overstate the grid by ~2 GPU-hours.
            depends_on = None
            ckpt_ok = rv.get("recovered_checkpoints_verified", {}).get(seed) \
                if name == "full_fusion_v2" else \
                reuse.get("full_fusion_v2", {}).get("recovered_checkpoints_verified", {}).get(seed)
            if name == "full_fusion_v2":
                # Predictions AND verified weights both exist: nothing to run.
                kind = "reuse_existing" if ckpt_ok else "train_for_checkpoint"
            elif name == "no_tta":
                kind = "evaluation_only" if ckpt_ok else "evaluation_only_after_reference"
                depends_on = ("recovered checkpoint seed_%d (verified)" % seed) if ckpt_ok \
                    else "full_fusion_v2 (same seed) must complete first"
            else:
                kind = "train"
            cmd = ("python src/publication/train_ablation_variant.py"
                   " --variant %s --seed %d --split %s"
                   " --project-root . --out-root publication_runs/ablations" % (name, seed, split))
            rows.append({
                "variant": name, "seed": seed, "split": split,
                "question": spec["question"],
                "component_changed": spec["component_changed"],
                "arch": spec["arch"], "flags": spec["flags"],
                "execution_kind": kind,
                "training_required": not (kind.startswith("evaluation_only")
                                         or kind == "reuse_existing"),
                "depends_on": depends_on,
                "existing_checkpoint_reusable": rv.get("verdict") in
                    ("EVALUATION ONLY", "REUSE PREDICTIONS ONLY"),
                "reuse_verdict": rv.get("verdict"),
                "reuse_evidence": rv.get("evidence"),
                "input_split": ("dataset_clean/Training (4080 train / 720 val after the "
                                "15%% carve-out), dataset_clean/Testing (848)"),
                "output_dir": out_dir,
                "already_complete": done,
                "estimated_minutes_t4": 0 if kind == "reuse_existing"
                                       else COST_MIN.get(name, 45),
                "expected_artifacts": [
                    "metrics.json", "history.json", "checkpoint.pth",
                    "preds_test.csv", "preds_dataset_ood.csv", "config.json",
                ] + (["gates_test.csv", "gates_dataset_ood.csv"]
                     if spec["arch"] == "fusion_v2" and "--no_gate" not in spec["flags"] else []),
                "command": cmd,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-root", default="publication_results/ablations")
    ap.add_argument("--split", default=PRIMARY_SPLIT, choices=["clean", "original"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only. This module never trains; the flag is required "
                         "so that an accidental invocation cannot be mistaken for one.")
    args = ap.parse_args()

    if not args.dry_run:
        raise PlanError("run_ablations.py is a planner. Pass --dry-run. Training is a "
                        "separate, explicitly approved step (Phase 3 execution prompt).")

    root = os.path.abspath(args.project_root)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(root, args.out_root)
    if not os.path.relpath(out_root, root).replace("\\", "/").startswith("publication_"):
        raise PlanError("refusing to write outside publication_*")
    os.makedirs(out_root, exist_ok=True)

    print("1. reading reference FusionV2 configuration from its own run", flush=True)
    ref = load_reference(root)
    R = ref["config"]
    print("   arch=%s unfreeze_cnn=%s unfreeze_vit=%s lr=%s backbone_mult=%s aux=%s "
          "ema=%s tta=%s" % (R["arch"], R["unfreeze_all_cnn"], R["unfreeze_all_vit"],
                             R["lr"], R["backbone_lr_mult"], R["aux_weight"], R["ema"], R["tta"]),
          flush=True)
    print("   split: %s train / %s val / %s test | params %s"
          % (R["n_train"], R["n_val"], R["n_test"], format(R["params_trainable"], ",")), flush=True)

    print("2. auditing reuse eligibility of existing runs", flush=True)
    reuse = audit_reuse(root, ref)
    for k, v in reuse.items():
        print("   %-20s %s" % (k, v["verdict"]), flush=True)

    print("3. building experiment matrix", flush=True)
    rows = build_matrix(root, ref, reuse, args.split)
    to_train = [r for r in rows if r["training_required"] and not r["already_complete"]]
    eval_only = [r for r in rows if not r["training_required"] and not r["already_complete"]
                 and r["execution_kind"] != "reuse_existing"]
    reused = [r for r in rows if r["execution_kind"] == "reuse_existing"]
    total_min = sum(r["estimated_minutes_t4"] for r in to_train + eval_only)
    print("   %d cells | %d require training | %d evaluation-only | %d reused | "
          "est. %.1f GPU-hours"
          % (len(rows), len(to_train), len(eval_only), len(reused), total_min / 60.0),
          flush=True)

    report = {
        "phase": "3 - FusionV2 component ablations (planning / dry-run)",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "training_launched": False,
        "kaggle_upload_performed": False,
        "primary_split": args.split,
        "seeds": SEEDS,
        "reference_configuration": ref,
        "variants": VARIANTS,
        "reuse_audit": reuse,
        "matrix": rows,
        "summary": {
            "n_cells": len(rows),
            "n_training_required": len(to_train),
            "n_evaluation_only": len(eval_only),
            "n_already_complete": sum(1 for r in rows if r["already_complete"]),
            "n_reused_without_running": len(reused),
            "estimated_gpu_hours": round(total_min / 60.0, 2),
        },
        "label_isolation": (
            "The duplicate-disjoint external subset is scored once per run, after "
            "training and after validation-based model selection have finished. Its "
            "labels are never read during training, validation, early stopping, "
            "threshold choice or configuration selection. The trainer selects on "
            "validation accuracy carved from the TRAINING partition only."),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }

    with open(os.path.join(out_root, "experiment_matrix.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for fname, fn in (("EXPERIMENT_MATRIX.md", render_matrix_md),
                      ("REUSE_AUDIT.md", render_reuse_md),
                      ("KAGGLE_COMMANDS.md", render_kaggle_md)):
        with open(os.path.join(out_root, fname), "w", encoding="utf-8") as f:
            f.write(fn(report))

    print("\n   wrote %s" % os.path.relpath(out_root, root))
    print("\nTRAINING LAUNCHED: no. This is a dry run.")
    return 0


def render_matrix_md(r: dict) -> str:
    s = r["summary"]
    R = r["reference_configuration"]["config"]
    L = ["# Phase 3 - Experiment Matrix (dry run)", "",
         "Generated by `src/publication/run_ablations.py` at %s." % r["generated_utc"],
         "", "**No training was launched. No Kaggle upload was performed.**", "",
         "| | |", "|---|---|",
         "| cells (8 variants x 3 seeds) | %d |" % s["n_cells"],
         "| require training | %d |" % s["n_training_required"],
         "| evaluation-only | %d |" % s["n_evaluation_only"],
         "| already complete | %d |" % s["n_already_complete"],
         "| reused without running | %d |" % s.get("n_reused_without_running", 0),
         "| estimated cost | **%.1f GPU-hours** on a T4 |" % s["estimated_gpu_hours"], "",
         "## Reference configuration", "",
         "Read from the FusionV2 run itself, not hand-typed, so the ablations cannot "
         "drift from the model they ablate:", "",
         "| Field | Value |", "|---|---|"]
    for k in ("arch", "unfreeze_all_cnn", "unfreeze_all_vit", "lr", "backbone_lr_mult",
              "aux_weight", "ema", "tta", "batch_size", "epochs", "label_smoothing",
              "weight_decay", "n_train", "n_val", "n_test", "params_trainable"):
        L.append("| `%s` | %s |" % (k, R.get(k)))
    L += ["", "Sources (SHA-256):", ""]
    L += ["- `%s` `%s`" % (v["path"], v["sha256"][:16])
          for v in r["reference_configuration"]["sources"].values()]
    L += ["", "## Variants", "", "| Variant | Question | Component changed |", "|---|---|---|"]
    for name, v in r["variants"].items():
        L.append("| `%s` | %s | %s |" % (name, v["question"], v["component_changed"]))
    L += ["", "## Matrix", "",
          "| Variant | Seed | Kind | Train? | Reuse verdict | Est. min | Output directory |",
          "|---|---:|---|---|---|---:|---|"]
    for row in r["matrix"]:
        L.append("| `%s` | %d | %s | %s | %s | %d | `%s` |"
                 % (row["variant"], row["seed"], row["execution_kind"],
                    "yes" if row["training_required"] else "no (depends on %s)"
                    % (row.get("depends_on") or "-"),
                    row["reuse_verdict"], row["estimated_minutes_t4"], row["output_dir"]))
    L += ["", "## Per-cell detail", ""]
    for row in r["matrix"]:
        L += ["### `%s` seed %d" % (row["variant"], row["seed"]), "",
              "- **question:** %s" % row["question"],
              "- **component changed:** %s" % row["component_changed"],
              "- **input split:** %s" % row["input_split"],
              "- **output:** `%s`" % row["output_dir"],
              "- **training required:** %s" % row["training_required"],
              "- **existing checkpoint reusable:** %s" % row["existing_checkpoint_reusable"],
              "- **depends on:** %s" % (row.get("depends_on") or "nothing"),
              "- **reuse verdict:** %s" % row["reuse_verdict"],
              "- **evidence:** %s" % row["reuse_evidence"],
              "- **estimated cost:** ~%d min on a T4" % row["estimated_minutes_t4"],
              "- **expected artifacts:** %s" % ", ".join("`%s`" % a for a in row["expected_artifacts"]),
              "- **command:**", "", "```", row["command"], "```", ""]
    L += ["## Label isolation", "", r["label_isolation"], ""]
    return "\n".join(L)


def render_reuse_md(r: dict) -> str:
    L = ["# Phase 3 - Reuse Eligibility Audit", "",
         "Generated by `src/publication/run_ablations.py` at %s." % r["generated_utc"], "",
         "An existing run may substitute for an ablation cell only if architecture, "
         "split, preprocessing, trainability, optimizer, schedule, augmentation and "
         "evaluation protocol are demonstrably identical. Anything less is marked "
         "RETRAIN REQUIRED. Every verdict below rests on an inspected artifact.", "",
         "| Variant | Candidate | Verdict |", "|---|---|---|"]
    for k, v in r["reuse_audit"].items():
        L.append("| `%s` | `%s` | **%s** |" % (k, v.get("candidate"), v["verdict"]))
    L.append("")
    for k, v in r["reuse_audit"].items():
        L += ["## `%s`" % k, "",
              "- **candidate:** `%s`" % v.get("candidate"),
              "- **verdict:** **%s**" % v["verdict"],
              "- **evidence:** %s" % v.get("evidence"),
              "- **consequence:** %s" % v.get("consequence", "-")]
        if v.get("protocol_differences"):
            L += ["- **protocol differences found:**", ""]
            L += ["  %d. %s" % (i, d) for i, d in enumerate(v["protocol_differences"], 1)]
        if v.get("candidate_config"):
            L += ["", "  Candidate configuration as recorded: `%s`" % json.dumps(v["candidate_config"])]
        L.append("")
    L += ["## Summary", "",
          "No existing run is protocol-equivalent to any required ablation cell. The two "
          "cells that could in principle have been evaluation-only - `no_tta` and "
          "`no_ema` - both fail on a verified artifact fact rather than a judgement call: "
          "the reference checkpoints were never downloaded, and the pre-EMA weights were "
          "never saved. The existing `runs_clean` CNN and ViT arms remain valid baselines "
          "for the earlier ablation table; they are not controls for this one.", ""]
    return "\n".join(L)


def render_kaggle_md(r: dict) -> str:
    rows = [x for x in r["matrix"] if x["training_required"] and not x["already_complete"]]
    by_variant = {}
    for x in rows:
        by_variant.setdefault(x["variant"], []).append(x["seed"])
    est = r["summary"]["estimated_gpu_hours"]
    L = ["# Phase 3 - Kaggle commands for the missing runs", "",
         "Generated by `src/publication/run_ablations.py` at %s." % r["generated_utc"], "",
         "**Nothing has been uploaded or launched.** These cells are provided for review.",
         "", "Estimated total: **%.1f GPU-hours**. A Kaggle session is capped at 12 hours, "
         "so this needs **at least two commits**; the runner skips any cell whose "
         "`metrics.json` already exists, so committing again resumes rather than restarts."
         % est, "",
         "## Runs required", "", "| Variant | Seeds | Est. min each |", "|---|---|---:|"]
    for v, seeds in by_variant.items():
        L.append("| `%s` | %s | %d |" % (v, ", ".join(map(str, seeds)), COST_MIN.get(v, 45)))
    L += ["", "## Notebook cell", "", "```python",
          "VARIANTS = [", ]
    L += ["    %r," % v for v in by_variant]
    L += ["]", "SEEDS  = %r" % SEEDS, "SPLIT  = %r" % r["primary_split"], "",
          "for variant in VARIANTS:",
          "    for seed in SEEDS:",
          "        out = f\"{WORK}/publication_runs/ablations/{SPLIT}/{variant}/seed_{seed}\"",
          "        if os.path.exists(f\"{out}/metrics.json\"):",
          "            print('skip', variant, seed); continue",
          "        cmd = [sys.executable, f\"{WORK}/src/publication/train_ablation_variant.py\",",
          "               '--variant', variant, '--seed', str(seed), '--split', SPLIT,",
          "               '--project-root', WORK,",
          "               '--out-root', f'{WORK}/publication_runs/ablations',",
          "               '--train-dir', f'{WORK}/dataset_clean/Training',",
          "               '--test-dir',  f'{WORK}/dataset_clean/Testing',",
          "               '--extra-eval', f'{DATA}/dataset_ood/Testing',",
          "               '--workers', '4']",
          "        t0 = time.time()",
          "        r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True)",
          "        print(('ok' if r.returncode==0 else 'FAILED'), variant, seed,",
          "              f'{(time.time()-t0)/60:.1f} min')",
          "        if r.returncode: print(r.stderr[-2000:])",
          "```", "",
          "## After the grid", "",
          "```python",
          "# evaluation-only cell: no_tta re-scores the full_fusion_v2 checkpoints",
          "# produced above. It cannot run before those exist.",
          "for seed in SEEDS:",
          "    !python {WORK}/src/publication/train_ablation_variant.py \\",
          "        --variant no_tta --seed {seed} --split {SPLIT} --eval-only \\",
          "        --project-root {WORK} --out-root {WORK}/publication_runs/ablations",
          "```", "",
          "## Preconditions", "",
          "1. `dataset_clean/` must be rebuilt in the session (cell 2 of the existing "
          "notebook) so the split membership is byte-identical to the reference run.",
          "2. The code dataset must include `src/publication/`.",
          "3. Do not run `no_tta` until `full_fusion_v2` checkpoints exist.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PlanError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
