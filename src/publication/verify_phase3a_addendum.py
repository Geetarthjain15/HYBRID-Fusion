"""
Phase 3A Addendum - provenance, alignment and configuration verification.

Phase 3A established that the archives were authentic and that each run met its
artifact contract. Three gaps remained, and this closes them:

  1. THE REFERENCE RUNS WERE NEVER RE-VALIDATED HERE. They were verified during
     checkpoint recovery, but Phase 3A took them on trust. They are re-derived
     from their own prediction files.

  2. ALIGNMENT WAS PER-FILE, NOT CANONICAL. Each run was checked internally
     consistent; no check proved that all 24 runs scored THE SAME 848 images
     with the same labels. Two runs could each be self-consistent and still be
     comparing different things. A canonical mapping is built from
     dataset_clean/split_manifest.json and every run must match its fingerprint.

  3. CONFIGURATION WAS COMPARED ON HAND-PICKED FIELDS. Every protocol fact is
     now resolved onto a canonical key from whichever schema path recorded it -
     the reference trainer writes these flat, the variant trainer nests them
     under resolved_config - and the comparison happens on those canonical keys.
     Unrecognised top-level fields are surfaced rather than ignored, so a
     difference cannot hide in a field nobody thought to name.

Two facts about the available evidence, established by inspection and reported
rather than worked around:

  - No run stores a macro-F1. Macro-F1 is recomputed for every run, and the
    absence of a stored counterpart is stated explicitly instead of being
    described as a reproduction.
  - The reference runs hold no checkpoint.pth; their weights are the separately
    verified recovered checkpoints. That linkage is recorded, not reported as a
    missing artifact.

Read-only with respect to every existing artifact.

    python src/publication/verify_phase3a_addendum.py --project-root .
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from session_artifact_contract import (CLASSES, CLEAN_TEST_N, EXTERNAL_N,
                                       PROB_SUM_TOL, METRIC_TOL_PP, design)

SEEDS = [42, 43, 44]
REFERENCE_DIR = "runs_v2_clean/fusion_v2_seed%d"
IMPORT_ROOT = "publication_runs/ablations/clean"
PHASE3A = "publication_results/ablations/phase3a"
SPLIT_MANIFEST = "dataset_clean/split_manifest.json"
RECOVERED_MANIFEST = "publication_results/ablations/session2/checkpoint_manifest.json"
RECOVERED_CKPT = "publication_runs/recovered_checkpoints/clean/full_fusion_v2/seed_%d/checkpoint.pth"

# Verified reference summary that must be reproduced (CLAUDE.md ground truth).
REF_ACC = (98.27, 0.07)
REF_F1 = (98.28, 0.04)

TRAINING_VARIANTS = ["no_gate", "no_multiscale", "no_auxiliary_loss",
                     "no_ema", "cnn_branch_only", "vit_branch_only"]
EVAL_VARIANTS = ["no_tta"]


class AddendumError(Exception):
    """A Phase 3A Addendum stop condition was hit."""


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _rel(path: str) -> str:
    """Relative path when that is meaningful, absolute otherwise. On Windows a
    path on another drive has no relative form, which is a ValueError, not a
    reason to fail verification."""
    try:
        return os.path.relpath(path).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def banner(m):
    print("\n" + "=" * 74 + "\n  %s\n" % m + "=" * 74, flush=True)


def stamp(m):
    print("[%s] %s" % (_dt.datetime.now().strftime("%H:%M:%S"), m), flush=True)


# ------------------------------------------------------- canonical mapping ---

def canonical_mapping(root: str) -> dict:
    """Authoritative filename -> true label for the 848 clean-test images.

    The split manifest is the preferred source: machine-readable, carrying
    image-level identity and label, and independent of any model output.
    dataset_clean/Testing holds no images locally, so directory enumeration is
    unavailable and is reported as such rather than substituted."""
    p = os.path.join(root, SPLIT_MANIFEST)
    if not os.path.exists(p):
        raise AddendumError("split manifest absent: %s" % SPLIT_MANIFEST)
    entries = json.load(open(p, encoding="utf-8"))
    test = [e for e in entries if e.get("new_split") == "Testing"]
    mapping = {}
    for e in test:
        fn = e["file"]
        if fn in mapping and mapping[fn] != e["class"]:
            raise AddendumError("split manifest gives two labels for %s" % fn)
        mapping[fn] = e["class"]
    if len(mapping) != CLEAN_TEST_N:
        raise AddendumError("canonical mapping has %d unique identities, expected %d"
                            % (len(mapping), CLEAN_TEST_N))
    bad = sorted(set(mapping.values()) - set(CLASSES))
    if bad:
        raise AddendumError("canonical mapping has labels outside the class set: %s" % bad)
    ordered = sorted(mapping.items())
    fingerprint = hashlib.sha256(
        "\n".join("%s\t%s" % (f, l) for f, l in ordered).encode()).hexdigest()
    from collections import Counter
    return {"source": SPLIT_MANIFEST, "source_sha256": sha256(p),
            "n_identities": len(mapping), "mapping": mapping,
            "fingerprint_sha256": fingerprint,
            "class_counts": dict(Counter(mapping.values())),
            "evidence_note": ("derived from the split manifest (preferred source 1); "
                              "dataset_clean/Testing contains no images locally, so "
                              "directory enumeration was unavailable")}


def fingerprint_of(files, labels) -> str:
    return hashlib.sha256(
        "\n".join("%s\t%s" % (f, l) for f, l in sorted(zip(files, labels))).encode()
    ).hexdigest()


# ------------------------------------------------------------- predictions ---

def load_predictions(path: str, expect_n: int) -> dict:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise AddendumError("%s: empty" % path)
    prob_cols = [c for c in rows[0] if c.startswith("p_")]
    classes = [c[2:] for c in prob_cols]
    if classes != sorted(classes):
        raise AddendumError("%s: probability columns not in sorted class order: %s"
                            % (path, classes))
    if classes != CLASSES:
        raise AddendumError("%s: class set %s != documented %s" % (path, classes, CLASSES))
    files = [r["filename"] for r in rows]
    seen, dupes = set(), set()
    for f_ in files:
        (dupes.add(f_) if f_ in seen else seen.add(f_))
    if dupes:
        raise AddendumError("%s: duplicate identities: %s" % (path, sorted(dupes)[:5]))
    if len(rows) != expect_n:
        raise AddendumError("%s: %d rows, expected %d" % (path, len(rows), expect_n))
    P = np.array([[float(r[c]) for c in prob_cols] for r in rows], dtype=np.float64)
    if not np.isfinite(P).all():
        raise AddendumError("%s: non-finite probabilities" % path)
    if P.min() < 0.0 or P.max() > 1.0:
        raise AddendumError("%s: probabilities outside [0,1] (%.6f..%.6f)"
                            % (path, P.min(), P.max()))
    dev = float(np.abs(P.sum(1) - 1.0).max())
    if dev > PROB_SUM_TOL:
        raise AddendumError("%s: rows deviate from sum 1 by %.2e" % (path, dev))
    y = np.array([classes.index(r["true_label"]) for r in rows])
    pred = np.array([classes.index(r["pred_label"]) for r in rows])
    strict = 0
    for i in np.nonzero(pred != P.argmax(1))[0]:
        t = np.sort(P[i])[-2:]
        if t[1] - t[0] > 1e-9:
            strict += 1
    if strict:
        raise AddendumError("%s: pred_label != argmax on %d strictly ranked rows"
                            % (path, strict))
    return {"files": files, "true_labels": [r["true_label"] for r in rows],
            "y": y, "pred": pred, "P": P, "classes": classes, "n": len(rows),
            "max_prob_sum_deviation": dev}


def accuracy_macro_f1(y, pred, k):
    acc = float((pred == y).mean())
    f1s = []
    for c in range(k):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return acc, float(np.mean(f1s))


def align_to_canonical(pred_data: dict, canon: dict) -> dict:
    """Match by stable identity, never by row position."""
    got = set(pred_data["files"])
    want = set(canon["mapping"])
    missing = sorted(want - got)
    extra = sorted(got - want)
    label_mismatch = [f for f, l in zip(pred_data["files"], pred_data["true_labels"])
                      if f in canon["mapping"] and canon["mapping"][f] != l]
    fp = fingerprint_of(pred_data["files"], pred_data["true_labels"])
    return {"n_identities": len(got), "n_missing": len(missing), "n_extra": len(extra),
            "missing_examples": missing[:5], "extra_examples": extra[:5],
            "n_true_label_mismatch": len(label_mismatch),
            "label_mismatch_examples": label_mismatch[:5],
            "fingerprint_sha256": fp,
            "matches_canonical": (not missing and not extra and not label_mismatch
                                  and fp == canon["fingerprint_sha256"])}


# ---------------------------------------------------------------------------
# Canonical protocol comparison.
#
# The reference and the ablations were written by two trainers with different
# config SCHEMAS: train_ablation.py records protocol fields flat at top level,
# train_ablation_variant.py nests them under resolved_config. A naive recursive
# diff reports every such field twice - once "absent from the run", once
# "absent from the reference" - and buries the real signal.
#
# So each protocol fact is resolved from whichever path the schema used, and the
# comparison happens on canonical keys. Where neither config records a field,
# the value is taken from the shared module constant in train_ablation.py, which
# BOTH trainers import - that is evidence, not an assumption, and the provenance
# is recorded per field.
# ---------------------------------------------------------------------------

# canonical key -> candidate paths, in priority order
PROTOCOL_SPEC = {
    "arch":              ["resolved_config.arch", "arch"],
    "lr":                ["lr"],
    "backbone_lr_mult":  ["resolved_config.backbone_lr_mult", "backbone_lr_mult"],
    "aux_weight":        ["resolved_config.aux_weight", "aux_weight"],
    "ema":               ["resolved_config.ema", "ema"],
    "tta":               ["resolved_config.tta", "tta"],
    "gated":             ["resolved_config.gated", "gated"],
    "multiscale":        ["resolved_config.multiscale", "multiscale"],
    "unfreeze_all_cnn":  ["resolved_config.unfreeze_all_cnn", "unfreeze_all_cnn"],
    "unfreeze_all_vit":  ["resolved_config.unfreeze_all_vit", "unfreeze_all_vit"],
    "batch_size":        ["batch_size"],
    "epochs":            ["epochs"],
    "weight_decay":      ["weight_decay"],
    "label_smoothing":   ["label_smoothing"],
    "patience":          ["patience"],
    "val_fraction":      ["val_fraction"],
    "val_split_seed":    ["val_split_seed"],
}

# Fields that are OUTCOMES or run identity, never protocol. Excluded by name so
# the exclusion list is auditable rather than implicit.
NON_PROTOCOL = {
    "seed", "variant", "run", "output_dir", "out_root", "argv", "component_changed",
    "question", "train_dir", "test_dir", "external_eval_dir", "vit_source",
    "device", "seeding", "env", "environment", "total_minutes", "epochs_run",
    "test_acc", "best_val_acc", "per_class_test_acc", "final_train_acc",
    "overfitting", "batches_per_epoch", "note", "label_isolation", "phase",
    "generated_utc", "mode", "trained", "weights_updated", "checkpoint_written",
    "source_checkpoint", "source_checkpoint_sha256", "architecture_loaded",
    "tta_used", "params_total", "params_trainable", "n_train", "n_val", "n_test",
    "split", "resolved_config", "extra_eval_acc", "history", "metrics",
}

# ---------------------------------------------------------------------------
# Resolving protocol values the reference run did not record.
#
# Three evidence classes are used, and every resolved value carries the class it
# came from so a reader can weigh it:
#
#   recorded_in_config        the value is in the run's own JSON.
#   shared_module_literal     the value is a module-level constant in
#                             src/train_ablation.py, read as a literal from the
#                             source (that file imports torch, which is absent
#                             locally, so it is parsed rather than executed).
#                             Both trainers take their defaults from it.
#   derived_from_params_total the value is inferred from the parameter count the
#                             run itself recorded, using the recorded
#                             (params_total, config) pairs of the imported runs
#                             as a physical discriminator.
#
# A value that none of these three can supply is left UNRESOLVED and reported as
# a named comparability limitation. It is never filled in by assumption.
# ---------------------------------------------------------------------------

REFERENCE_TRAINER = "src/train_ablation.py"

# canonical key -> module-level constant name in the shared trainer
MODULE_CONSTANT = {
    "lr": "LR", "batch_size": "BATCH_SIZE", "epochs": "EPOCHS",
    "weight_decay": "WEIGHT_DECAY", "label_smoothing": "LABEL_SMOOTHING",
    "patience": "PATIENCE", "val_fraction": "VAL_FRACTION",
    "val_split_seed": "VAL_SPLIT_SEED",
}


def module_constants(root: str) -> dict:
    """Read module-level literal constants from the shared trainer by parsing it.

    Importing is not an option: train_ablation.py imports torch, which is not
    installed locally. Parsing reads the same literals the interpreter would
    bind, without running anything. The file's digest and each constant's line
    number are recorded so the reading is checkable."""
    import ast
    path = os.path.join(root, REFERENCE_TRAINER)
    if not os.path.exists(path):
        raise AddendumError("shared trainer absent: %s" % REFERENCE_TRAINER)
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in MODULE_CONSTANT.values():
                try:
                    found[t.id] = {"value": ast.literal_eval(node.value),
                                   "line": node.lineno}
                except ValueError:
                    pass
    missing = sorted(set(MODULE_CONSTANT.values()) - set(found))
    return {"source": REFERENCE_TRAINER, "source_sha256": sha256(path),
            "constants": found, "missing": missing,
            "note": ("parsed, not imported: %s imports torch, which is not "
                     "installed in this environment" % REFERENCE_TRAINER)}


def derive_reference_architecture(ref_cfg: dict, param_cohorts: dict) -> dict:
    """Infer architecture flags the reference did not record, from its own
    recorded params_total, using the imported runs as a discriminator table.

    Only a flag that actually changes the parameter count can be derived this
    way. Gating cannot: src/fusion_v2.py constructs self.gate unconditionally
    and only skips using it when gated is false, so a gated and an ungated
    FusionV2 have identical parameter counts. That is recorded as an
    unresolvable key rather than guessed."""
    out = {}
    ref_n = ref_cfg.get("params_total")
    ref_arch = ref_cfg.get("arch")
    for key, cohorts in param_cohorts.items():
        by_value = {}
        for value, counts in cohorts.items():
            by_value[value] = sorted(counts)
        # the reference must match exactly one cohort's parameter count
        hits = [v for v, counts in by_value.items() if ref_n in counts]
        misses = [v for v, counts in by_value.items() if ref_n not in counts]
        if len(hits) == 1 and misses:
            out[key] = {
                "value": hits[0],
                "evidence_class": "derived_from_params_total",
                "source": ("reference params_total=%s matches only the %s=%r "
                           "cohort among imported %s runs; the %s cohort(s) "
                           "record %s"
                           % (ref_n, key, hits[0], ref_arch,
                              ", ".join("%s=%r" % (key, v) for v in misses),
                              ", ".join(str(by_value[v]) for v in misses))),
                "reference_params_total": ref_n,
                "cohorts": {str(v): by_value[v] for v in by_value},
            }
        else:
            out[key] = {
                "value": None,
                "evidence_class": "unresolvable",
                "source": ("parameter count does not discriminate %s: "
                           "reference params_total=%s, cohorts %s"
                           % (key, ref_n,
                              {str(v): by_value[v] for v in by_value})),
                "cohorts": {str(v): by_value[v] for v in by_value},
            }
    return out


def reference_spec_agreement(ref_cfg: dict, expectation: dict) -> dict:
    """Check the reference run's RECORDED fields against the spec's reference
    expectation, and report how far that licenses using the spec for the two
    fields the reference did not record.

    This is the weakest evidence class used here and is labelled as such: the
    reference runs predate this pipeline and were launched by src/train_ablation.py
    directly, so the spec is a statement of intent, not a log of that invocation.
    It is admitted only for a key no stronger source can supply, and only when
    every field the run did record agrees with it."""
    checked, disagree = {}, []
    for k, want in expectation.items():
        if k in ref_cfg:
            got = ref_cfg[k]
            agrees = got == want
            checked[k] = {"spec": want, "recorded": got, "agrees": agrees}
            if not agrees:
                disagree.append(k)
    unrecorded = sorted(k for k in expectation if k not in ref_cfg)
    return {"source": "run_ablations.py :: VARIANTS['full_fusion_v2'].flags",
            "checked_fields": checked, "n_checked": len(checked),
            "disagreements": disagree, "unrecorded_fields": unrecorded,
            "usable": not disagree and bool(checked),
            "expectation": expectation}


def param_cohort_table(root: str, keys=("multiscale", "gated")) -> dict:
    """Recorded (params_total, flag) pairs from the imported fusion runs.

    Read straight from each run's metrics.json. Only runs whose architecture
    equals the reference architecture are usable as a discriminator, since a
    branch-only control has a different parameter count for reasons unrelated
    to these flags."""
    table = {k: {} for k in keys}
    for variant in TRAINING_VARIANTS + EVAL_VARIANTS:
        for s in SEEDS:
            mp = os.path.join(root, IMPORT_ROOT, variant, "seed_%d" % s, "metrics.json")
            if not os.path.exists(mp):
                continue
            m = json.load(open(mp, encoding="utf-8"))
            rc = m.get("resolved_config", {})
            if rc.get("arch") != "fusion_v2":
                continue
            n = m.get("params_total")
            for k in keys:
                if k in rc and n is not None:
                    table[k].setdefault(rc[k], set()).add(n)
    return table


def _dig(cfg: dict, path: str):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def extract_protocol(cfg: dict, resolvers: dict, is_reference: bool) -> dict:
    """Resolve every canonical protocol key, recording where each value came from.

    Order of preference: the run's own record, then the shared module literal,
    then - for the reference only - a value derived from its recorded parameter
    count. Anything left over stays None and is reported, not filled in."""
    consts = resolvers["module_constants"]["constants"]
    derived = resolvers["derived_reference"] if is_reference else {}
    spec = resolvers["reference_spec"]
    out = {}
    for key, paths in PROTOCOL_SPEC.items():
        for path in paths:
            val, found = _dig(cfg, path)
            if found:
                out[key] = {"value": val, "evidence_class": "recorded_in_config",
                            "source": path}
                break
        else:
            cname = MODULE_CONSTANT.get(key)
            if cname and cname in consts:
                out[key] = {"value": consts[cname]["value"],
                            "evidence_class": "shared_module_literal",
                            "source": "%s:%d  %s" % (REFERENCE_TRAINER,
                                                     consts[cname]["line"], cname)}
            elif key in derived and derived[key]["value"] is not None:
                out[key] = {"value": derived[key]["value"],
                            "evidence_class": derived[key]["evidence_class"],
                            "source": derived[key]["source"]}
            elif is_reference and spec.get("usable") and key in spec["expectation"]:
                out[key] = {
                    "value": spec["expectation"][key],
                    "evidence_class": "reference_spec_corroborated",
                    "source": ("%s; the reference run records no value for this key, "
                               "and all %d fields it does record agree with that spec"
                               % (spec["source"], spec["n_checked"])),
                    "weaker_evidence": True,
                    "unresolved_from_run_record": True,
                }
            elif key in derived:
                out[key] = {"value": None, "evidence_class": "unresolvable",
                            "source": derived[key]["source"]}
            else:
                out[key] = {"value": None, "evidence_class": "unresolvable",
                            "source": "not recorded in this configuration and no "
                                      "shared default or physical discriminator "
                                      "available"}
    return out


def unmapped_fields(cfg: dict) -> list:
    """Top-level keys that are neither protocol nor documented non-protocol.
    Surfaced so an unexpected field cannot pass unnoticed."""
    mapped = set()
    for paths in PROTOCOL_SPEC.values():
        mapped.update(pp.split(".")[0] for pp in paths)
    return sorted(k for k in cfg if k not in mapped and k not in NON_PROTOCOL)


def config_diff(run_cfg: dict, ref_cfg: dict, resolvers: dict) -> dict:
    a = extract_protocol(run_cfg, resolvers, is_reference=False)
    b = extract_protocol(ref_cfg, resolvers, is_reference=True)
    diffs, unresolved = {}, []
    for key in sorted(PROTOCOL_SPEC):
        av, bv = a[key]["value"], b[key]["value"]
        if av is None or bv is None:
            unresolved.append({"key": key,
                               "side": ("reference" if bv is None and av is not None
                                        else "run" if av is None and bv is not None
                                        else "both"),
                               "run_value": av, "reference_value": bv,
                               "run_source": a[key]["source"],
                               "reference_source": b[key]["source"]})
            continue
        if av != bv:
            diffs[key] = {"reference": bv, "run": av,
                          "reference_source": b[key]["source"],
                          "run_source": a[key]["source"],
                          "reference_evidence_class": b[key]["evidence_class"],
                          "run_evidence_class": a[key]["evidence_class"]}
    weak = sorted(k for k, v in b.items() if v.get("weaker_evidence"))
    return {"differences": diffs, "unresolved": unresolved,
            "reference_weak_evidence_keys": weak,
            "run_protocol": {k: v["value"] for k, v in a.items()},
            "reference_protocol": {k: v["value"] for k, v in b.items()},
            "reference_evidence": {k: {"evidence_class": v["evidence_class"],
                                       "source": v["source"]} for k, v in b.items()},
            "unmapped_run_fields": unmapped_fields(run_cfg)}


# ---------------------------------------------------------------------------
# What each ablation was SUPPOSED to change - read from the spec, not asserted.
#
# An earlier version of this file hand-wrote these sets and got vit_branch_only
# wrong: it assumed a ViT-only model must also drop multi-scale, when in fact
# --no_multiscale is never passed to it and the run correctly records
# multiscale=true (the flag concerns CNN taps, which that architecture has none
# of). The sets are now computed from VARIANTS in run_ablations.py - the spec
# the runs were actually launched from - by resolving each variant's flags and
# the reference's flags onto canonical keys and taking the difference.
# ---------------------------------------------------------------------------

def _flags_to_protocol(entry: dict) -> dict:
    """Mirror of resolve() in train_ablation_variant.py, on canonical keys.

    train_ablation_variant.resolve() cannot be imported here: that module pulls
    in FusionV2 and therefore torch, which is absent locally."""
    f = entry["flags"]
    def opt(name, default):
        return type(default)(f[f.index(name) + 1]) if name in f else default
    return {
        "arch": entry["arch"],
        "unfreeze_all_cnn": "--unfreeze_all_cnn" in f,
        "unfreeze_all_vit": "--unfreeze_all_vit" in f,
        "backbone_lr_mult": opt("--backbone_lr_mult", 0.1),
        "aux_weight": opt("--aux_weight", 0.0),
        "ema": "--ema" in f,
        "tta": "--tta" in f,
        "gated": "--no_gate" not in f,
        "multiscale": "--no_multiscale" not in f,
    }


def intended_from_spec() -> dict:
    """{variant: set(canonical keys the spec says it changes)} plus the resolved
    reference expectation, for corroboration."""
    import run_ablations as RA
    ref = _flags_to_protocol(RA.VARIANTS["full_fusion_v2"])
    out = {}
    for name, entry in RA.VARIANTS.items():
        if name == "full_fusion_v2":
            continue
        got = _flags_to_protocol(entry)
        out[name] = {k for k in ref if got[k] != ref[k]}
    return {"reference_expectation": ref, "intended": out,
            "source": "run_ablations.py :: VARIANTS"}


PROTOCOL_KEYS = ["lr", "batch_size", "epochs", "weight_decay", "label_smoothing",
                 "patience", "val_fraction", "val_split_seed", "backbone_lr_mult"]


def classify_diffs(variant: str, d: dict, spec_intended: dict) -> dict:
    """Separate three outcomes that must not be conflated:

      a REJECTION   - the run differs from the reference in something it was not
                      supposed to change, so the comparison is confounded;
      a LIMITATION  - a key could not be resolved on one side, so comparability
                      for that key is unknown and is named as such;
      ACCEPTANCE    - every intended change is present and nothing else differs.

    An intended change whose reference-side value is unresolvable is not counted
    as present and not counted as absent. It is reported as unconfirmable."""
    diffs = d["differences"]
    intended = spec_intended.get(variant, set())
    unresolved_keys = {u["key"] for u in d["unresolved"]}

    found = sorted(k for k in diffs if k in intended)
    unintended = sorted(k for k in diffs if k not in intended)
    violations = [k for k in unintended if k in PROTOCOL_KEYS]
    missing = sorted(intended - set(diffs) - unresolved_keys)
    unconfirmable = sorted(intended & unresolved_keys)

    # unresolved keys the variant was NOT meant to change: comparability unknown
    limits = sorted(unresolved_keys - intended)
    # keys resolved only from the weakest evidence class; relevant when the key
    # is one this variant's conclusion depends on
    weak = d["reference_weak_evidence_keys"]
    weak_relevant = sorted(set(weak) & (intended | set(diffs)))

    if unintended:
        verdict = "UNINTENDED DIFFERENCE"
    elif missing:
        verdict = "INTENDED DIFFERENCE ABSENT"
    elif unconfirmable:
        verdict = "INTENDED DIFFERENCE NOT CONFIRMABLE"
    elif weak_relevant:
        verdict = "OK ON WEAKEST EVIDENCE CLASS"
    elif limits:
        verdict = "OK WITH NAMED COMPARABILITY LIMITATION"
    else:
        verdict = "OK"

    return {"all_differences": diffs,
            "intended_found": found, "intended_expected": sorted(intended),
            "intended_missing": missing, "intended_unconfirmable": unconfirmable,
            "unintended": unintended, "protocol_violations": violations,
            "comparability_limitations": limits,
            "reference_weak_evidence_keys": weak,
            "weak_evidence_affecting_conclusion": weak_relevant,
            "unresolved_keys": d["unresolved"],
            "unmapped_run_fields": d["unmapped_run_fields"],
            "run_protocol": d["run_protocol"],
            "reference_protocol": d["reference_protocol"],
            "reference_evidence": d["reference_evidence"],
            "n_differences": len(diffs),
            # only a confounded comparison blocks acceptance; an honestly named
            # limitation does not, and is carried into the report instead
            "blocks_acceptance": bool(unintended or missing),
            "verdict": verdict}


# ------------------------------------------------------------------ driver ---

def verify_run(run_dir: str, variant: str, seed: int, category: str,
               canon: dict, ref_cfg: dict, recovered: dict,
               resolvers: dict) -> dict:
    r = {"variant": variant, "seed": seed, "category": category,
         "run_dir": _rel(run_dir),
         "files": {}, "checks": {}, "problems": []}

    for name in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            r["files"][name] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}

    met = json.load(open(os.path.join(run_dir, "metrics.json"), encoding="utf-8"))
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else met

    r["checks"]["seed_matches"] = met.get("seed") == seed
    r["checks"]["split_is_clean"] = (met.get("split") == "clean"
                                     or "dataset_clean" in str(met.get("test_dir", "")))

    # --- predictions and canonical alignment -------------------------------
    pt = load_predictions(os.path.join(run_dir, "preds_test.csv"), CLEAN_TEST_N)
    r["alignment_clean_test"] = align_to_canonical(pt, canon)
    r["checks"]["canonical_alignment"] = r["alignment_clean_test"]["matches_canonical"]

    ood_p = os.path.join(run_dir, "preds_dataset_ood.csv")
    if os.path.exists(ood_p):
        po = load_predictions(ood_p, EXTERNAL_N)
        r["external"] = {"n": po["n"], "structurally_valid": True,
                         "max_prob_sum_deviation": po["max_prob_sum_deviation"],
                         "note": ("validated for structure only; external labels are "
                                  "never used for tuning or model selection")}

    # --- metrics: recomputed, and named stored fields ----------------------
    acc, f1 = accuracy_macro_f1(pt["y"], pt["pred"], len(pt["classes"]))
    comp = []
    stored_acc_field = "test_acc" if "test_acc" in met else None
    if stored_acc_field is not None:
        d = abs(100 * acc - 100 * met[stored_acc_field])
        comp.append({"metric": "accuracy", "stored_field": stored_acc_field,
                     "stored_value": met[stored_acc_field], "recomputed": acc,
                     "abs_diff_pp": d, "within_tolerance": d <= METRIC_TOL_PP})
        r["checks"]["accuracy_reproduces"] = d <= METRIC_TOL_PP
    else:
        comp.append({"metric": "accuracy", "stored_field": None,
                     "stored_value": None, "recomputed": acc,
                     "note": "no stored accuracy field available for comparison"})
    f1_fields = [k for k in met if "macro" in k.lower() and "f1" in k.lower()]
    if f1_fields:
        sv = met[f1_fields[0]]
        d = abs(100 * f1 - 100 * sv)
        comp.append({"metric": "macro_f1", "stored_field": f1_fields[0],
                     "stored_value": sv, "recomputed": f1, "abs_diff_pp": d,
                     "within_tolerance": d <= METRIC_TOL_PP})
        r["checks"]["macro_f1_reproduces"] = d <= METRIC_TOL_PP
    else:
        comp.append({"metric": "macro_f1", "stored_field": None, "stored_value": None,
                     "recomputed": f1,
                     "note": "macro-F1 independently recomputed; no stored macro-F1 "
                             "field available for direct comparison"})
    r["metrics"] = {"recomputed_accuracy": acc, "recomputed_macro_f1": f1,
                    "comparisons": comp}

    # --- configuration diff ------------------------------------------------
    if category == "reference":
        r["config_diff"] = {"verdict": "OK",
                            "note": "this run IS the reference; no diff applies",
                            "all_differences": {}, "unintended": [],
                            "intended_missing": [], "intended_unconfirmable": [],
                            "comparability_limitations": [],
                            "reference_weak_evidence_keys": [],
                            "weak_evidence_affecting_conclusion": [], "n_differences": 0,
                            "unresolved_keys": [], "unmapped_run_fields": [],
                            "blocks_acceptance": False, "protocol_violations": [],
                            "intended_expected": [], "intended_found": []}
    else:
        d = config_diff(cfg, ref_cfg, resolvers)
        r["config_diff"] = classify_diffs(variant, d, resolvers["spec_intended"])
        r["checks"]["config_controlled"] = not r["config_diff"]["blocks_acceptance"]

    # --- no_tta provenance --------------------------------------------------
    if variant == "no_tta":
        recorded = open(os.path.join(run_dir, "source_checkpoint_sha256.txt"),
                        encoding="utf-8").read().split()[0].lower()
        expected = recovered.get(seed, {}).get("sha256")
        r["source_checkpoint"] = {"recorded": recorded, "expected": expected,
                                  "matches": recorded == expected}
        r["checks"]["source_checkpoint_matches"] = recorded == expected
        r["checks"]["did_not_train"] = met.get("trained") is False
        r["checks"]["no_checkpoint_written"] = met.get("checkpoint_written") is False

    if category == "reference":
        ck = os.path.normpath(os.path.join(os.getcwd(), RECOVERED_CKPT % seed))
        r["checkpoint_linkage"] = {
            "checkpoint_in_run_dir": os.path.exists(os.path.join(run_dir, "checkpoint.pth")),
            "verified_recovered_checkpoint": RECOVERED_CKPT % seed,
            "recovered_present": os.path.exists(ck),
            "recovered_sha256": recovered.get(seed, {}).get("sha256"),
            "note": ("reference runs store no checkpoint.pth; their weights are the "
                     "separately verified recovered checkpoints")}

    failed = [k for k, v in r["checks"].items() if v is False]
    r["accepted"] = not failed and not r["problems"]
    r["failed_checks"] = failed
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-root", default="publication_results/ablations/phase3a_addendum")
    args = ap.parse_args()

    root = os.path.abspath(args.project_root)
    os.chdir(root)
    if not args.out_root.replace("\\", "/").startswith("publication_"):
        raise AddendumError("refusing to write outside publication_*")
    out_root = os.path.join(root, args.out_root)
    os.makedirs(out_root, exist_ok=True)

    rep = {"phase": "3A addendum - provenance, alignment, configuration",
           "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
           "training_performed": False, "reimport_performed": False,
           "runs": [], "limitations": []}

    recovered = {}
    rp = os.path.join(root, RECOVERED_MANIFEST)
    if os.path.exists(rp):
        for c in json.load(open(rp, encoding="utf-8"))["checkpoints"]:
            recovered[c["seed"]] = c

    banner("TASK 2 - CANONICAL IMAGE AND LABEL MAPPING")
    canon = canonical_mapping(root)
    rep["canonical_mapping"] = {k: v for k, v in canon.items() if k != "mapping"}
    stamp("source: %s (sha256 %s)" % (canon["source"], canon["source_sha256"][:16]))
    stamp("identities: %d | class counts: %s" % (canon["n_identities"], canon["class_counts"]))
    stamp("fingerprint: %s" % canon["fingerprint_sha256"])
    stamp(canon["evidence_note"])

    ref_cfg = json.load(open(os.path.join(root, REFERENCE_DIR % SEEDS[0], "metrics.json"),
                             encoding="utf-8"))
    consts = module_constants(root)
    if consts["missing"]:
        raise AddendumError("shared module constants not found in %s: %s"
                            % (REFERENCE_TRAINER, consts["missing"]))
    cohorts = param_cohort_table(root)
    derived = derive_reference_architecture(ref_cfg, cohorts)
    spec = intended_from_spec()
    agree = reference_spec_agreement(ref_cfg, spec["reference_expectation"])
    if agree["disagreements"]:
        raise AddendumError("the reference run disagrees with the reference spec on %s"
                            % agree["disagreements"])
    resolvers = {"module_constants": consts, "derived_reference": derived,
                 "spec_intended": spec["intended"], "reference_spec": agree}
    rep["intended_differences_from_spec"] = {"source": spec["source"],
        "reference_expectation": spec["reference_expectation"],
        "per_variant": {k: sorted(v) for k, v in spec["intended"].items()}}
    rep["reference_spec_agreement"] = agree
    rep["shared_module_constants"] = consts
    rep["derived_reference_architecture"] = derived
    stamp("shared module constants read from %s (sha256 %s)"
          % (consts["source"], consts["source_sha256"][:16]))
    for k, v in sorted(derived.items()):
        stamp("  reference %-11s = %r  [%s]" % (k, v["value"], v["evidence_class"]))
    stamp("reference spec agreement: %d/%d recorded fields agree, unrecorded %s"
          % (sum(1 for v in agree["checked_fields"].values() if v["agrees"]),
             agree["n_checked"], agree["unrecorded_fields"]))
    for k, v in sorted(spec["intended"].items()):
        stamp("  intended change  %-18s %s" % (k, sorted(v)))

    banner("TASK 1 - REVALIDATE THE THREE REFERENCE RUNS (read-only)")
    ref_results = []
    for s in SEEDS:
        d = os.path.join(root, REFERENCE_DIR % s)
        r = verify_run(d, "full_fusion_v2", s, "reference", canon, ref_cfg,
                       recovered, resolvers)
        ref_results.append(r)
        rep["runs"].append(r)
        stamp("  seed %d  %s  acc %.4f  macroF1 %.4f  aligned=%s"
              % (s, "ACCEPT" if r["accepted"] else "REJECT",
                 r["metrics"]["recomputed_accuracy"], r["metrics"]["recomputed_macro_f1"],
                 r["checks"].get("canonical_alignment")))

    accs = [r["metrics"]["recomputed_accuracy"] for r in ref_results]
    f1s = [r["metrics"]["recomputed_macro_f1"] for r in ref_results]
    summary = {
        "accuracy_mean_pct": 100 * float(np.mean(accs)),
        "accuracy_sd_pct": 100 * float(np.std(accs, ddof=1)),
        "macro_f1_mean_pct": 100 * float(np.mean(f1s)),
        "macro_f1_sd_pct": 100 * float(np.std(f1s, ddof=1)),
        "verified_accuracy": REF_ACC, "verified_macro_f1": REF_F1,
    }
    summary["accuracy_mean_diff_pp"] = abs(summary["accuracy_mean_pct"] - REF_ACC[0])
    summary["accuracy_sd_diff_pp"] = abs(summary["accuracy_sd_pct"] - REF_ACC[1])
    summary["macro_f1_mean_diff_pp"] = abs(summary["macro_f1_mean_pct"] - REF_F1[0])
    summary["macro_f1_sd_diff_pp"] = abs(summary["macro_f1_sd_pct"] - REF_F1[1])
    summary["reproduces"] = all(summary[k] <= METRIC_TOL_PP for k in
                                ("accuracy_mean_diff_pp", "accuracy_sd_diff_pp",
                                 "macro_f1_mean_diff_pp", "macro_f1_sd_diff_pp"))
    rep["reference_summary"] = summary
    stamp("  recomputed accuracy  %.2f +/- %.2f  (verified %.2f +/- %.2f)"
          % (summary["accuracy_mean_pct"], summary["accuracy_sd_pct"], *REF_ACC))
    stamp("  recomputed macro-F1  %.2f +/- %.2f  (verified %.2f +/- %.2f)"
          % (summary["macro_f1_mean_pct"], summary["macro_f1_sd_pct"], *REF_F1))
    stamp("  reproduces within %.2f pp: %s" % (METRIC_TOL_PP, summary["reproduces"]))

    fps = {r["alignment_clean_test"]["fingerprint_sha256"] for r in ref_results}
    rep["reference_seeds_share_mapping"] = len(fps) == 1
    stamp("  three reference seeds share one image/label mapping: %s" % (len(fps) == 1))

    banner("TASK 3/4 - IMPORTED RUNS: ALIGNMENT, CONFIG DIFF, METRICS")
    for variant in TRAINING_VARIANTS + EVAL_VARIANTS:
        cat = "evaluation_only" if variant in EVAL_VARIANTS else "training"
        for s in SEEDS:
            d = os.path.join(root, IMPORT_ROOT, variant, "seed_%d" % s)
            if not os.path.isdir(d):
                raise AddendumError("imported run absent: %s" % d)
            r = verify_run(d, variant, s, cat, canon, ref_cfg, recovered, resolvers)
            rep["runs"].append(r)
            cd = r["config_diff"]
            stamp("  %-18s seed %d  %s  align=%s  %s  (%d diffs, %d unintended, "
                  "%d limits)"
                  % (variant, s, "ACCEPT" if r["accepted"] else "REJECT",
                     r["checks"].get("canonical_alignment"), cd["verdict"],
                     cd["n_differences"], len(cd["unintended"]),
                     len(cd["comparability_limitations"]) + len(cd["intended_unconfirmable"])))
            if cd["unintended"]:
                for k in cd["unintended"][:6]:
                    stamp("        unintended: %s  ref=%r run=%r"
                          % (k, cd["all_differences"][k]["reference"],
                             cd["all_differences"][k]["run"]))

    banner("TASK 5 - IMPORT INTEGRITY (no re-import)")
    man = json.load(open(os.path.join(root, PHASE3A, "archive_verification.json"),
                         encoding="utf-8"))
    src_by_key = {(x["variant"], x["seed"]): x for x in man["runs"]}
    integ = {"checked": 0, "mismatched": [], "missing": [], "runs": []}
    for r in rep["runs"]:
        if r["category"] == "reference":
            continue
        src = src_by_key.get((r["variant"], r["seed"]))
        if src is None:
            integ["missing"].append("%s seed %d" % (r["variant"], r["seed"]))
            continue
        per = {"variant": r["variant"], "seed": r["seed"],
               "source_archive": src.get("source_archive"),
               "source_member_prefix": src.get("source_member_prefix"),
               "mode": src.get("mode"), "files": {}}
        for name, info in r["files"].items():
            recorded = src.get("files", {}).get(name, {}).get("sha256")
            integ["checked"] += 1
            ok = recorded is not None and recorded == info["sha256"]
            per["files"][name] = {"sha256": info["sha256"], "recorded": recorded,
                                  "match": ok}
            if not ok:
                integ["mismatched"].append("%s seed %d :: %s" % (r["variant"], r["seed"], name))
        integ["runs"].append(per)
    integ["all_match"] = not integ["mismatched"] and not integ["missing"]
    rep["import_integrity"] = integ
    stamp("  %d files re-hashed | mismatched %d | unmapped %d -> %s"
          % (integ["checked"], len(integ["mismatched"]), len(integ["missing"]),
             "PASS" if integ["all_match"] else "FAIL"))

    # ---- counts and verdict ----------------------------------------------
    cats = {}
    for r in rep["runs"]:
        cats.setdefault(r["category"], []).append(r)
    rep["counts"] = {k: len(v) for k, v in cats.items()}
    aligned = [r for r in rep["runs"] if r["checks"].get("canonical_alignment")]
    rep["all_runs_share_canonical_mapping"] = len(aligned) == len(rep["runs"])
    unintended = [r for r in rep["runs"] if r["config_diff"]["unintended"]]
    confounded = [r for r in rep["runs"] if r["config_diff"]["blocks_acceptance"]]
    limited = [{"variant": r["variant"], "seed": r["seed"],
                "comparability_limitations": r["config_diff"]["comparability_limitations"],
                "intended_unconfirmable": r["config_diff"]["intended_unconfirmable"],
                "weak_evidence_affecting_conclusion":
                    r["config_diff"]["weak_evidence_affecting_conclusion"]}
               for r in rep["runs"]
               if r["config_diff"]["comparability_limitations"]
               or r["config_diff"]["intended_unconfirmable"]
               or r["config_diff"]["weak_evidence_affecting_conclusion"]]
    rep["configuration_comparability_limitations"] = limited
    rejected = [r for r in rep["runs"] if not r["accepted"]]
    rep["n_rejected"] = len(rejected)

    rep["limitations"] = [
        "No run stores a macro-F1 value. Macro-F1 is independently recomputed for "
        "every run; there is no stored macro-F1 field available for direct "
        "comparison, so no claim of stored-macro-F1 reproduction is made.",
        "The reference runs contain no checkpoint.pth. Their weights are the "
        "separately verified recovered checkpoints, recorded per seed as a linkage "
        "rather than reported as a missing artifact.",
        "dataset_clean/Testing holds no images locally, so the canonical mapping "
        "was derived from the split manifest (preferred source 1) and could not be "
        "cross-checked against a directory enumeration.",
        "The external 499-image predictions were validated for structure only. "
        "External labels are not used for tuning or model selection.",
        "Phase 3A manifests were left exactly as issued, so this addendum is an "
        "additive record of what was checked later.",
        "The reference runs do not record whether the fusion gate was active. "
        "src/fusion_v2.py constructs the gate module unconditionally and only "
        "skips using it in forward(), so the recorded parameter count is identical "
        "with and without gating and cannot settle the question; src/train_ablation.py "
        "guards its gate dump on the architecture only, so the presence of gate "
        "telemetry in the reference runs does not settle it either. The reference "
        "value is therefore taken from the reference specification in "
        "run_ablations.py, corroborated by all seven reference fields that ARE "
        "recorded agreeing with that specification. This is the weakest evidence "
        "class used in this addendum and it is the basis on which the no_gate "
        "comparison rests.",
        "The reference runs record no patience, val_fraction or val_split_seed. "
        "Those values are read as module-level literals from src/train_ablation.py, "
        "whose digest is recorded; that file imports torch, which is not installed "
        "locally, so it was parsed rather than executed. This is code-level "
        "evidence, not a recorded value from the run.",
        "The reference multiscale setting is not recorded either. It is derived "
        "from the parameter count the reference itself recorded, which matches only "
        "the multiscale cohort of the imported fusion runs and differs from the "
        "non-multiscale cohort by 77,824 parameters.",
    ]

    ok = (rep["reference_seeds_share_mapping"] and summary["reproduces"]
          and rep["all_runs_share_canonical_mapping"] and integ["all_match"]
          and not confounded and not rejected
          and rep["counts"].get("training") == 18
          and rep["counts"].get("evaluation_only") == 3
          and rep["counts"].get("reference") == 3)
    rep["acceptance_status"] = "PASS" if ok else "FAIL"

    with open(os.path.join(out_root, "verification_addendum.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    with open(os.path.join(out_root, "VERIFICATION_ADDENDUM.md"), "w",
              encoding="utf-8") as f:
        f.write(render_md(rep))

    banner("PHASE 3A ADDENDUM: %s" % rep["acceptance_status"])
    stamp("counts: %s" % rep["counts"])
    stamp("all runs share the canonical mapping: %s" % rep["all_runs_share_canonical_mapping"])
    stamp("unintended configuration differences: %d" % len(unintended))
    stamp("runs carrying a named comparability limitation: %d" % len(limited))
    stamp("wrote %s" % os.path.relpath(out_root, root))
    return 0 if ok else 1


def render_md(r: dict) -> str:
    s = r.get("reference_summary", {})
    c = r["canonical_mapping"]
    L = ["# Phase 3A Addendum - Provenance, Alignment and Configuration", "",
         "Generated by `src/publication/verify_phase3a_addendum.py` at %s."
         % r["generated_utc"], "",
         "**Status: %s.** No training, no re-import, no existing artifact modified."
         % r["acceptance_status"], "",
         "## Run inventory", "", "| Category | Count |", "|---|---:|"]
    for k in ("training", "evaluation_only", "reference"):
        L.append("| %s | %d |" % (k, r["counts"].get(k, 0)))
    L += ["", "The three reference runs are **reused**, not newly trained ablations, "
              "and are counted separately throughout.", "",
          "## Canonical image/label mapping", "",
          "| Field | Value |", "|---|---|",
          "| source | `%s` |" % c["source"],
          "| source SHA-256 | `%s` |" % c["source_sha256"][:32],
          "| identities | %d |" % c["n_identities"],
          "| class counts | %s |" % c["class_counts"],
          "| **fingerprint** | `%s` |" % c["fingerprint_sha256"], "",
          c["evidence_note"], "",
          "Every accepted run must reproduce this fingerprint exactly. Matching is "
          "by stable image identity, never by row position: two runs could each be "
          "internally consistent and still be scoring different images.", "",
          "All runs share the canonical mapping: **%s**"
          % r["all_runs_share_canonical_mapping"], "",
          "## Reference reproduction", "", "| Metric | Recomputed | Verified | diff (pp) |",
          "|---|---|---|---:|"]
    if s:
        L += ["| accuracy | %.2f +/- %.2f | %.2f +/- %.2f | %.4f / %.4f |"
              % (s["accuracy_mean_pct"], s["accuracy_sd_pct"], s["verified_accuracy"][0],
                 s["verified_accuracy"][1], s["accuracy_mean_diff_pp"], s["accuracy_sd_diff_pp"]),
              "| macro-F1 | %.2f +/- %.2f | %.2f +/- %.2f | %.4f / %.4f |"
              % (s["macro_f1_mean_pct"], s["macro_f1_sd_pct"], s["verified_macro_f1"][0],
                 s["verified_macro_f1"][1], s["macro_f1_mean_diff_pp"], s["macro_f1_sd_diff_pp"]),
              "", "Reproduces within %.2f pp: **%s**. Three reference seeds share one "
                  "image/label mapping: **%s**."
              % (METRIC_TOL_PP, s["reproduces"], r["reference_seeds_share_mapping"]), ""]
    L += ["## Per-run verification", "",
          "| Category | Variant | Seed | Aligned | Config verdict | Diffs | Unintended | Recomputed acc | Recomputed macro-F1 |",
          "|---|---|---:|---|---|---:|---:|---:|---:|"]
    for run in r["runs"]:
        cd = run["config_diff"]
        L.append("| %s | `%s` | %d | %s | %s | %d | %d | %.4f | %.4f |"
                 % (run["category"], run["variant"], run["seed"],
                    run["checks"].get("canonical_alignment"), cd["verdict"],
                    cd["n_differences"], len(cd["unintended"]),
                    run["metrics"]["recomputed_accuracy"],
                    run["metrics"]["recomputed_macro_f1"]))
    L += ["", "## Configuration differences", "",
          "Each protocol fact is resolved onto a canonical key from whichever path "
          "the recording schema used: the reference trainer (`src/train_ablation.py`) "
          "writes them flat at top level, the variant trainer "
          "(`src/publication/train_ablation_variant.py`) nests them under "
          "`resolved_config`. Comparing raw nested dictionaries would report every "
          "such field twice and bury the real signal, so the comparison happens on "
          "canonical keys instead. Branch-only controls necessarily differ in which "
          "backbone exists; those are listed as intended architecture-dependent "
          "differences rather than silently tolerated.", ""]
    nonref = [x for x in r["runs"] if x["category"] != "reference"]
    if nonref:
        cd0 = nonref[0]["config_diff"]
        rp, ev = cd0["reference_protocol"], cd0["reference_evidence"]
        L += ["Every resolved reference value, with the class of evidence it rests "
              "on. `recorded_in_config` is a value the run itself wrote down; "
              "`shared_module_literal` is a constant parsed from the trainer both "
              "trainers import; `derived_from_params_total` is inferred from the "
              "parameter count the run recorded; `reference_spec_corroborated` is "
              "the weakest class, admitted only where nothing stronger exists.", "",
              "| Canonical key | Reference value | Evidence class | Source |",
              "|---|---|---|---|"]
        for k in sorted(rp):
            e = ev.get(k, {})
            L.append("| `%s` | `%r` | `%s` | %s |"
                     % (k, rp[k], e.get("evidence_class", "?"),
                        str(e.get("source", "")).replace("|", "/")[:170]))
        L.append("")
        unres = sorted({x["key"] for run in nonref
                        for x in run["config_diff"]["unresolved_keys"]})
        unmapped = sorted({k for run in nonref
                           for k in run["config_diff"]["unmapped_run_fields"]})
        weak = sorted({k for run in nonref
                       for k in run["config_diff"]["reference_weak_evidence_keys"]})
        L += ["Canonical keys that could not be resolved on both sides: %s"
              % (", ".join("`%s`" % k for k in unres) or "**none**"), "",
              "Canonical keys resolved only on the weakest evidence class: %s"
              % (", ".join("`%s`" % k for k in weak) or "**none**"), "",
              "Top-level configuration fields neither classified as protocol nor "
              "documented as run identity: %s"
              % (", ".join("`%s`" % k for k in unmapped) or "**none**"), ""]
        sp = r.get("intended_differences_from_spec", {})
        if sp:
            L += ["### Intended differences, read from the launch specification", "",
                  "Taken from `%s` rather than asserted here: each variant's flags "
                  "and the reference's flags are resolved onto canonical keys and "
                  "differenced. An earlier hand-written version of this table listed "
                  "`multiscale` for `vit_branch_only`, which is wrong - "
                  "`--no_multiscale` is never passed to it and the run correctly "
                  "records `multiscale=true`." % sp["source"], "",
                  "| Variant | Intended canonical changes |", "|---|---|"]
            for k, v in sorted(sp["per_variant"].items()):
                L.append("| `%s` | %s |" % (k, ", ".join("`%s`" % x for x in v) or "-"))
            L.append("")
        ag = r.get("reference_spec_agreement", {})
        if ag:
            L += ["### Reference-specification agreement", "",
                  "| Field | Specification | Recorded in reference run | Agrees |",
                  "|---|---|---|---|"]
            for k, v in sorted(ag["checked_fields"].items()):
                L.append("| `%s` | `%r` | `%r` | %s |"
                         % (k, v["spec"], v["recorded"], v["agrees"]))
            L += ["", "Fields the reference run does not record: %s. The reference "
                      "runs predate this pipeline and were launched from "
                      "`src/train_ablation.py` directly, so this specification is a "
                      "statement of intent, not a log of that invocation."
                  % (", ".join("`%s`" % k for k in ag["unrecorded_fields"]) or "none"), ""]
        lim = r.get("configuration_comparability_limitations", [])
        L += ["### Runs carrying a named configuration limitation", ""]
        if lim:
            L += ["| Variant | Seed | Unresolved but irrelevant | Intended change "
                  "unconfirmable | Weakest-class evidence the conclusion rests on |",
                  "|---|---:|---|---|---|"]
            for x in lim:
                L.append("| `%s` | %d | %s | %s | %s |"
                         % (x["variant"], x["seed"],
                            ", ".join("`%s`" % k for k in x["comparability_limitations"]) or "-",
                            ", ".join("`%s`" % k for k in x["intended_unconfirmable"]) or "-",
                            ", ".join("`%s`" % k for k in
                                      x["weak_evidence_affecting_conclusion"]) or "-"))
            L.append("")
        else:
            L += ["None.", ""]
    seen = set()
    for run in r["runs"]:
        if run["category"] == "reference" or run["variant"] in seen:
            continue
        seen.add(run["variant"])
        cd = run["config_diff"]
        L += ["### `%s`" % run["variant"], "",
              "- expected: %s" % (", ".join("`%s`" % k for k in cd["intended_expected"]) or "none"),
              "- found: %s" % (", ".join("`%s`" % k for k in cd["intended_found"]) or "none"),
              "- unintended: %s" % (", ".join("`%s`" % k for k in cd["unintended"]) or "**none**"),
              "- protocol violations: %s" % (", ".join(cd["protocol_violations"]) or "**none**"), ""]
        if cd["all_differences"]:
            L += ["| Field | Reference | Run |", "|---|---|---|"]
            for k, v in sorted(cd["all_differences"].items()):
                L.append("| `%s` | `%r` | `%r` |" % (k, v["reference"], v["run"]))
            L.append("")
    L += ["## Metric comparison", "",
          "| Variant | Seed | Metric | Stored field | Stored | Recomputed | diff (pp) |",
          "|---|---:|---|---|---:|---:|---:|"]
    for run in r["runs"]:
        for m in run["metrics"]["comparisons"]:
            L.append("| `%s` | %d | %s | %s | %s | %.6f | %s |"
                     % (run["variant"], run["seed"], m["metric"],
                        ("`%s`" % m["stored_field"]) if m["stored_field"] else "**none**",
                        ("%.6f" % m["stored_value"]) if m["stored_value"] is not None else "-",
                        m["recomputed"],
                        ("%.4f" % m["abs_diff_pp"]) if "abs_diff_pp" in m else "-"))
    i = r["import_integrity"]
    L += ["", "## Import integrity", "",
          "| Field | Value |", "|---|---|",
          "| files re-hashed | %d |" % i["checked"],
          "| mismatched | %d |" % len(i["mismatched"]),
          "| unmapped to a Phase 3A record | %d |" % len(i["missing"]),
          "| all match | **%s** |" % i["all_match"], ""]
    if i["mismatched"]:
        L += ["Mismatches:", ""] + ["- %s" % x for x in i["mismatched"][:20]] + [""]
    L += ["## Limitations and unavailable evidence", ""]
    L += ["%d. %s" % (n, x) for n, x in enumerate(r["limitations"], 1)]
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AddendumError as e:
        print("\nSTOP CONDITION: %s" % e, file=sys.stderr)
        sys.exit(2)
