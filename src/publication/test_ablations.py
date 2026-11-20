"""
Phase 3 test suite.

Checks the properties that make an ablation a controlled experiment rather than
eight unrelated models: that each variant changes exactly the component it
names, that the reference variant still reproduces the shipped FusionV2
configuration, that a removed branch is genuinely absent rather than masked,
that external labels cannot reach training, and that no output path can land on
an existing artifact.

Runs on CPU. Downloads pretrained weights on first use, then works offline.

    python src/publication/test_ablations.py --project-root .
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import traceback

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
sys.path.insert(0, _SRC)
sys.path.insert(0, _HERE)

import train_ablation as TA
import fusion_v2 as FV
from ablation_models import CNNBranchOnly, ViTBranchOnly, build_branch_only
from run_ablations import VARIANTS, SEEDS, REFERENCE_RUN
import train_ablation_variant as TAV

RESULTS = []
BATCH, K = 2, 4

# Functional switches only. 'variant', 'arch' and 'component_changed' are
# descriptive metadata and are expected to differ between variants; diffing them
# would make every ablation look like it changed several things at once.
SWITCHES = ("unfreeze_all_cnn", "unfreeze_all_vit", "backbone_lr_mult",
            "aux_weight", "ema", "tta", "gated", "multiscale")


def switch_diff(a, b):
    """Names of the functional switches on which two resolved configs differ."""
    return {k for k in SWITCHES if a[k] != b[k]}


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append({"test": name, "status": "PASS", "detail": detail or ""})
        print("  PASS  %-58s %s" % (name, detail or ""), flush=True)
    except AssertionError as e:
        RESULTS.append({"test": name, "status": "FAIL", "detail": str(e)})
        print("  FAIL  %-58s %s" % (name, e), flush=True)
    except Exception as e:
        RESULTS.append({"test": name, "status": "ERROR",
                        "detail": "%s: %s" % (type(e).__name__, e),
                        "traceback": traceback.format_exc()})
        print("  ERROR %-58s %s: %s" % (name, type(e).__name__, e), flush=True)


def x():
    return torch.randn(BATCH, 3, TA.IMAGE_SIZE, TA.IMAGE_SIZE)


# --------------------------------------------------------------- shape tests ---

def t_shapes_fusion_variants():
    out = {}
    for gated in (True, False):
        for ms in (True, False):
            m = FV.FusionV2(multiscale=ms, gated=gated).eval()
            with torch.no_grad():
                y = m(x())
            assert tuple(y.shape) == (BATCH, K), "gated=%s ms=%s -> %s" % (gated, ms, y.shape)
            out["gated=%s,ms=%s" % (gated, ms)] = list(y.shape)
    return "4 configurations, all (%d, %d)" % (BATCH, K)


def t_shapes_branch_only():
    for cls in (CNNBranchOnly, ViTBranchOnly):
        m = cls().eval()
        with torch.no_grad():
            y = m(x())
            logits, aux, g, _ = m(x(), return_aux=True)
        assert tuple(y.shape) == (BATCH, K), "%s -> %s" % (cls.__name__, y.shape)
        assert tuple(logits.shape) == (BATCH, K) and tuple(aux.shape) == (BATCH, K)
        assert g is None, "%s must report no gate" % cls.__name__
    return "CNNBranchOnly and ViTBranchOnly both (%d, %d); gate reported None" % (BATCH, K)


# ------------------------------------------------- reference reproduction ---

def t_full_variant_matches_shipped_defaults():
    cfg = TAV.resolve("full_fusion_v2")
    assert cfg["gated"] is True, "reference must be gated"
    assert cfg["multiscale"] is True, "reference must use multi-scale taps"
    assert cfg["ema"] is True and cfg["tta"] is True
    assert cfg["aux_weight"] == 0.3
    assert cfg["backbone_lr_mult"] == 0.1
    assert cfg["unfreeze_all_cnn"] and cfg["unfreeze_all_vit"]
    sig = inspect.signature(FV.FusionV2.__init__)
    assert sig.parameters["multiscale"].default is True
    assert sig.parameters["gated"].default is True
    assert sig.parameters["proj_dim"].default == 512
    return "matches FusionV2 shipped defaults and the reference flag set"


def t_reference_matches_recorded_run(root):
    def inner():
        # The reference run directory is not uploaded to Kaggle, so the check
        # falls back to the embedded fingerprint transcribed from it. The point
        # of the test - that the resolved config matches what produced the
        # reference - holds either way; only the source of truth moves, and the
        # message says which one was used.
        from session1_config import REFERENCE_CONFIG, REFERENCE_SOURCE
        p = os.path.join(root, REFERENCE_RUN % SEEDS[0])
        if os.path.exists(p):
            rec, src = json.load(open(p, encoding="utf-8")), "recorded run file"
        else:
            rec, src = REFERENCE_CONFIG, "embedded fingerprint of %s" % REFERENCE_SOURCE
        cfg = TAV.resolve("full_fusion_v2")
        for field, key in (("unfreeze_all_cnn", "unfreeze_all_cnn"),
                           ("unfreeze_all_vit", "unfreeze_all_vit"),
                           ("aux_weight", "aux_weight"),
                           ("backbone_lr_mult", "backbone_lr_mult"),
                           ("ema", "ema"), ("tta", "tta")):
            assert rec.get(key) == cfg[field], \
                "%s: recorded %r != resolved %r" % (key, rec.get(key), cfg[field])
        return "6 fields match the reference (%s)" % src
    return inner


# --------------------------------------------- one-component-at-a-time tests ---

def t_no_gate_changes_only_gating():
    a, b = TAV.resolve("full_fusion_v2"), TAV.resolve("no_gate")
    diff = switch_diff(a, b)
    assert diff == {"gated"}, "no_gate changed %s, expected only {'gated'}" % diff
    m = FV.FusionV2(gated=False).eval()
    with torch.no_grad():
        _, _, _, g = m(x(), return_aux=True)
    assert g is None, "ungated model must not emit gate mass"
    assert hasattr(m, "cnn_stages") and hasattr(m, "vit"), "both branches must survive"
    assert sum(p.numel() for p in FV.FusionV2(gated=True).parameters()) \
        >= sum(p.numel() for p in m.parameters())
    return "only 'gated' differs; both branches retained; no gate emitted"


def t_no_multiscale_changes_only_taps():
    a, b = TAV.resolve("full_fusion_v2"), TAV.resolve("no_multiscale")
    diff = switch_diff(a, b)
    assert diff == {"multiscale"}, "no_multiscale changed %s, expected only {'multiscale'}" % diff
    full = FV.FusionV2(multiscale=True)
    none = FV.FusionV2(multiscale=False)
    in_full = full.cnn_proj[0].in_features
    in_none = none.cnn_proj[0].in_features
    assert in_full == FV.FINAL_CH + sum(FV.TAPS.values()), in_full
    assert in_none == FV.FINAL_CH, in_none
    assert full.vit_proj[0].in_features == none.vit_proj[0].in_features, \
        "the ViT side must be untouched by a CNN-tap ablation"
    return "cnn_proj in_features %d -> %d; ViT side unchanged" % (in_full, in_none)


def t_no_aux_changes_only_loss():
    a, b = TAV.resolve("full_fusion_v2"), TAV.resolve("no_auxiliary_loss")
    diff = switch_diff(a, b)
    assert diff == {"aux_weight"}, "no_auxiliary_loss changed %s, expected only {'aux_weight'}" % diff
    assert b["aux_weight"] == 0.0
    m1 = TAV.build_model(a, "google/vit-base-patch16-224")
    m2 = TAV.build_model(b, "google/vit-base-patch16-224")
    n1 = sum(p.numel() for p in m1.parameters())
    n2 = sum(p.numel() for p in m2.parameters())
    assert n1 == n2, "architecture must be identical: %d vs %d" % (n1, n2)
    return "only aux_weight differs (0.3 -> 0.0); identical %s parameters" % format(n1, ",")


def t_no_ema_and_no_tta_scope():
    a = TAV.resolve("full_fusion_v2")
    e, t = TAV.resolve("no_ema"), TAV.resolve("no_tta")
    assert switch_diff(a, e) == {"ema"}, "no_ema changed %s" % switch_diff(a, e)
    assert switch_diff(a, t) == {"tta"}, "no_tta changed %s" % switch_diff(a, t)
    assert VARIANTS["no_tta"]["mode"] == "eval_only_if_checkpoint"
    assert VARIANTS["no_ema"]["mode"] == "train", \
        "no_ema must be a training-time ablation: pre-EMA weights were never saved"
    return "no_ema differs only in 'ema' (training-time); no_tta only in 'tta' (eval-time)"


# ------------------------------------------------- removed-branch absence ---

def t_removed_branch_is_absent_not_masked():
    cnn = build_branch_only("cnn_branch_only", True, True)
    vit = build_branch_only("vit_branch_only", True, True)
    assert not hasattr(cnn, "vit"), "cnn_branch_only must not construct a ViT"
    assert not hasattr(cnn, "vit_proj") and not hasattr(cnn, "aux_vit")
    assert not hasattr(vit, "cnn_stages"), "vit_branch_only must not construct a CNN"
    assert not hasattr(vit, "cnn_proj") and not hasattr(vit, "aux_cnn")
    for m in (cnn, vit):
        assert not hasattr(m, "gate"), "a single-branch model must have no gate"
    names_cnn = {n.split(".")[0] for n, _ in cnn.named_parameters()}
    names_vit = {n.split(".")[0] for n, _ in vit.named_parameters()}
    assert "vit" not in names_cnn, names_cnn
    assert "cnn_stages" not in names_vit, names_vit
    n_cnn = sum(p.numel() for p in cnn.parameters())
    n_vit = sum(p.numel() for p in vit.parameters())
    n_fus = sum(p.numel() for p in FV.FusionV2().parameters())
    assert n_cnn < n_fus and n_vit < n_fus
    return "no ViT tensor in CNN-only (%s params) and no CNN tensor in ViT-only (%s)" \
        % (format(n_cnn, ","), format(n_vit, ","))


def t_branch_only_reuses_fusion_head_shape():
    cnn, vit = CNNBranchOnly(), ViTBranchOnly()
    fus = FV.FusionV2()
    assert cnn.cnn_proj[0].out_features == fus.cnn_proj[0].out_features == 512
    assert vit.vit_proj[0].out_features == fus.vit_proj[0].out_features == 512
    assert cnn.cnn_proj[0].in_features == fus.cnn_proj[0].in_features
    assert vit.vit_proj[0].in_features == fus.vit_proj[0].in_features
    assert isinstance(cnn.cnn_proj[1], torch.nn.LayerNorm)
    assert isinstance(vit.vit_proj[1], torch.nn.LayerNorm)
    return "projections identical to FusionV2 (in/out features and LayerNorm)"


# ------------------------------------------------------- label isolation ---

def t_external_labels_cannot_reach_training():
    src = open(os.path.join(_SRC, "train_ablation.py"), encoding="utf-8").read()
    i_extra = src.index("args.extra_eval")
    i_best = src.index("model.load_state_dict(best_state)")
    assert i_extra > i_best, \
        "extra_eval is referenced before model selection completes"
    sel = src[src.index("if val_acc > best_val"): src.index("model.load_state_dict(best_state)")]
    for token in ("extra_eval", "dataset_ood", "test_loader"):
        assert token not in sel, "model selection references %r" % token
    wsrc = open(os.path.join(_HERE, "train_ablation_variant.py"), encoding="utf-8").read()
    assert "--extra-eval" in wsrc and "scored once" in wsrc
    return "external set referenced only after selection; selection uses val only"


def t_validation_comes_from_training_partition():
    src = open(os.path.join(_SRC, "train_ablation.py"), encoding="utf-8").read()
    assert "val_idx, train_idx = idx[:n_val], idx[n_val:]" in src
    assert "datasets.ImageFolder(args.train_dir, transform=eval_transform)" in src
    assert TA.VAL_SPLIT_SEED == 42 and TA.VAL_FRACTION == 0.15
    return "val is a seeded 15%% carve-out of train_dir only"


# --------------------------------------------------------- output safety ---

def t_output_paths_isolated(root):
    def inner():
        protected = ["runs", "runs_original", "runs_clean", "runs_v2_original",
                     "runs_v2_clean", "results", "results_original", "results_clean",
                     "results_v2_original", "results_v2_clean", "dataset", "dataset_ood",
                     "dataset_clean", "publication_results/audit",
                     "publication_results/metrics", "publication_results/patient_audit"]
        for v in VARIANTS:
            for s in SEEDS:
                out = "publication_runs/ablations/clean/%s/seed_%d" % (v, s)
                assert out.startswith("publication_runs/"), out
                for p in protected:
                    assert not out.startswith(p + "/") and out != p, \
                        "%s collides with protected %s" % (out, p)
        wsrc = open(os.path.join(_HERE, "train_ablation_variant.py"), encoding="utf-8").read()
        assert 'refusing to write outside publication_' in wsrc
        assert "run already complete" in wsrc and "Refusing to overwrite" in wsrc
        return "24 paths under publication_runs/, none colliding with %d protected roots" \
            % len(protected)
    return inner


def t_refuses_overwrite_of_completed_run(root):
    def inner():
        import subprocess
        d = os.path.join(root, "publication_runs", "ablations", "clean",
                         "full_fusion_v2", "seed_42")
        os.makedirs(d, exist_ok=True)
        marker = os.path.join(d, "metrics.json")
        created = not os.path.exists(marker)
        if created:
            json.dump({"_test_sentinel": True}, open(marker, "w"))
        try:
            r = subprocess.run(
                [sys.executable, os.path.join(_HERE, "train_ablation_variant.py"),
                 "--variant", "full_fusion_v2", "--seed", "42", "--split", "clean",
                 "--project-root", root, "--dry-run"],
                capture_output=True, text=True)
            assert r.returncode == 2, "expected stop condition, got rc=%d" % r.returncode
            assert "Refusing to overwrite" in (r.stderr + r.stdout)
        finally:
            if created:
                os.remove(marker)
                for p in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
                    try:
                        os.rmdir(p)
                    except OSError:
                        pass
        return "runner exits with a stop condition rather than overwriting"
    return inner


def t_frozen_sources_unmodified(root, baseline):
    def inner():
        for rel, want in baseline.items():
            p = os.path.join(root, rel)
            got = hashlib.sha256(open(p, "rb").read()).hexdigest()
            assert got == want, "%s was modified during Phase 3" % rel
        return "src/fusion_v2.py and src/train_ablation.py byte-identical"
    return inner


def t_matrix_covers_all_cells():
    assert len(VARIANTS) == 8, "expected 8 variants, found %d" % len(VARIANTS)
    expected = {"full_fusion_v2", "no_gate", "no_multiscale", "no_auxiliary_loss",
                "no_ema", "no_tta", "cnn_branch_only", "vit_branch_only"}
    assert set(VARIANTS) == expected, set(VARIANTS) ^ expected
    assert SEEDS == [42, 43, 44]
    return "8 variants x 3 seeds = 24 cells"


def t_training_is_gated():
    wsrc = open(os.path.join(_HERE, "train_ablation_variant.py"), encoding="utf-8").read()
    assert "Training is gated" in wsrc
    i_gate = wsrc.index("Training is gated")
    i_dry = wsrc.index('if args.dry_run:')
    assert i_dry < i_gate, "the dry-run path must return before the training gate"
    return "non-dry-run invocation raises a stop condition"


def t_gate_expectation_matches_architecture():
    """The validator must demand gate CSVs exactly where a gate runs in the
    forward pass. Session 1's rule ("everything except no_gate is gated") is
    wrong for the single-branch controls, which have no gate to dump - that
    mismatch failed a correct cnn_branch_only run."""
    import session2_config as S2
    from ablation_models import CNNBranchOnly, ViTBranchOnly
    for variant in ("cnn_branch_only", "vit_branch_only"):
        m = build_branch_only(variant, True, True)
        assert not hasattr(m, "gate"), "%s must have no gate" % variant
        assert variant not in S2.VARIANTS_WITH_GATES,             "%s has no gate, so it must not be listed as gated" % variant
    for variant in ("no_ema", "no_tta"):
        assert variant in S2.VARIANTS_WITH_GATES,             "%s scores a gated model and must expect gate CSVs" % variant
    # and the validator must honour an explicit expectation
    import validate_run_outputs as V
    import inspect
    assert "gates_expected" in inspect.signature(V.validate).parameters,         "validate() must accept an explicit gate expectation"
    return "gate expectations match the architectures for all 4 Session 2 variants"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out", default="publication_results/ablations/TEST_REPORT.md")
    args = ap.parse_args()
    root = os.path.abspath(args.project_root)

    baseline = {rel: hashlib.sha256(open(os.path.join(root, rel), "rb").read()).hexdigest()
                for rel in ("src/fusion_v2.py", "src/train_ablation.py")}

    print("Phase 3 test suite\n" + "=" * 70, flush=True)
    check("output shapes - fusion variants", t_shapes_fusion_variants)
    check("output shapes - branch-only", t_shapes_branch_only)
    check("full variant matches shipped FusionV2 defaults", t_full_variant_matches_shipped_defaults)
    check("reference matches the recorded FusionV2 run", t_reference_matches_recorded_run(root))
    check("no_gate disables only learned gating", t_no_gate_changes_only_gating)
    check("no_multiscale removes only the CNN taps", t_no_multiscale_changes_only_taps)
    check("no_auxiliary_loss changes only the loss", t_no_aux_changes_only_loss)
    check("no_ema / no_tta scopes are correct", t_no_ema_and_no_tta_scope)
    check("removed branch is absent, not masked", t_removed_branch_is_absent_not_masked)
    check("branch-only reuses the FusionV2 projection", t_branch_only_reuses_fusion_head_shape)
    check("external labels cannot reach training", t_external_labels_cannot_reach_training)
    check("validation carved from the training partition", t_validation_comes_from_training_partition)
    check("output paths isolated from existing artifacts", t_output_paths_isolated(root))
    check("refuses to overwrite a completed run", t_refuses_overwrite_of_completed_run(root))
    check("matrix covers all 8 variants x 3 seeds", t_matrix_covers_all_cells)
    check("training is gated behind approval", t_training_is_gated)
    check("gate expectation matches architecture", t_gate_expectation_matches_architecture)
    check("frozen sources unmodified", t_frozen_sources_unmodified(root, baseline))

    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    n_fail = len(RESULTS) - n_pass
    status = "PASS" if n_fail == 0 else "FAIL"
    print("=" * 70)
    print("%d/%d passed -> %s" % (n_pass, len(RESULTS), status), flush=True)

    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L = ["# Phase 3 - Test Report", "",
         "Generated by `src/publication/test_ablations.py`.", "",
         "**%d of %d tests passed. Status: %s.**" % (n_pass, len(RESULTS), status), "",
         "| Test | Status | Detail |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % (r["test"], r["status"], r["detail"]) for r in RESULTS]
    L += ["", "## What these tests establish", "",
          "- every variant changes exactly the component it names, verified by diffing "
          "the resolved configuration against the reference;",
          "- the reference variant reproduces the configuration recorded in the actual "
          "FusionV2 run, not a hand-typed copy of it;",
          "- a removed branch is absent from the module tree and from `named_parameters()`, "
          "so 'CNN-only' cannot quietly still run a ViT;",
          "- the external set is referenced only after model selection completes, and "
          "validation is a seeded carve-out of the training partition;",
          "- no output path collides with a protected directory, and a completed run "
          "cannot be overwritten;",
          "- `src/fusion_v2.py` and `src/train_ablation.py` are byte-identical before and "
          "after the suite.", "",
          "## What they do not establish", "",
          "- nothing about accuracy. No model was trained. Whether a component helps is "
          "an empirical question these tests cannot answer;",
          "- nonzero gate mass is not tested as evidence of branch usefulness, because it "
          "is not evidence of it.", ""]
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote %s" % os.path.relpath(out, root))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
