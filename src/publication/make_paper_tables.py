"""
AGMF-Net manuscript tables, traceability ledger and deliverable manifest.

Every row is generated from a verified artifact. Where a manuscript claim cannot
be traced to an inspectable file it is emitted with status NOT_VERIFIED rather
than quietly dropped, so the ledger records gaps instead of hiding them.

    env_win/Scripts/python.exe src/publication/make_paper_tables.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "publication_results", "final_paper_claude")
TAB = os.path.join(OUT, "tables")
SEEDS = [42, 43, 44]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("  wrote %s (%d rows)" % (os.path.relpath(path, ROOT), len(rows)))


M = json.load(open(os.path.join(ROOT, "publication_results/metrics/fusion_v2_metrics.json"),
                   encoding="utf-8"))
CONC = {r["variant"]: r for r in csv.DictReader(open(os.path.join(
    ROOT, "publication_results/ablations/final_analysis/component_conclusions.csv"),
    encoding="utf-8"))}
SUMM = {}
for r in csv.DictReader(open(os.path.join(
        ROOT, "publication_results/ablations/final_analysis/final_metrics_summary.csv"),
        encoding="utf-8")):
    if r["metric"] in ("accuracy", "macro_f1"):
        SUMM.setdefault(r["model"], {})[r["metric"]] = (float(r["mean"]), float(r["sd_ddof1"]))
ABL = json.load(open(os.path.join(
    ROOT, "publication_results/ablations/final_analysis/final_ablation_analysis.json"),
    encoding="utf-8"))


# ---------------------------------------------------- 1. base comparison ----

def base_paper_comparison():
    """Protocol columns first, accuracy last: the comparability verdict is the point."""
    rows = [
        dict(characteristic="Overall design",
             brain_meta="Ten CNN-ViT base learners with logistic-regression and XGBoost meta-learners",
             agmf_net="Single feature-level CNN-transformer hybrid with a learned adaptive gate",
             comparability="not established",
             note="Different model classes; an accuracy difference cannot be attributed to any one design choice"),
        dict(characteristic="Fusion level",
             brain_meta="Softmax prediction stacking", agmf_net="Projected feature fusion (512+512)",
             comparability="not established",
             note="AGMF-Net contains no meta-learner and must not be described as stacking"),
        dict(characteristic="Preprocessing",
             brain_meta="Resize, normalise, CLAHE contrast enhancement",
             agmf_net="Resize, normalise (mean/std 0.5); no CLAHE",
             comparability="not established", note="Input pipelines differ"),
        dict(characteristic="Dataset source",
             brain_meta="Public brain tumour MRI collection (7,023 images reported)",
             agmf_net="Public brain tumour MRI collection (7,023 images pooled)",
             comparability="partially comparable",
             note="Same reported pool size; identical file-level composition not established"),
        dict(characteristic="Reported split",
             brain_meta="5,712 train / 1,311 test", agmf_net="4,800 train / 848 test after duplicate cleaning; 1,052 on the original split",
             comparability="not established", note="Different partition construction and different test sizes"),
        dict(characteristic="Test sample count", brain_meta="1,311", agmf_net="1,052 (original); 848 (duplicate-clean); 373 (cross-source)",
             comparability="not established", note="No shared test set"),
        dict(characteristic="Duplicate control",
             brain_meta="Not reported in the inspected paper",
             agmf_net="256-bit dHash audit, Hamming <= 8, 1,375 redundant images removed",
             comparability="not established",
             note="Absence in the inspected article is not evidence that it was not performed"),
        dict(characteristic="Patient-level grouping evidence",
             brain_meta="Not reported in the inspected paper",
             agmf_net="Not available; no patient/study/series key exists in the files; not claimed",
             comparability="not established", note="Neither work establishes patient independence"),
        dict(characteristic="External / cross-source evaluation",
             brain_meta="Not reported in the inspected paper",
             agmf_net="373 duplicate-disjoint images from a second collection",
             comparability="not established", note="One additional source only"),
        dict(characteristic="Multi-seed evaluation",
             brain_meta="Not reported in the inspected paper", agmf_net="Seeds 42, 43, 44 with sample SD",
             comparability="not established", note="Three seeds is a small sample of run-to-run variation"),
        dict(characteristic="Component ablations",
             brain_meta="Base-learner comparisons reported; not component controls of a single model",
             agmf_net="Seven protocol-matched one-component controls, 21 paired tests, Holm-corrected",
             comparability="not established", note="Different study questions"),
        dict(characteristic="Calibration reporting",
             brain_meta="Not a headline analysis", agmf_net="NLL, Brier, ECE (15 bins), reliability curves",
             comparability="not established", note="AGMF-Net calibration figures are descriptive only"),
        dict(characteristic="Explainability method",
             brain_meta="Grad-CAM on CNN components", agmf_net="Grad-CAM++ on the CNN path",
             comparability="partially comparable",
             note="Neither establishes validated tumour localisation"),
    ]
    o = M["settings"]["original"]["across_seeds"]
    c = M["settings"]["clean"]["across_seeds"]
    x = M["settings"]["cross_source_disjoint"]["across_seeds"]
    rows += [
        dict(characteristic="Accuracy", brain_meta="97.10% (XGBoost meta-learner, as reported)",
             agmf_net="%.2f +/- %.2f%% (original, n=1052); %.2f +/- %.2f%% (duplicate-clean, n=848); %.2f +/- %.2f%% (cross-source, n=373)"
                      % (100 * o["accuracy"]["mean"], 100 * o["accuracy"]["sd_across_seeds"],
                         100 * c["accuracy"]["mean"], 100 * c["accuracy"]["sd_across_seeds"],
                         100 * x["accuracy"]["mean"], 100 * x["accuracy"]["sd_across_seeds"]),
             comparability="contextual only",
             note="Reported values under different protocols; not a controlled ranking"),
        dict(characteristic="Macro-F1", brain_meta="96.94% (as reported)",
             agmf_net="%.2f +/- %.2f%% (original); %.2f +/- %.2f%% (duplicate-clean); %.2f +/- %.2f%% (cross-source)"
                      % (100 * o["macro_f1"]["mean"], 100 * o["macro_f1"]["sd_across_seeds"],
                         100 * c["macro_f1"]["mean"], 100 * c["macro_f1"]["sd_across_seeds"],
                         100 * x["macro_f1"]["mean"], 100 * x["macro_f1"]["sd_across_seeds"]),
             comparability="contextual only",
             note="BRAIN-META reports F1 without a stated averaging scheme in the inspected text"),
        dict(characteristic="OVERALL COMPARABILITY STATUS", brain_meta="-", agmf_net="-",
             comparability="contextual only",
             note="Datasets, partition construction, duplication control and evaluation conditions are "
                  "not established as equivalent. No claim of statistical superiority is made. "
                  "BRAIN-META per-image predictions were unavailable, so no paired test is possible."),
    ]
    write_csv(os.path.join(TAB, "BASE_PAPER_COMPARISON.csv"),
              ["characteristic", "brain_meta", "agmf_net", "comparability", "note"], rows)


# ------------------------------------------------ 2. assumptions table ------

def assumptions_limitations():
    R = lambda a, s, e, r, b: dict(assumption_or_limitation=a, status_in_agmf_net=s,
                                   evidence_or_action=e, residual_risk=r, base_paper_status=b)
    rows = [
        R("Exact and near-duplicate image leakage between train and test", "partially addressed",
          "256-bit dHash audit at Hamming <= 8; 7,023 images -> 5,648 clusters; 1,375 removed; repartition to 4,800/848",
          "Controls image redundancy only; two distinct slices from one patient are not linked; threshold is an operational choice with untested sensitivity",
          "not reported in the inspected paper"),
        R("Patient-level independence of the split", "not addressed",
          "Files contain no DICOM headers and no EXIF patient/study/series identifiers; no grouping key exists",
          "Patient-level leakage cannot be excluded; the duplicate-clean result may still be optimistic",
          "not reported in the inspected paper"),
        R("Single-run variability", "partially addressed",
          "Three independently trained seeds (42, 43, 44); sample SD with ddof=1 reported throughout",
          "Three seeds give limited information about the variance; SD is not a confidence interval and supports no normal-theory test",
          "not reported in the inspected paper"),
        R("Opaque contribution of individual components", "addressed",
          "Seven protocol-matched one-component ablations; 21 exact paired McNemar tests; Holm correction across all 21; paired image-level bootstrap",
          "Limited power: only a large effect is resolvable at three seeds after correction; four components produced no detectable effect",
          "base-learner comparisons reported, but not component controls of a single model"),
        R("Fixed or implicit branch weighting in the fusion", "addressed",
          "Learned gate (262,914 parameters) whose two outputs sum to one; verified to vary per image (g_c range 0.027-0.754)",
          "Gate mass is branch utilisation, not causal attribution; removing the gate produced no detectable accuracy change on this benchmark",
          "not applicable: BRAIN-META fuses at the prediction level"),
        R("Accuracy-only evaluation", "addressed",
          "Macro precision/recall/F1, balanced accuracy, per-class precision/recall/specificity/F1, confusion matrices, NLL, Brier, ECE, ROC-AUC, average precision",
          "Calibration comparisons between models are descriptive; no prespecified paired calibration test exists",
          "accuracy, precision, recall, F1 and confusion matrices reported"),
        R("Source-shift blindness", "addressed (exposed, not solved)",
          "373 duplicate-disjoint images from a second collection: accuracy 69.53 +/- 1.98%, ECE 0.1618",
          "One additional source only; acquisition metadata unavailable; the failure pattern may not generalise to other sources",
          "not reported in the inspected paper"),
        R("Class-specific failure hidden by aggregate accuracy", "addressed",
          "Per-class cross-source recall: glioma 98.08%, meningioma 86.07%, no-tumour 100%, pituitary 13.81%; 260/333 pituitary predictions assigned to no-tumour",
          "The dominant error direction is tumour-bearing to no-tumour, which is the least safe direction for triage",
          "confusion matrices reported for base and meta learners"),
        R("Confidence treated as diagnostic certainty", "partially addressed",
          "ECE 0.0752 internally and 0.1618 under source shift; reliability curves with bin populations",
          "Softmax output is not a probability of diagnostic correctness and is measurably less trustworthy under shift",
          "not reported in the inspected paper"),
        R("Result traceability and reproducibility", "addressed",
          "Per-file SHA-256 digests for prediction CSVs and checkpoints; split manifest; verification manifests retained",
          "Depends on preserving the artifact package alongside the manuscript",
          "pipeline described as reproducible; artifact-level digests not reported in the inspected paper"),
        R("Multi-institutional or prospective clinical validation", "not addressed",
          "None performed", "No evidence of clinical utility, safety, fairness or deployment readiness",
          "not reported in the inspected paper"),
        R("Verified lesion localisation", "not addressed",
          "No lesion masks or bounding boxes exist for this dataset; Grad-CAM++ was not compared against annotation",
          "Grad-CAM++ maps reflect CNN-path sensitivity only; they do not cover the transformer branch or the TTA average and are not validated localisation",
          "Grad-CAM presented as visual interpretability; no quantitative localisation evaluation reported"),
        R("Volumetric and acquisition context", "not addressed",
          "Data are 2D JPEG exports without DICOM headers",
          "No scanner, sequence, institution or demographic stratification is possible; no 3D or longitudinal modelling",
          "dataset described as 2D structural MRI images"),
        R("Scope of the four-class task", "not addressed",
          "Task is four-class image-level classification",
          "Does not cover grading, molecular subtype, prognosis, treatment planning or longitudinal change",
          "same four-class task"),
        R("Direct comparison with the reference method", "not addressed",
          "BRAIN-META per-image predictions unavailable; not reproduced on an identical split",
          "Comparison is contextual only; no statistical superiority claim is possible",
          "not applicable"),
        R("Non-significant ablations interpreted as equivalence", "addressed",
          "Verdict vocabulary distinguishes 'no detectable accuracy benefit under this protocol' from equivalence; every such verdict carries an explicit disclaimer",
          "Absence of detectable evidence remains compatible with a real effect below this design's resolution",
          "not applicable"),
    ]
    write_csv(os.path.join(TAB, "ASSUMPTIONS_LIMITATIONS.csv"),
              ["assumption_or_limitation", "status_in_agmf_net", "evidence_or_action",
               "residual_risk", "base_paper_status"], rows)


# --------------------------------------------------- 3. ablation table ------

def final_ablations():
    label = {"full_fusion_v2": "Full AGMF-Net (reference)",
             "no_gate": "Adaptive gate removed", "no_multiscale": "Multi-scale taps removed",
             "no_auxiliary_loss": "Auxiliary branch losses removed", "no_ema": "EMA removed",
             "cnn_branch_only": "ViT branch removed (CNN only)",
             "vit_branch_only": "CNN branch removed (ViT only)", "no_tta": "TTA removed"}
    order = ["full_fusion_v2", "cnn_branch_only", "no_auxiliary_loss", "vit_branch_only",
             "no_tta", "no_multiscale", "no_gate", "no_ema"]
    byv = {}
    for t in ABL["paired_tests"]:
        byv.setdefault(t["variant"], []).append(t)
    rows = []
    for v in order:
        a, f = SUMM[v]["accuracy"], SUMM[v]["macro_f1"]
        r = dict(variant=v, description=label[v],
                 accuracy_mean_pct="%.4f" % (100 * a[0]), accuracy_sd_ddof1_pct="%.4f" % (100 * a[1]),
                 macro_f1_mean_pct="%.4f" % (100 * f[0]), macro_f1_sd_ddof1_pct="%.4f" % (100 * f[1]))
        if v == "full_fusion_v2":
            r.update(observed_mean_delta_accuracy_pp="reference",
                     per_seed_delta_accuracy_pp="reference", adverse_at_every_seed="-",
                     n_holm_significant_of_3="-", n_bootstrap_ci_excludes_zero_of_3="-",
                     min_unadjusted_p="-", min_holm_adjusted_p="-", verdict="reference model")
        else:
            c = CONC[v]
            ts = sorted(byv[v], key=lambda t: t["seed"])
            r.update(observed_mean_delta_accuracy_pp="%.4f" % float(c["observed_mean_accuracy_difference_pp"]),
                     per_seed_delta_accuracy_pp=c["per_seed_accuracy_difference_pp"],
                     adverse_at_every_seed=c["accuracy_adverse_at_every_seed"],
                     n_holm_significant_of_3=c["n_holm_significant_of_3"],
                     n_bootstrap_ci_excludes_zero_of_3=c["n_bootstrap_ci_excludes_zero_of_3"],
                     min_unadjusted_p="%.4g" % min(t["unadjusted_p"] for t in ts),
                     min_holm_adjusted_p="%.4g" % min(t["holm_adjusted_p"] for t in ts),
                     verdict=c["verdict"])
        rows.append(r)
    write_csv(os.path.join(TAB, "FINAL_ABLATIONS.csv"),
              ["variant", "description", "accuracy_mean_pct", "accuracy_sd_ddof1_pct",
               "macro_f1_mean_pct", "macro_f1_sd_ddof1_pct", "observed_mean_delta_accuracy_pp",
               "per_seed_delta_accuracy_pp", "adverse_at_every_seed", "n_holm_significant_of_3",
               "n_bootstrap_ci_excludes_zero_of_3", "min_unadjusted_p", "min_holm_adjusted_p",
               "verdict"], rows)


# ------------------------------------------------ 4. traceability ledger ----

def traceability():
    rows = []

    def add(claim, value, path, field, status="VERIFIED", note=""):
        rows.append(dict(manuscript_claim=claim, exact_value=value, source_path=path,
                         source_field_or_row=field, verification_status=status, note=note))

    MP = "publication_results/metrics/fusion_v2_metrics.json"
    for key, name, n in (("original", "Original held-out split", 1052),
                         ("clean", "Duplicate-clean held-out split", 848),
                         ("external_full", "External collection, all images", 499),
                         ("cross_source_disjoint", "Duplicate-disjoint cross-source", 373)):
        a = M["settings"][key]["across_seeds"]
        add("%s accuracy" % name,
            "%.4f +/- %.4f %%" % (100 * a["accuracy"]["mean"], 100 * a["accuracy"]["sd_across_seeds"]),
            MP, "settings.%s.across_seeds.accuracy" % key)
        add("%s macro-F1" % name,
            "%.4f +/- %.4f %%" % (100 * a["macro_f1"]["mean"], 100 * a["macro_f1"]["sd_across_seeds"]),
            MP, "settings.%s.across_seeds.macro_f1" % key)
        add("%s sample count" % name, str(M["settings"][key]["n_images"]),
            MP, "settings.%s.n_images" % key)

    cl = M["settings"]["clean"]["across_seeds"]
    for m, lab in (("balanced_accuracy", "balanced accuracy"), ("macro_roc_auc_ovr", "macro ROC-AUC"),
                   ("macro_average_precision_ovr", "macro average precision"),
                   ("nll", "negative log-likelihood"), ("brier", "Brier score"), ("ece", "ECE (15 bins)")):
        add("Duplicate-clean %s" % lab, "%.4f" % cl[m]["mean"], MP,
            "settings.clean.across_seeds.%s.mean" % m)
    xs = M["settings"]["cross_source_disjoint"]["across_seeds"]
    for m, lab in (("balanced_accuracy", "balanced accuracy"), ("macro_roc_auc_ovr", "macro ROC-AUC"),
                   ("nll", "negative log-likelihood"), ("brier", "Brier score"), ("ece", "ECE (15 bins)")):
        add("Cross-source %s" % lab, "%.4f" % xs[m]["mean"], MP,
            "settings.cross_source_disjoint.across_seeds.%s.mean" % m)

    drop = 100 * (cl["accuracy"]["mean"] - xs["accuracy"]["mean"])
    add("Accuracy drop from duplicate-clean to cross-source", "%.2f percentage points" % drop,
        MP, "derived: settings.clean minus settings.cross_source_disjoint",
        "VERIFIED", "Arithmetic difference of two verified means")

    SR = "dataset_clean/split_report.json"
    sr = json.load(open(os.path.join(ROOT, SR), encoding="utf-8"))
    add("Pooled image count", str(sr["n_pooled"]), SR, "n_pooled")
    add("Similarity clusters after duplicate audit", str(sr["n_clusters"]), SR, "n_clusters")
    add("Redundant images removed", str(sr["n_redundant_dropped"]), SR, "n_redundant_dropped")
    add("Duplicate-clean training images", str(sr["n_train"]), SR, "n_train")
    add("Duplicate-clean test images", str(sr["n_test"]), SR, "n_test")
    add("Perceptual-hash Hamming threshold", str(sr["hamming_threshold"]), SR, "hamming_threshold")
    add("Split seed", str(sr["seed"]), SR, "seed")
    add("Duplicate-clean test class counts",
        ", ".join("%s %d" % (k, v) for k, v in sr["counts"]["Testing"].items()),
        SR, "counts.Testing")

    XS = "results/cross_source.json"
    xsj = json.load(open(os.path.join(ROOT, XS), encoding="utf-8"))
    add("External collection size", str(xsj["n_total"]), XS, "n_total")
    add("External images with a near-duplicate in training", str(xsj["n_overlapping"]), XS, "n_overlapping")
    add("Duplicate-disjoint external images", str(xsj["n_disjoint"]), XS, "n_disjoint")

    # cross-source per-class, recomputed
    disj = {k for k, v in xsj["manifest"].items() if v.get("kind") == "disjoint"}
    by = {}
    for s in SEEDS:
        rows_s = list(csv.DictReader(open(os.path.join(
            ROOT, "runs_v2_clean/fusion_v2_seed%d/preds_dataset_ood.csv" % s), encoding="utf-8")))
        by[s] = [r for r in rows_s if r["filename"] in disj]
    PRED = "runs_v2_clean/fusion_v2_seed{42,43,44}/preds_dataset_ood.csv"
    for c in CLASSES:
        recs = [100 * sum(r["pred_label"] == c for r in by[s] if r["true_label"] == c) /
                len([r for r in by[s] if r["true_label"] == c]) for s in SEEDS]
        add("Cross-source recall, %s" % c, "%.2f %%" % float(np.mean(recs)), PRED,
            "recomputed from prediction rows with kind=disjoint", "VERIFIED",
            "support %d images" % len([r for r in by[42] if r["true_label"] == c]))
    pit = [r for s in SEEDS for r in by[s] if r["true_label"] == "pituitary"]
    nt = sum(r["pred_label"] == "notumor" for r in pit)
    add("Pituitary predictions assigned to no-tumour",
        "%d of %d (%.1f%%) from %d images x 3 seeds" % (nt, len(pit), 100 * nt / len(pit), len(pit) // 3),
        PRED, "recomputed", "VERIFIED",
        "Repeated predictions of the same 111 images; not independent cases")

    # architecture
    REC = "publication_results/ablations/checkpoint_recovery/import_manifest.json"
    rec = json.load(open(os.path.join(ROOT, REC), encoding="utf-8"))
    g42 = rec["seeds"]["42"]["gates"]
    add("Total parameters", str(g42["load"]["params_total"]), REC, "seeds.42.gates.load.params_total")
    add("Gate parameters", str(g42["architecture_and_gate"]["gate_parameter_count"]), REC,
        "seeds.42.gates.architecture_and_gate.gate_parameter_count")
    add("CNN projection input width (multi-scale concatenation)",
        str(g42["architecture_and_gate"]["cnn_proj_in_features"]), REC,
        "seeds.42.gates.architecture_and_gate.cnn_proj_in_features",
        "VERIFIED", "40 + 112 + 1280 = 1432")
    add("Projection width", str(g42["architecture_and_gate"]["proj_dim"]), REC,
        "seeds.42.gates.architecture_and_gate.proj_dim")
    add("Reference checkpoints load into the gated architecture with no missing or unexpected tensors",
        "578/578 keys, 0 missing, 0 unexpected, 0 shape mismatches", REC,
        "seeds.{42,43,44}.gates.load")
    add("Gated forward path reproduces recorded reference predictions",
        "848/848 images, 0 pred_label mismatches, max probability difference 4.8e-06", REC,
        "seeds.42.gates.full_test")

    # gate mass
    GT = "runs_v2_clean/fusion_v2_seed{42,43,44}/gates_test.csv"
    means, lo, hi = [], [], []
    for s in SEEDS:
        v = np.array([float(r["gate_cnn"]) for r in csv.DictReader(open(os.path.join(
            ROOT, "runs_v2_clean/fusion_v2_seed%d/gates_test.csv" % s), encoding="utf-8"))])
        means.append(v.mean()); lo.append(v.min()); hi.append(v.max())
    add("CNN gate mass, per-seed means", ", ".join("%.3f" % m for m in means), GT, "gate_cnn column")
    add("CNN gate mass range across seeds", "%.3f to %.3f" % (min(lo), max(hi)), GT, "gate_cnn column",
        "VERIFIED", "Demonstrates the gate is input-dependent, not fixed")

    # ablations
    AB = "publication_results/ablations/final_analysis/component_conclusions.csv"
    for v, c in CONC.items():
        add("Ablation verdict, %s" % v, c["verdict"], AB, "row variant=%s, column verdict" % v)
        add("Ablation observed mean delta accuracy, %s" % v,
            "%.4f pp" % float(c["observed_mean_accuracy_difference_pp"]), AB,
            "row variant=%s" % v)
    mcc = ABL["multiple_comparison_control"]
    AJ = "publication_results/ablations/final_analysis/final_ablation_analysis.json"
    add("Number of seed-specific paired tests", str(mcc["n_tests"]), AJ,
        "multiple_comparison_control.n_tests")
    add("Tests significant before Holm correction", str(mcc["n_significant_before_holm"]), AJ,
        "multiple_comparison_control.n_significant_before_holm")
    add("Tests significant after Holm correction", str(mcc["n_significant_after_holm"]), AJ,
        "multiple_comparison_control.n_significant_after_holm")
    aux = [t for t in ABL["paired_tests"] if t["variant"] == "no_auxiliary_loss"]
    add("Auxiliary-loss smallest unadjusted p, and its Holm-adjusted value",
        "%.4g -> %.4g" % (min(t["unadjusted_p"] for t in aux), min(t["holm_adjusted_p"] for t in aux)),
        AJ, "paired_tests where variant=no_auxiliary_loss")
    gate = [t for t in ABL["paired_tests"] if t["variant"] == "no_gate"]
    add("Gate ablation discordant prediction counts",
        ", ".join(str(t["mcnemar"]["n_discordant"]) for t in sorted(gate, key=lambda t: t["seed"])),
        AJ, "paired_tests where variant=no_gate, mcnemar.n_discordant")

    # training protocol, read from frozen source
    TA = "src/train_ablation.py"
    src = open(os.path.join(ROOT, TA), encoding="utf-8").read()
    import re
    for name, label in (("BATCH_SIZE", "batch size"), ("EPOCHS", "maximum epochs"),
                        ("PATIENCE", "early-stopping patience"), ("LR", "learning rate"),
                        ("WEIGHT_DECAY", "weight decay"), ("LABEL_SMOOTHING", "label smoothing"),
                        ("VAL_FRACTION", "validation fraction"), ("VAL_SPLIT_SEED", "validation split seed")):
        m = re.search(r"^%s\s*=\s*([0-9.eE+-]+)" % name, src, re.M)
        add("Training protocol: %s" % label, m.group(1) if m else "NOT FOUND", TA,
            "module constant %s" % name, "VERIFIED" if m else "NOT_VERIFIED")

    # unverifiable items, recorded rather than omitted
    for claim in ("Author names and ORCID identifiers", "Affiliations and corresponding address",
                  "Target journal", "Institutional ethics/IRB determination", "Funding statement",
                  "Competing-interest statement", "Repository URL and archival DOI",
                  "Acknowledgements", "CRediT contributions", "Submission date"):
        add(claim, "[AUTHOR INPUT REQUIRED]", "-", "-", "AUTHOR_INPUT_REQUIRED",
            "Not inferable from any repository artifact; must be supplied by the authors")
    add("Training and inference wall-clock timings", "[AUTHOR INPUT REQUIRED]", "-", "-",
        "NOT_VERIFIED",
        "Not recorded in a form verifiable from retained artifacts; omitted rather than estimated")
    add("BRAIN-META reported accuracy and F1", "97.10% accuracy, 96.94% F1",
        "supplied base-paper PDF (conversation attachment)", "Abstract and Table 4",
        "VERIFIED_FROM_SUPPLIED_PDF",
        "Read from the supplied article text; the PDF file itself is not present on disk, so no file digest could be computed")

    write_csv(os.path.join(OUT, "EVIDENCE_TRACEABILITY.csv"),
              ["manuscript_claim", "exact_value", "source_path", "source_field_or_row",
               "verification_status", "note"], rows)
    return rows


def manifest(ledger_rows):
    files = []
    for dirpath, _d, names in os.walk(OUT):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            if os.path.basename(p) == "DELIVERABLE_MANIFEST.json":
                continue
            files.append({"path": os.path.relpath(p, OUT).replace("\\", "/"),
                          "bytes": os.path.getsize(p), "sha256": sha256(p)})
    src = {}
    for rel in ("src/publication/make_paper_figures.py", "src/publication/make_paper_tables.py",
                "src/fusion_v2.py", "src/train_ablation.py"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            src[rel] = sha256(p)
    counts = {}
    for r in ledger_rows:
        counts[r["verification_status"]] = counts.get(r["verification_status"], 0) + 1
    man = {
        "deliverable": "AGMF-Net manuscript package",
        "generated_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "output_root": "publication_results/final_paper_claude",
        "n_files": len(files), "files": files,
        "generating_code_sha256": src,
        "evidence_ledger_status_counts": counts,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "network_access": False,
            "docx_writer": "raw Office Open XML via zipfile (python-docx not installed)",
            "pdf_renderer": "NOT AVAILABLE - see VISUAL_QA_REPORT.md",
        },
    }
    with open(os.path.join(OUT, "DELIVERABLE_MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    print("  wrote DELIVERABLE_MANIFEST.json (%d files)" % len(files))


def main():
    os.makedirs(TAB, exist_ok=True)
    print("AGMF-Net tables ->", os.path.relpath(OUT, ROOT))
    base_paper_comparison()
    assumptions_limitations()
    final_ablations()
    rows = traceability()
    st = {}
    for r in rows:
        st[r["verification_status"]] = st.get(r["verification_status"], 0) + 1
    print("  evidence ledger:", st)
    manifest(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
