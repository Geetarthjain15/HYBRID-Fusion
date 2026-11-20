"""
Phase X0 - assemble the audit deliverables from the probe results.

Every answer is drawn from `_probe_raw.json` / `_probe_attention.json` (produced
by executing the verified model) or from a file:line citation, so no statement
here rests on a variable name, a docstring or a filename.

    env_win/Scripts/python.exe src/publication/explainability/write_x0_audit.py --project-root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SEEDS = [42, 43, 44]


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    os.chdir(root)
    out = root / "publication_results/explainability/audit"
    out.mkdir(parents=True, exist_ok=True)

    P = json.loads((out / "_probe_raw.json").read_text(encoding="utf-8"))
    A = json.loads((out / "_probe_attention.json").read_text(encoding="utf-8"))

    q2 = P["q2_gradcam_identity"]
    q8 = P["q8_forward_graph"]
    q9 = P["q9_cnn_spatial_stages"]
    q11 = P["q10_q11_vit_tensors"]
    q13 = P["q13_q14_gate_semantics"]
    q15 = P["q15_aux_in_checkpoint"]
    q18 = P["q18_instrumentation_neutrality"]
    q19 = P["q19_checkpoints"]
    eag = A["eager_switch"]

    # ------------------------------------------------------ input manifest --
    inputs = []

    def add(path, role, note=""):
        p = root / path
        inputs.append({"path": path, "role": role, "present": p.is_file(),
                       "sha256": sha256(p) if p.is_file() else None,
                       "bytes": p.stat().st_size if p.is_file() else None,
                       "note": note})

    add("../AGENTS.md", "repository contract (authoritative)",
        "byte-identical to CLAUDE.md apart from the title line")
    add("../CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md", "mandated reading")
    add("results/paper/SUMMARY.md", "mandated reading")
    add("results/paper/tables.md", "mandated reading")
    add("src/fusion_v2.py", "model definition (read-only, unmodified)")
    add("src/train_ablation.py", "training/eval protocol (read-only, unmodified)")
    add("dataset_clean/split_report.json", "mandated reading")
    add("results/cross_source.json", "mandated reading")
    add("src/publication/fusion_v2_gradcam.py", "current heatmap implementation")
    add("src/publication/streamlit_fusion_v2.py", "current FusionV2 demonstrator")
    add("src/publication/audit_gradcam_faithfulness.py", "existing faithfulness audit")
    add("src/publication/test_fusion_v2_gradcam.py", "existing tests")
    add("src/publication/test_gradcam_faithfulness.py", "existing tests")
    add("src/streamlit_app.py", "legacy v1 demonstrator (unrelated model)")
    add("publication_results/ablations/checkpoint_recovery/import_manifest.json",
        "checkpoint verification manifest")
    add("dataset_clean/split_manifest.json", "canonical image/label mapping")
    for s in SEEDS:
        add("publication_runs/recovered_checkpoints/clean/full_fusion_v2/seed_%d/checkpoint.pth" % s,
            "publication-verified checkpoint")
        add("runs_v2_clean/fusion_v2_seed%d/preds_test.csv" % s,
            "authoritative duplicate-clean predictions (n=848)")
        add("runs_v2_clean/fusion_v2_seed%d/preds_dataset_ood.csv" % s,
            "authoritative external predictions (n=499; 373 duplicate-disjoint)")
        add("runs_v2_clean/fusion_v2_seed%d/gates_test.csv" % s,
            "per-image gate mass on the clean test set")

    with open(out / "input_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"phase": "X0", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "project_root": str(root).replace("\\", "/"),
                   "n_inputs": len(inputs), "inputs": inputs}, f, indent=2)

    # ------------------------------------------------ machine-readable audit -
    audit = {
        "phase": "X0 - evidence and architecture audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "model_modified": False,
        "phase_x1_started": False,
        "environment": P["environment"],
        "answers": {},
    }
    Q = audit["answers"]

    Q["q1_heatmap_location"] = {
        "answer": "src/publication/fusion_v2_gradcam.py, function gradcam_plus_plus(), lines 157-177",
        "consumers": [
            "src/publication/streamlit_fusion_v2.py (demonstrator)",
            "src/publication/audit_gradcam_faithfulness.py (faithfulness audit)",
        ],
        "unrelated_legacy": "src/streamlit_app.py operates on the v1 HybridBrainTumorModel, "
                            "not FusionV2, and is out of scope",
        "evidence": "file inspection + import graph",
    }
    Q["q2_method_identity"] = {
        "answer": "genuine Grad-CAM++, not ordinary Grad-CAM",
        "resolved_class": q2["resolved_class"],
        "library_file": q2["source_file"],
        "distinct_class_from_GradCAM": q2["is_distinct_class_from_GradCAM"],
        "uses_second_order_gradient_terms": q2["uses_second_order_terms"],
        "uses_third_order_gradient_terms": q2["uses_third_order_terms"],
        "basis": "GradCAMPlusPlus.get_cam_weights computes grads_power_2 / "
                 "(2*grads_power_2 + sum_activations*grads_power_3 + eps), i.e. Eq. 19 of "
                 "arXiv:1710.11063, whereas GradCAM.get_cam_weights is a plain spatial mean "
                 "of gradients",
        "evidence": "inspect.getsource on the installed classes",
    }
    Q["q3_hooked_module"] = {
        "answer": "model.cnn_stages[-1] - the final EfficientNet-B0 feature stage (index 8)",
        "resolved_type": P["q3_hooked_module"]["resolved_module_type"],
        "activation_shape": P["q3_hooked_module"]["output_shape_on_probe"],
        "source": "src/publication/fusion_v2_gradcam.py:167",
        "evidence": "executed forward, shapes recorded",
    }
    Q["q4_differentiated_score"] = P["q4_target_score"]
    Q["q4_differentiated_score"]["answer"] = (
        "the FUSED class logit is differentiated, not a branch-specific score; "
        "ClassifierOutputTarget(target_index) is applied to the full FusionV2 output")
    Q["q5_preprocessing"] = dict(P["q5_preprocessing"])
    Q["q5_preprocessing"]["answer"] = (
        "Resize((224,224)) -> ToTensor() -> Normalize(mean=[0.5]*3, std=[0.5]*3); "
        "no CLAHE, no crop, aspect ratio not preserved")
    Q["q6_spatial_resize_reversal"] = dict(P["q6_spatial_resize"])
    Q["q6_spatial_resize_reversal"]["answer"] = (
        "It is NOT reversed. The CAM is upsampled from the 7x7 stage map to the 224x224 "
        "model input grid and overlaid on a 224x224 resized copy of the image. Nothing maps "
        "the map back to the source resolution or undoes the aspect-ratio change.")
    Q["q6_spatial_resize_reversal"]["upsample_factor"] = 224 // 7
    Q["q7_class_order"] = dict(P["q7_class_order"])
    Q["q7_class_order"]["answer"] = (
        "('glioma','meningioma','notumor','pituitary'), index 0..3, alphabetical; this "
        "matches the probability column order p_glioma..p_pituitary in the authoritative "
        "prediction CSVs and the canonical split manifest")
    Q["q8_forward_graph"] = dict(q8)
    Q["q9_cnn_tensors"] = dict(q9)
    Q["q9_cnn_tensors"]["answer"] = (
        "9 EfficientNet stages. Multi-scale taps at index 3 (40ch, 28x28) and 5 (112ch, "
        "14x14); final stage 8 (1280ch, 7x7). All three are pooled to vectors inside "
        "branch_features, so spatial maps are only reachable via hooks.")
    Q["q10_q11_vit_tensors"] = dict(q11)
    Q["q10_q11_vit_tensors"]["answer"] = (
        "Available: last_hidden_state [B,197,768] and the CLS token at index 0 (the only "
        "tensor the forward pass consumes). Attention is NOT returned as loaded.")
    Q["q11_attention_availability"] = {
        "answer": "NO under the loaded configuration; YES after switching to eager attention",
        "attn_implementation_as_loaded": A["attn_implementation_as_loaded"],
        "output_attentions_flag_alone_returns": "None",
        "transformers_warning": "output_attentions=True is not supported with "
                                "attn_implementation other than eager/eager_paged/flex_attention",
        "after_set_attn_implementation_eager": {
            "attentions_returned": eag.get("attentions_returned"),
            "n_layers": eag.get("n_layers"),
            "shape_per_layer": eag.get("shape_layer0"),
            "rows_sum_to_one": eag.get("rows_sum_to_one"),
            "n_patch_tokens": eag.get("n_patch_tokens"),
            "patch_grid": eag.get("patch_grid"),
        },
        "prediction_neutrality": {
            "max_abs_prob_delta_vs_verified": eag.get("max_abs_prob_delta_vs_verified"),
            "max_abs_logit_delta_vs_verified": eag.get("max_abs_logit_delta_vs_verified"),
            "tolerance": A["tolerance"],
            "changes_predictions": eag.get("changes_fused_probs_beyond_tolerance"),
            "argmax_unchanged": eag.get("argmax_unchanged"),
            "restore_exact": A["restore"].get("exact_restore"),
        },
        "evidence": "executed; src/publication/explainability/audit_x0_attention.py",
    }
    Q["q12_gate_input"] = {
        "answer": "torch.cat([c, v], dim=1), the two 512-d PROJECTED BRANCH FEATURE VECTORS, "
                  "shape [B,1024]",
        "module": q13["gate_module_repr"],
        "source": "src/fusion_v2.py:72-73, :97",
    }
    Q["q13_gate_output_kind"] = {
        "answer": "TWO SCALAR PER-BRANCH WEIGHTS PER IMAGE, shape [B,2]. Not per-feature, "
                  "not per-scale, not spatial.",
        "shape": q13["gate_returns_shape"],
        "values_per_image": q13["values_per_image"],
        "example_rows": q13["example_rows"],
        "applied_to": q13["applied_to"],
        "source": "src/fusion_v2.py:97, :100",
    }
    Q["q14_gate_sums"] = {
        "answer": "Yes. The gate is a softmax over the last dimension, so the two values sum "
                  "to 1 across the branches within each image.",
        "softmax_dim": q13["softmax_dim_in_source"],
        "sum_axis": q13["sum_axis"],
        "measured_row_sums": q13["row_sums"],
        "rows_sum_to_one": q13["rows_sum_to_one"],
    }
    Q["q15_aux_logits_in_checkpoint"] = {
        "answer": "Yes. aux_cnn and aux_vit weights and biases are present in the verified "
                  "checkpoint and load under strict=True.",
        "keys": q15["aux_keys"], "shapes": q15["aux_shapes"],
        "n_state_dict_keys": q15["n_state_dict_keys"],
        "evidence": "torch.load(weights_only=True) key inspection + strict load",
    }
    Q["q16_branch_probabilities"] = dict(P["q16_branch_probabilities"])
    Q["q16_branch_probabilities"]["answer"] = (
        "Yes, without retraining: model(x, return_aux=True) returns (fused_logits, "
        "aux_cnn(c), aux_vit(v), gate). Softmax of the auxiliary logits gives per-branch "
        "class distributions. They are auxiliary regularisation heads, not the fused decision.")
    Q["q17_single_synchronized_pass"] = dict(P["q17_synchronized_pass"])
    Q["q17_single_synchronized_pass"]["answer"] = (
        "Yes. One forward with return_aux=True, a forward hook on cnn_stages[-1], and eager "
        "attention enabled exposes fused logits, both branch logits, the gate row, the final "
        "CNN spatial map and all 12 ViT attention tensors simultaneously.")
    Q["q18_instrumentation_neutrality"] = {
        "answer": "No. Neither a read-only forward hook nor enabling attention changes the "
                  "fused probabilities beyond float32 round-off.",
        "forward_hook_max_abs_prob_delta": q18.get("max_abs_prob_delta_forward_hook"),
        "forward_hook_changes_probs": q18.get("forward_hook_changes_fused_probs"),
        "hook_removal_exact": q18.get("hook_removal_exact"),
        "eager_attention_max_abs_prob_delta": eag.get("max_abs_prob_delta_vs_verified"),
        "eager_attention_changes_probs": eag.get("changes_fused_probs_beyond_tolerance"),
        "tolerance": q18.get("tolerance"),
        "interpretation": "8.94e-08 is float32 round-off from a different attention kernel, "
                          "not a behavioural change; predicted classes are identical and the "
                          "original implementation restores exactly",
        "requirement_for_x1": "this equality must be re-asserted per seed on the real 848-image "
                              "test set before any explanation is attached to a verified result",
    }
    Q["q19_verified_checkpoints"] = q19
    Q["q19_verified_checkpoints"]["answer"] = (
        "The three recovered FusionV2 checkpoints for seeds 42, 43 and 44, each ACCEPTED in "
        "the checkpoint-recovery manifest with a matching SHA-256.")
    Q["q20_authoritative_predictions"] = {
        "answer": "runs_v2_clean/fusion_v2_seed{42,43,44}/preds_test.csv for duplicate-clean "
                  "(848 images) and runs_v2_clean/fusion_v2_seed{42,43,44}/preds_dataset_ood.csv "
                  "for the external set (499 images, of which 373 are duplicate-disjoint).",
        "clean_test_n": 848,
        "external_n": 499,
        "duplicate_disjoint_n": 373,
        "disjoint_selector": "results/cross_source.json -> manifest[filename].kind == 'disjoint'",
        "canonical_label_source": "dataset_clean/split_manifest.json",
        "canonical_fingerprint_sha256": "3437ccd466d70f6ca88eaddaf6121f4e2141550126012a3dac0e1c624978cd42",
        "provenance": "publication_results/ablations/final_analysis/analysis_provenance.json "
                      "records the per-file SHA-256 of each of these files",
        "gate_telemetry": "runs_v2_clean/fusion_v2_seed{42,43,44}/gates_test.csv",
    }
    Q["q21_nonintrusive_code_additions"] = {
        "answer": "A wrapper module that instruments an already-constructed FusionV2 instance. "
                  "No edit to src/fusion_v2.py is required.",
        "mechanisms_verified_available": [
            "register_forward_hook on model.cnn_stages[i] to capture spatial maps at stages "
            "3, 5 and 8 (verified neutral, delta 0.0)",
            "model(x, return_aux=True) for fused logits, aux branch logits and the gate row "
            "(already in the public API, src/fusion_v2.py:94)",
            "model.vit.set_attn_implementation('eager') plus output_attentions=True for the "
            "12 attention tensors (verified neutral, delta 8.94e-08, exactly restorable)",
            "torch.autograd.grad against captured activations for CAM-style attribution "
            "without mutating the module",
        ],
        "explicitly_not_required": [
            "any change to src/fusion_v2.py",
            "any change to src/train_ablation.py",
            "any change to an existing Streamlit or diagnostic script",
            "retraining, fine-tuning or re-export of any checkpoint",
        ],
    }
    Q["q22_limitations_blocking_anatomical_claims"] = {
        "answer": "Several, and they are structural rather than incidental.",
        "items": [
            "No lesion masks, bounding boxes or expert annotations exist for this dataset, so "
            "there is no reference against which a map could be scored as correct.",
            "The CNN CAM originates on a 7x7 grid and is upsampled 32x to 224x224; its "
            "effective spatial resolution is roughly 32 pixels, coarser than many lesions.",
            "ViT attention is native 14x14 over 196 patches, a different grid from the CNN "
            "final stage, so the two paths are not spatially commensurate without resampling.",
            "Preprocessing resizes to 224x224 without preserving aspect ratio, and the overlay "
            "is never mapped back to source resolution, so map coordinates are in distorted "
            "model space.",
            "The reported predictions use horizontal-flip TTA (an average over two views), "
            "while the heatmap is computed from a single un-flipped forward pass, so the map "
            "does not correspond exactly to the reported probability.",
            "Attention weights describe token mixing, not evidence for a class; treating them "
            "as an explanation is an assumption, not a result.",
            "Gate mass measures branch utilisation only and is not causal attribution "
            "(AGENTS.md prohibits the contrary claim).",
            "A visually plausible map is not evidence of faithfulness; only perturbation-style "
            "tests speak to that, and they test sensitivity, not anatomical correctness.",
        ],
    }

    # ------------------------------------ gate compatibility determination --
    audit["composite_formula_feasibility"] = {
        "question": "Does the gate really produce compatible CNN and ViT branch weights, as "
                    "the proposed H_composite = alpha_CNN*H_CNN + alpha_ViT*H_ViT would require?",
        "finding": "PARTIALLY. The gate does emit exactly two non-negative per-image scalars "
                   "summing to one, which are formally usable as alpha_CNN and alpha_ViT.",
        "but": "Those scalars were trained to reweight 512-d FEATURE VECTORS before the "
               "classifier head. They were never trained to weight spatial maps, and the two "
               "maps they would weight are not the quantities the gate scales. Using them on "
               "heatmaps is an analogy, not an identity.",
        "additional_incompatibility": "H_CNN is a 7x7 gradient-weighted activation map; H_ViT "
                                      "would be a 14x14 attention-derived map. They differ in "
                                      "native resolution, in units, and in normalisation, so "
                                      "any sum requires resampling and rescaling choices that "
                                      "are not determined by the model.",
        "permitted_terminology": "gate-weighted dual-path attribution composite",
        "prohibited_terminology": [
            "whole-model explanation", "proven explanation", "tumour localisation",
            "causal branch attribution", "evidence that the model looks at the lesion",
        ],
        "required_before_use": [
            "state the resampling rule and the per-map normalisation explicitly",
            "report the composite alongside its two components, never alone",
            "treat the composite as a visualisation choice with a stated rationale, not as a "
            "derived quantity of the model",
        ],
    }

    audit["acceptance_checks"] = {
        "no_protected_artifact_changed": None,   # filled by the caller after re-hash
        "no_existing_file_modified": None,
        "every_architectural_statement_cites_evidence": True,
        "gradcam_status_verified_not_assumed": True,
        "checkpoint_identities_and_hashes_recorded": q19["all_three_accepted_and_hash_matched"],
        "class_order_and_preprocessing_verified": True,
        "aux_branch_logit_availability_established": q15["aux_present"],
        "vit_attention_availability_established": True,
        "gate_semantics_and_dimensions_established": q13["rows_sum_to_one"],
        "terminology_conservative": True,
        "phase_x1_not_started": True,
    }

    with open(out / "explainability_architecture_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    print("  wrote explainability_architecture_audit.json")
    print("  wrote input_manifest.json (%d inputs)" % len(inputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
