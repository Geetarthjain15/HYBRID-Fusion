# Supplementary evidence and reproducibility appendix for AGMF-Net

## S1. Identity and scope

AGMF-Net is the publication-facing name of the verified repository class `FusionV2`. No checkpoint was retrained or altered for manuscript preparation. The headline model is the gated, multi-scale EfficientNet-B0 plus ViT-B/16 architecture with auxiliary supervision, EMA, and horizontal-flip TTA.

## S2. Immutable evidence

- Original split: 1,052 images; accuracy 99.14 ± 0.10%; macro-F1 99.12 ± 0.09%.
- Duplicate-clean split: 848 images; accuracy 98.27 ± 0.07%; macro-F1 98.28 ± 0.04%.
- Duplicate-disjoint cross-source subset: 373 images; accuracy 69.53 ± 1.98%; macro-F1 68.69 ± 2.21%.
- Seeds: 42, 43, 44.
- Bootstrap: 2,000 stratified image resamples per seed; seed 12,345.
- ECE: 15 equal-width bins on maximum class probability.

## S3. Verified checkpoints

| Seed | SHA-256 | Clean accuracy |
|---:|---|---:|
| 42 | `9b328ba83df5d44b542e19faae68fd1493fee61e0c5b80e493c2f8a387ed592e` | 98.23% |
| 43 | `9f97b6aa0c9ec3f1124e593c8c29e11d31f4aef8bc08e20a40e7ed1f29e80f25` | 98.23% |
| 44 | `4155a2b71df99a6159e1e4a5dbe635f4e9ea7852a6704d804dd1fb5d43aca434` | 98.35% |

All checkpoints loaded with 578/578 keys, zero missing keys, zero unexpected keys, and zero shape mismatches. Recomputed labels matched all 848 stored predictions for every seed. Maximum probability differences were below `5e-6` and were attributed to CPU/CUDA floating-point kernel differences.

## S4. Duplicate-clean partition

The recorded cleaning procedure pooled 7,023 images, identified 5,648 perceptual-similarity clusters, and removed 1,375 redundant images. The resulting partition contains 4,800 training and 848 testing images. The test counts are glioma 243, meningioma 228, no-tumor 116, and pituitary 261. The similarity procedure does not identify patients.

## S5. Ablation matrix and inference-only control

Seven variants were compared with the full model: no gate, no multi-scale taps, no auxiliary loss, no EMA, CNN branch only, ViT branch only, and no TTA. Training variants used three seeds. No-TTA reused the corresponding verified checkpoint and changed only inference augmentation. Tests were paired by exact filename and true label. Holm correction covered all 21 seed-specific McNemar tests.

## S6. Statistical independence boundary

The 848 test images are the sampling units for image-level bootstrap intervals. The three model seeds are repeated predictions of the same images and are summarized separately. Neither the 2,544 seed-image outputs nor the 333 pituitary outputs in cross-source analysis are treated as independent patient cases.

## S7. Patient-group audit

Ten candidate grouping keys were evaluated; none was accepted. The local data contain JPEG exports without DICOM, EXIF, patient, study, or series identifiers. Numeric filename suffixes are export counters. Perceptual clusters and sequential ranges were explicitly rejected as patient proxies. The permitted conclusion is: **patient-level independence cannot be verified from the available dataset metadata.**

## S8. Explainability boundary

Grad-CAM++ targets the selected fused class logit at the last spatial EfficientNet feature stage. It explains the CNN pathway for an original-view forward pass. It does not spatially explain the ViT pathway or the average of original and horizontally flipped TTA views. No tumor mask, bounding box, or clinician localization was available. Figures are therefore qualitative only.

## S9. Base-paper comparison boundary

BRAIN-META reports 97.10% accuracy and 96.94% F1 for an XGBoost meta-learner on 1,311 test images. AGMF-Net uses different partitions, preprocessing, software, and architecture. BRAIN-META per-image predictions were unavailable. The comparison is contextual; no statistical superiority claim is permitted.

## S10. Reproduction commands

```powershell
env_win\Scripts\python.exe src\publication\evaluate_fusion_v2.py --project-root . --out-root publication_results\metrics --bootstrap 2000 --bootstrap-seed 12345
env_win\Scripts\python.exe src\publication\test_final_analysis.py
env_win\Scripts\python.exe src\publication\test_phase3c_artifacts.py
env_win\Scripts\python.exe src\publication\test_phase3d_evidence.py
env_win\Scripts\python.exe src\publication\test_fusion_v2_gradcam.py
```

Environment versions and source-file hashes are stored in the corresponding JSON manifests. Reproduction should use those machine-readable records rather than versions inferred from prose.

## S11. Claims checklist

Allowed: image-level internal results; duplicate-clean results; substantial cross-source degradation; adaptive gate utilization; controlled ablation conclusions. Prohibited without new evidence: patient-independent evaluation; clinical readiness; universal hospital/scanner generalization; statistical superiority over BRAIN-META; gate mass as causal explanation; Grad-CAM++ as verified tumor localization.
