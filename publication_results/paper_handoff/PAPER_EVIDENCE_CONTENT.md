# AGMF-Net paper content and evidence pack

## 1. Paper identity

**Publication name:** AGMF-Net  
**Expansion:** Adaptive Gated Multi-scale Fusion Network  
**Repository implementation:** `FusionV2`  
**Task:** four-class 2D brain MRI image classification  
**Classes and fixed order:** glioma, meningioma, no-tumour (`notumor` in artifacts), pituitary

Recommended title:

> AGMF-Net: Adaptive Gated Multi-scale CNN–Vision Transformer Fusion with Duplicate-Aware and Cross-Source Evaluation for Brain MRI Classification

## 2. Core novelty statement

AGMF-Net replaces a probability-stacked collection of many CNN–ViT learners with one feature-level hybrid. The network extracts local features from multiple EfficientNet-B0 depths, combines them with a global ViT-B/16 representation after mapping both branches into the same feature width, and learns an image-dependent two-way gate. Auxiliary branch objectives preserve branch discriminability during training. The methodological contribution is coupled to a conservative evaluation design that distinguishes conventional, duplicate-clean and duplicate-disjoint cross-source performance.

The novelty must not be reduced to “higher accuracy than BRAIN-META.” The strongest defensible contribution is the combination of a specific adaptive feature-fusion design with duplicate-aware, calibration-aware, ablation-based and cross-source evaluation.

## 3. Why the earlier hybrid architecture was changed

The previous hybrid used the final 1,280-dimensional EfficientNet representation and the 768-dimensional ViT class token, concatenated them into 2,048 dimensions, and trained a small fusion head. The CNN and most ViT layers were frozen. Four limitations motivated AGMF-Net:

1. The CNN pathway had restricted fine-tuning even though a fully fine-tuned CNN control was stronger.
2. Only the final CNN output was used, discarding intermediate feature scales.
3. Plain unequal-width concatenation offered no explicit mechanism to balance branch utilization.
4. Neither branch had an independent training objective, so one pathway could become a passenger.

AGMF-Net addresses these design limitations with full fine-tuning at a reduced backbone learning rate, multi-depth pooling, equal-width projections, a learned gate and auxiliary classifiers.

## 4. Verified architecture

- CNN backbone: EfficientNet-B0.
- CNN taps: stage 3 (`40` channels), stage 5 (`112` channels), final stage (`1280` channels).
- Pooled CNN concatenation: `1432` dimensions.
- CNN projection: Linear `1432→512`, LayerNorm, GELU.
- ViT backbone: ViT-B/16 (`google/vit-base-patch16-224` architecture).
- ViT representation: 768-dimensional class token.
- ViT projection: Linear `768→512`, LayerNorm, GELU.
- Gate input: concatenated 1,024-dimensional projected branches.
- Gate: Linear `1024→256`, GELU, Linear `256→2`, softmax.
- Weighted representation: `[2g_c c ; 2g_v v]`, 1,024 dimensions.
- Final head: LayerNorm, dropout `0.2`, Linear `1024→4`.
- Auxiliary heads: separate Linear `512→4` CNN and ViT classifiers during training.
- Verified total parameters: `91,211,918`.
- The gate has `262,914` learned parameters and was verified to vary across input images.

Gate mass is utilization, not causal attribution. The gate is dynamic, not fixed at `0.55/0.45`, `0.15/0.85`, or any manually selected value.

## 5. Dataset and duplicate-clean partition

- Aggregate image pool: `7,023` JPEG images.
- Labels: glioma, meningioma, no-tumour, pituitary.
- Local files contain no DICOM headers, EXIF patient identifiers, study identifiers or series identifiers.
- Duplicate audit outcome: `5,648` image-similarity clusters.
- Redundant images removed: `1,375`.
- Duplicate-clean training images: `4,800`.
- Duplicate-clean testing images: `848`.
- Test class counts: glioma `243`, meningioma `228`, no-tumour `116`, pituitary `261`.
- Recorded split seed: `42`.
- Recorded perceptual-hash Hamming threshold: `8`.

The cleaning procedure treats image similarity. It does not identify patients and cannot support a patient-independent claim.

## 6. Training protocol

- Image size: `224×224`.
- Normalization: channel mean `0.5`, standard deviation `0.5`.
- Training augmentation: RandomResizedCrop scale `0.8–1.0`, horizontal flip, rotation up to `15°`, brightness jitter `0.2`, contrast jitter `0.2`.
- Optimizer: AdamW.
- Learning rate: `1e-4`.
- Backbone learning-rate multiplier: `0.1`.
- Weight decay: `0.05`.
- Batch size: `16`.
- Label smoothing: `0.1`.
- Maximum epochs: `15`.
- Early-stopping patience: `4`.
- Auxiliary-loss weight: `0.3`.
- EMA enabled in the full configuration.
- Horizontal-flip TTA enabled in the full configuration.
- Seeds: `42`, `43`, `44`.
- External labels were not used for training, validation, threshold selection or model selection.

## 7. Verified performance

| Evaluation | Images | Accuracy | Macro-F1 | Meaning |
|---|---:|---:|---:|---|
| Conventional/original held-out image split | 1,052 | 99.14 ± 0.10% | 99.12 ± 0.09% | secondary internal result |
| Duplicate-clean held-out image split | 848 | 98.27 ± 0.07% | 98.28 ± 0.04% | primary internal result |
| Duplicate-disjoint cross-source subset | 373 | 69.53 ± 1.98% | 68.69 ± 2.21% | source-shift stress test |
| BRAIN-META XGBoost, paper reported | 1,311 | 97.10% | 96.94% | contextual value only |

The AGMF-Net `±` values are sample SD across three training seeds. Do not label them as confidence intervals.

### Duplicate-clean aggregate probability metrics

- balanced accuracy: approximately `98.20%`;
- macro ROC-AUC: approximately `0.99947`;
- macro average precision: approximately `0.99854`;
- negative log-likelihood: approximately `0.11875`;
- multiclass Brier score: approximately `0.03349`;
- ECE with 15 equal-width bins: approximately `0.07518`.

Verify exact rounding from `fusion_v2_metrics.json` before use.

### Cross-source aggregate probability metrics

- balanced accuracy: approximately `74.49%`;
- macro ROC-AUC: approximately `0.98492`;
- macro average precision: approximately `0.96637`;
- NLL: approximately `0.74927`;
- Brier score: approximately `0.41920`;
- ECE: approximately `0.16176`.

## 8. Ablation evidence

| Model | Accuracy mean ± seed SD | Macro-F1 mean ± seed SD | Accuracy change vs full | Correct interpretation |
|---|---:|---:|---:|---|
| full AGMF-Net | 98.27 ± 0.07% | 98.28 ± 0.04% | 0.0000 pp | reference |
| no gate | 98.27 ± 0.18% | 98.23 ± 0.13% | 0.0000 pp | no detectable accuracy benefit under this protocol |
| no multi-scale taps | 98.23 ± 0.20% | 98.19 ± 0.21% | −0.0393 pp | no detectable accuracy benefit under this protocol |
| no auxiliary loss | 97.72 ± 0.48% | 97.73 ± 0.39% | −0.5503 pp | suggestive but inconclusive |
| no EMA | 98.31 ± 0.30% | 98.35 ± 0.24% | +0.0393 pp | no detectable accuracy benefit under this protocol |
| CNN branch only | 95.68 ± 0.34% | 95.90 ± 0.40% | −2.5943 pp | ViT contribution supported; 3/3 corrected-significant seed comparisons |
| ViT branch only | 98.03 ± 0.14% | 97.96 ± 0.16% | −0.2358 pp | CNN contribution suggestive but inconclusive |
| no TTA | 98.19 ± 0.25% | 98.21 ± 0.19% | −0.0786 pp | no detectable accuracy benefit under this protocol |

The ablation family contains 21 seed-specific exact McNemar tests. Holm correction was applied across all 21. Predictions were paired on the same 848 image filenames and labels. The 2,544 seed-image predictions were never treated as independent samples.

## 9. Cross-source class failure

The 373 duplicate-disjoint images contain 52 glioma, 122 meningioma, 88 no-tumour and 111 pituitary images. Mean pituitary recall is approximately 13.81%, while no-tumour recall is 100%. Across seeds 42–44, the 111 pituitary images yield 333 repeated predictions; 260 of these predictions are no-tumour (78.1%). They are repeated predictions of the same images, not 333 independent patients.

Permitted interpretation: cross-source degradation was substantial and class-specific.  
Prohibited interpretation: the model is robust across hospitals or scanners.

## 10. BRAIN-META comparison content

| Characteristic | BRAIN-META | AGMF-Net | Comparison boundary |
|---|---|---|---|
| Overall design | ten CNN–ViT base learners plus meta-learning | one EfficientNet-B0–ViT-B/16 feature-level hybrid | architectural comparison |
| Fusion | softmax prediction stacking | projected feature fusion with adaptive gate | different fusion levels |
| Meta-learner | logistic regression and XGBoost | none | do not describe AGMF-Net as stacking |
| Preprocessing | resize, normalization and CLAHE reported | resize, tensor conversion, mean/std 0.5; no CLAHE | protocol differs |
| Reported split | 5,712 training / 1,311 testing; paper describes 80/20 | conventional `n=1052` and duplicate-clean `n=848` tests | not the same split |
| Best reported accuracy | 97.10% | 99.14 ± 0.10% conventional; 98.27 ± 0.07% duplicate-clean | contextual values only |
| Best reported F1 | 96.94% | 99.12 ± 0.09% conventional; 98.28 ± 0.04% duplicate-clean | contextual values only |
| Multiple seeds | not established from available paper evidence | seeds 42, 43, 44 | report only verified facts |
| Duplicate-clean test | not reported | yes | image-level, not patient-level |
| Calibration | not a headline analysis | NLL, Brier, ECE, reliability | estimator documented |
| Controlled ablations | base-model comparisons reported; not AGMF-Net component controls | seven controlled variants | different study questions |
| Cross-source test | not established in supplied paper | 373 duplicate-disjoint images | only one additional source |
| Explainability | Grad-CAM on CNN components | CNN-path Grad-CAM++ for fused score | neither establishes clinical localization |

Never write “AGMF-Net outperforms BRAIN-META” without immediately qualifying that this is an uncontrolled reported-value comparison. Prefer: “AGMF-Net’s reported values were numerically higher under a different evaluation protocol.”

## 11. Assumptions and limitations addressed by the project

| Issue | Treatment in AGMF-Net project | Residual limitation |
|---|---|---|
| exact/near image duplication | pooled duplicate audit and duplicate-clean repartition | similarity threshold is operational and does not identify patients |
| seed sensitivity | three independently trained seeds and sample SD | three seeds provide limited variability information |
| component attribution | one-component controls and branch-only models | power is limited for small effects |
| paired comparison | exact filename/label alignment and seed-specific McNemar tests | applies only to the 848-image clean test |
| multiple testing | Holm correction across all 21 tests | correction reduces power |
| class imbalance | macro metrics, balanced accuracy and per-class tables | does not recreate clinical prevalence |
| confidence/calibration | NLL, Brier, ECE, reliability diagrams | descriptive; no prospective calibration |
| source shift | duplicate-disjoint cross-source evaluation | one additional collection, unknown acquisition metadata |
| result traceability | prediction CSV hashes, checkpoint hashes and manifests | depends on preserving the artifact package |

## 12. Assumptions and limitations not resolved

1. Patient, study and series identity are unavailable.
2. Patient-independent splitting cannot be verified.
3. Scanner, sequence, institution and acquisition metadata are unavailable.
4. The benchmark contains 2D JPEG exports rather than complete DICOM studies.
5. No prospective or multicentre clinical validation was performed.
6. No clinical workflow, reader study or patient-outcome evaluation was performed.
7. No verified lesion masks or bounding boxes exist for quantitative localization.
8. Grad-CAM++ does not spatially explain the ViT branch or the full TTA average.
9. Gate mass is not a causal explanation of branch importance.
10. Only three training seeds were evaluated.
11. The external analysis uses one additional source and 373 images.
12. BRAIN-META was not reproduced on the identical split and its predictions are unavailable.
13. Non-significant ablations do not establish equivalence or absence of utility.
14. A high softmax value does not establish diagnostic correctness.

## 13. Claims language

### Allowed

- AGMF-Net uses adaptive gated, multi-scale CNN–ViT feature fusion.
- AGMF-Net achieved the verified conventional and duplicate-clean image-level results.
- Duplicate-disjoint evaluation revealed substantial cross-source degradation.
- The ViT branch contribution was supported by consistent paired accuracy evidence.
- Auxiliary-loss and CNN-branch contributions were suggestive but inconclusive.
- Gate mass measures input-dependent branch utilization.

### Prohibited without new evidence

- clinically ready;
- patient-independent split;
- free from all leakage;
- robust across hospitals, scanners or populations;
- statistically superior to BRAIN-META;
- every architectural component improves accuracy;
- gate mass proves causal branch importance;
- Grad-CAM++ proves tumour localization or medically correct reasoning.

## 14. Required figures already available

The following have verified PNG and PDF versions under `publication_results/manuscript_draft/figures/`:

1. `figure_1_agmf_net_architecture`;
2. `figure_2_ablation_accuracy`;
3. `figure_3_clean_confusion`;
4. `figure_4_clean_reliability`;
5. `figure_5_gate_mass`;
6. `figure_6_gradcam_examples`;
7. `figure_7_cross_source_confusion`.

The Grad-CAM++ examples were selected deterministically as the first filename-ordered clean-test image from each class, not chosen after inspecting attractive heatmaps.

## 15. Placeholders that must remain until authors provide facts

- author names and ORCID identifiers;
- affiliations and correspondence address;
- supervisor/co-author contributions;
- target journal;
- institutional ethics/IRB determination;
- funding statement;
- competing-interest statement;
- repository URL and archival DOI;
- acknowledgements;
- exact submission date.

Do not infer or invent any of these.
