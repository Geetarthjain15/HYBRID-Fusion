# AGMF-Net — Supplementary material

Companion to *AGMF-Net: an adaptive gated multi-scale CNN–transformer fusion network with duplicate-aware and cross-source evaluation for brain MRI classification*.

Every value below is reproduced from an artifact in the project record. Source paths appear beside each table. The machine-readable claim ledger is `EVIDENCE_TRACEABILITY.csv`.

---

## S1. Architecture specification

Source: `publication_results/ablations/checkpoint_recovery/import_manifest.json`, verified by strict state-dictionary load of all three seed checkpoints.

| Component | Specification |
|---|---|
| Convolutional backbone | EfficientNet-B0, fully fine-tuned at 0.1 × base learning rate |
| Convolutional taps | stage 3 (40 ch), stage 5 (112 ch), final stage (1280 ch) |
| Pooling | global average pooling per tap, concatenated to 1432 dimensions |
| CNN projection | Linear 1432 → 512, LayerNorm, GELU |
| Transformer backbone | ViT-B/16, `google/vit-base-patch16-224` configuration |
| Transformer representation | 768-dimensional class token |
| ViT projection | Linear 768 → 512, LayerNorm, GELU |
| Gate input | concatenation of both projections, 1024 dimensions |
| Gate | Linear 1024 → 256, GELU, Linear 256 → 2, softmax |
| Fused representation | [2·g_c·c ; 2·g_v·v], 1024 dimensions |
| Classifier head | LayerNorm, dropout 0.2, Linear 1024 → 4 |
| Auxiliary heads | Linear 512 → 4 per branch, training only, weight 0.3 |
| Total parameters | 91,211,918 |
| Gate parameters | 262,914 |

The factor of 2 in the fused representation keeps the expected scale comparable to plain concatenation: since g_c + g_v = 1, the gated and ungated forms coincide exactly at g_c = g_v = 0.5.

### S1.1 Verification that the reference results were produced with the gate active

Because the reference training logs do not record the gating flag, the gate's status was established from the checkpoints themselves.

| Seed | State-dict keys | Missing | Unexpected | Shape mismatches | Probe mean g_c | 848 predictions reproduced | Max probability difference |
|---|---:|---:|---:|---:|---:|---|---:|
| 42 | 578 | 0 | 0 | 0 | 0.2808 | yes | 4.8 × 10⁻⁶ |
| 43 | 578 | 0 | 0 | 0 | 0.3006 | yes | 3.9 × 10⁻⁶ |
| 44 | 578 | 0 | 0 | 0 | 0.3369 | yes | 4.0 × 10⁻⁶ |

The gated and ungated forward paths are the same function only at g_c = 0.5. The observed probe mass is far from 0.5, and evaluating through the gated path reproduces every recorded reference prediction, so the reference results were produced with the gate active.

Note that the probe means above (0.28–0.34) were measured on the recovery probe batch, whereas the full-test-set means in S5 (0.085–0.115) are over all 848 clean-test images. Both are reported as measured; they are different samples.

---

## S2. Training protocol

Source: module constants in `src/train_ablation.py`, read from the frozen source file.

| Setting | Value |
|---|---|
| Input size | 224 × 224 |
| Normalisation | channel mean 0.5, standard deviation 0.5 |
| Training augmentation | RandomResizedCrop scale 0.8–1.0; horizontal flip; rotation ≤ 15°; brightness jitter 0.2; contrast jitter 0.2 |
| Optimiser | AdamW |
| Base learning rate | 1e-4 |
| Backbone learning-rate multiplier | 0.1 |
| Weight decay | 0.05 |
| Batch size | 16 |
| Label smoothing | 0.1 |
| Maximum epochs | 15 |
| Early-stopping patience | 4 |
| Validation fraction | 0.15 |
| Validation split seed | 42 |
| Auxiliary loss weight | 0.3 |
| EMA | enabled in the full configuration |
| Test-time augmentation | horizontal flip, full configuration |
| Training seeds | 42, 43, 44 |

No contrast-limited adaptive histogram equalisation is applied. External labels were never used for training, validation, early stopping, threshold selection or model selection.

---

## S3. Dataset partition

Source: `dataset_clean/split_report.json` and `results/cross_source.json`.

| Quantity | Value |
|---|---:|
| Pooled images | 7,023 |
| Perceptual hash | 256-bit difference hash |
| Near-duplicate threshold | Hamming ≤ 8 |
| Similarity clusters | 5,648 |
| Redundant images removed | 1,375 |
| Duplicate-clean training images | 4,800 |
| Duplicate-clean test images | 848 |
| Split seed | 42 |
| External collection | 499 images |
| External images with a near-duplicate in training | 126 |
| Duplicate-disjoint external images | 373 |

Duplicate-clean test class counts: glioma 243, meningioma 228, no-tumour 116, pituitary 261.
Duplicate-disjoint class counts: glioma 52, meningioma 122, no-tumour 88, pituitary 111.

---

## S4. Full metric panel

Source: `publication_results/metrics/fusion_v2_metrics.json`. Means over seeds 42, 43, 44.

| Metric | Duplicate-clean (n = 848) | Duplicate-disjoint cross-source (n = 373) |
|---|---:|---:|
| Accuracy | 0.98270 | 0.69526 |
| Macro-F1 | 0.98276 | 0.68687 |
| Balanced accuracy | 0.98202 | 0.74489 |
| Macro ROC-AUC (one-vs-rest) | 0.99947 | 0.98492 |
| Macro average precision (one-vs-rest) | 0.99854 | 0.96637 |
| Negative log-likelihood | 0.11875 | 0.74927 |
| Multiclass Brier score | 0.03349 | 0.41920 |
| Expected calibration error (15 bins) | 0.07518 | 0.16176 |

Average precision is the step-wise mean of precision at each positive. It is not the trapezoidal area under an interpolated precision–recall curve, and the two are computed differently.

Macro ROC-AUC falls by only 0.015 under source shift while accuracy falls by 28.74 percentage points. Ranking quality therefore degrades far less than decision quality, which is why source-specific recalibration is identified as a concrete avenue in the main text.

### S4.1 Per-class recall

| Class | Support (clean) | Recall, clean | Support (cross-source) | Recall, cross-source | Change |
|---|---:|---:|---:|---:|---:|
| Glioma | 243 | 98.22% | 52 | 98.08% | −0.14 pp |
| Meningioma | 228 | 97.37% | 122 | 86.07% | −11.30 pp |
| No tumour | 116 | 97.99% | 88 | 100.00% | +2.01 pp |
| Pituitary | 261 | 99.23% | 111 | 13.81% | −85.42 pp |

Recall for the two classes that hold up (glioma, no-tumour) is essentially unchanged, while meningioma loses 11 points and pituitary loses 85. An aggregate accuracy figure averages these together and conceals the asymmetry.

---

## S5. Gate behaviour

Source: `runs_v2_clean/fusion_v2_seed{42,43,44}/gates_test.csv`, 848 rows each.

| Seed | Mean g_c | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| 42 | 0.1002 | 0.0632 | 0.0312 | 0.5889 |
| 43 | 0.0849 | 0.0478 | 0.0324 | 0.4928 |
| 44 | 0.1149 | 0.0693 | 0.0269 | 0.7539 |

The per-image range spans more than an order of magnitude within each seed, which is the evidence that the gate is input-dependent rather than a fixed weighting. Gate mass measures branch utilisation. It does not establish causal branch importance, and the ablation in S6 is the more direct test of contribution.

---

## S6. Ablation statistics

Source: `publication_results/ablations/final_analysis/`. Family of 21 seed-specific exact two-sided McNemar tests, Holm-corrected across all 21. Four tests were significant before correction and three after.

| Variant | Seed | b (full right, ablation wrong) | c (full wrong, ablation right) | Discordant | p unadjusted | p Holm-adjusted |
|---|---:|---:|---:|---:|---:|---:|
| cnn_branch_only | 42 | 31 | 6 | 37 | 4.13e-05 | 8.66e-04 |
| cnn_branch_only | 43 | 25 | 5 | 30 | 3.25e-04 | 6.50e-03 |
| cnn_branch_only | 44 | 30 | 9 | 39 | 1.07e-03 | 2.02e-02 |
| no_auxiliary_loss | 42 | 10 | 2 | 12 | 3.86e-02 | 6.94e-01 |
| no_auxiliary_loss | 43 | 10 | 5 | 15 | 3.02e-01 | 1.00 |
| no_auxiliary_loss | 44 | 4 | 3 | 7 | 1.00 | 1.00 |
| vit_branch_only | 42 | 3 | 2 | 5 | 1.00 | 1.00 |
| vit_branch_only | 43 | 7 | 4 | 11 | 5.49e-01 | 1.00 |
| vit_branch_only | 44 | 5 | 3 | 8 | 7.27e-01 | 1.00 |
| no_gate | 42 | 1 | 1 | 2 | 1.00 | 1.00 |
| no_gate | 43 | 4 | 3 | 7 | 1.00 | 1.00 |
| no_gate | 44 | 1 | 2 | 3 | 1.00 | 1.00 |
| no_multiscale | 42 | 4 | 5 | 9 | 1.00 | 1.00 |
| no_multiscale | 43 | 3 | 4 | 7 | 1.00 | 1.00 |
| no_multiscale | 44 | 7 | 4 | 11 | 5.49e-01 | 1.00 |
| no_ema | 42 | 0 | 1 | 1 | 1.00 | 1.00 |
| no_ema | 43 | 7 | 5 | 12 | 7.74e-01 | 1.00 |
| no_ema | 44 | 1 | 3 | 4 | 6.25e-01 | 1.00 |
| no_tta | 42 | 1 | 0 | 1 | 1.00 | 1.00 |
| no_tta | 43 | 3 | 1 | 4 | 6.25e-01 | 1.00 |
| no_tta | 44 | 1 | 2 | 3 | 1.00 | 1.00 |

Only the discordant pairs carry information about a difference. Where the discordant count is very small — one or two pairs for several variants — the test has almost no power, and a non-significant result there says little about the underlying effect.

### S6.1 Decision rule for the verdicts

Verdicts were assigned by a rule fixed before the comparison and using only prespecified quantities: the direction of the per-seed accuracy differences, the Holm-adjusted p-values over the 21-test family, and the paired image-level bootstrap intervals. No effect-size threshold takes part in any verdict.

| Verdict | Assigned when |
|---|---|
| supported by consistent accuracy evidence | adverse at all three seeds **and** all three Holm-adjusted tests significant at 0.05 |
| suggestive but inconclusive | adverse at all three seeds but corrected evidence insufficient at some seed |
| no detectable accuracy benefit under this protocol | not adverse at all three seeds, no Holm-adjusted test significant, no paired interval excluding zero |
| inconclusive | direction not consistently adverse yet some corrected test or interval fires |

The third category records an absence of detectable evidence. It is not equivalence, it does not show a component has zero effect, and it does not show a component is unnecessary.

### S6.2 Paired bootstrap

Resampling unit: image. Stratified by true class. 10,000 iterations at a fixed recorded seed. One index set per iteration applied to both models so the pairing is preserved. These are image-level intervals within a single training seed; they are not patient-level intervals, and they are not the seed standard deviation. The two quantities are reported separately throughout and never combined.

---

## S7. Reproduction

| Step | Command |
|---|---|
| Figures | `env_win/Scripts/python.exe src/publication/make_paper_figures.py` |
| Tables and evidence ledger | `env_win/Scripts/python.exe src/publication/make_paper_tables.py` |
| Word manuscript | `env_win/Scripts/python.exe src/publication/make_paper_docx.py` |

Each script reads only from verified artifacts and writes only into `publication_results/final_paper_claude/`. No script trains a model, modifies a checkpoint, or alters any prior result.

---

## S8. Items requiring author input

The following cannot be derived from any repository artifact and are left as explicit placeholders in the manuscript: author names and ORCID identifiers; affiliations and corresponding address; target journal; institutional ethics or IRB determination; funding statement; competing-interest statement; repository URL and archival DOI; acknowledgements; CRediT contribution statement; submission date; and training/inference wall-clock timings, which were not recorded in a verifiable form.
