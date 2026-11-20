# AGMF-Net: an adaptive gated multi-scale CNN–transformer fusion network with duplicate-aware and cross-source evaluation for brain MRI classification

**Authors:** [AUTHOR INPUT REQUIRED — names and ORCID identifiers]

**Affiliations:** [AUTHOR INPUT REQUIRED — institutional affiliations]

**Corresponding author:** [AUTHOR INPUT REQUIRED — name, address, e-mail]

**Target venue:** [AUTHOR INPUT REQUIRED]

---

## Highlights

- AGMF-Net fuses multi-scale EfficientNet-B0 features with a ViT-B/16 class token through a learned, image-dependent two-way gate rather than fixed concatenation.
- On a duplicate-cleaned held-out split of 848 images the model reached 98.27 ± 0.07% accuracy and 98.28 ± 0.04% macro-F1 across three training seeds.
- On 373 duplicate-disjoint images from a second collection accuracy fell to 69.53 ± 1.98%, a drop of 28.74 percentage points.
- Seven protocol-matched ablations with Holm-corrected paired tests showed that only removal of the transformer branch produced a resolved accuracy loss; the gate, multi-scale taps, EMA and TTA produced no detectable accuracy benefit under this protocol.
- The cross-source failure was class-specific: pituitary recall fell to 13.81% while no-tumour recall remained at 100%.

## Graphical abstract

Figure 2 (evaluation map) together with Figure 3 (regime collapse) is proposed as the graphical abstract: it shows in one view which images each reported number is computed on, and how performance changes between those settings.

---

## Abstract

Automated classification of brain tumours from structural MRI is usually reported as a single accuracy figure on a held-out split of one public collection. Two practices weaken that figure: near-duplicate images shared between training and test partitions, and the absence of any evaluation on data from a different source. We describe **AGMF-Net (Adaptive Gated Multi-scale Fusion Network)**, a single feature-level hybrid that combines convolutional and transformer representations, and we pair it with an evaluation protocol designed to expose both problems.

AGMF-Net pools EfficientNet-B0 activations from three depths (40, 112 and 1280 channels, concatenated to 1432 dimensions), projects them to 512 dimensions, projects the 768-dimensional ViT-B/16 class token to the same width, and combines the two through a learned gate whose two outputs sum to one and vary from image to image. Auxiliary classifiers attached to each branch keep both pathways discriminative during training. The network has 91,211,918 parameters, of which the gate accounts for 262,914.

We evaluated the model on 7,023 pooled images under three settings. On the conventional held-out split (n = 1,052) accuracy was 99.14 ± 0.10% and macro-F1 99.12 ± 0.09%. After a perceptual-hash audit removed 1,375 redundant images and the data were repartitioned, accuracy on the duplicate-clean test set (n = 848) was 98.27 ± 0.07% and macro-F1 98.28 ± 0.04%. On 373 duplicate-disjoint images drawn from a second collection, accuracy fell to 69.53 ± 1.98% and macro-F1 to 68.69 ± 2.21%. All intervals are sample standard deviations across three training seeds and are not confidence intervals.

Seven component ablations were trained under an identical protocol and compared against the full model on the same 848 images using exact paired McNemar tests, with Holm correction applied across all 21 seed-specific comparisons. Only removal of the transformer branch produced a loss that survived correction at every seed (−2.59 percentage points, three of three tests). Removal of the auxiliary losses (−0.55 pp) and of the convolutional branch (−0.24 pp) was adverse at every seed but did not survive correction. Removal of the gate, the multi-scale taps, EMA or TTA produced no detectable accuracy benefit under this protocol; we report these as absences of detectable evidence, not as demonstrations of equivalence.

The cross-source degradation was strongly class-specific. Across three seeds the 111 duplicate-disjoint pituitary images produced 333 predictions, of which 260 (78.1%) were assigned to the no-tumour class, while no-tumour recall remained at 100%. These are repeated predictions of the same 111 images and are not 333 independent cases. Expected calibration error rose from 0.0752 internally to 0.1618 under source shift.

The contribution is therefore twofold: a specific adaptive fusion design, and an evaluation discipline showing that a high benchmark score on this dataset family can coexist with substantial, class-specific fragility under a change of image source. The work is an image-level research demonstrator. It is not a medical device, the splits are not patient-independent, and no clinical validation was performed.

**Keywords:** brain MRI; image-level classification; convolutional–transformer fusion; adaptive gating; near-duplicate leakage; cross-source evaluation; calibration; ablation study

---

## Specifications table

| Item | Entry |
|---|---|
| Subject area | Computer Science |
| More specific subject area | Machine learning for medical image analysis |
| Name of the method | AGMF-Net: Adaptive Gated Multi-scale Fusion Network for brain MRI image classification |
| Name and reference of original method | Napa KK, Murugan S, Murugan JS, Jayanthi A. BRAIN-META: a reproducible CNN–vision transformer meta-ensemble pipeline for explainable brain tumor classification. MethodsX 2026;16:103769. https://doi.org/10.1016/j.mex.2025.103769 |
| Resource availability | Source code, prediction files, checkpoint digests and result manifests are retained with the project record. [AUTHOR INPUT REQUIRED — repository URL and archival DOI] |
| Data | Publicly available brain tumour MRI collection (see Dataset section). [AUTHOR INPUT REQUIRED — confirm redistribution terms for the target venue] |

---

## Background

Magnetic resonance imaging is the principal non-invasive modality for characterising intracranial masses, and the volume of imaging now produced makes computational triage attractive. Over the past decade, convolutional neural networks have become the default tool for classifying tumour type from 2D structural slices, and more recently vision transformers have been added on the argument that self-attention captures relationships between distant regions that a convolutional kernel cannot.

The two families carry genuinely different inductive biases. Convolutions impose locality and translation equivariance, which suits texture, edge and boundary cues. Self-attention imposes neither, and can relate a lesion to midline structures or ventricular geometry elsewhere in the slice. Combining them is therefore reasonable in principle, and a large literature now reports hybrid CNN–transformer classifiers for brain MRI.

What that literature reports less often is how much of the resulting accuracy is attributable to the combination, and whether the accuracy survives contact with data from elsewhere. Two specific problems recur.

The first is near-duplicate leakage. Public brain MRI collections are frequently assembled from several upstream sources and redistributed after resizing, re-encoding or augmentation. The same underlying slice can therefore appear more than once in a pooled set, sometimes in both the training and the test partition. A model that has memorised such an image is scored as though it had generalised to it. The effect is invisible in a conventional accuracy figure and is not removed by stratified splitting, because stratification balances class frequency, not image identity.

The second is the absence of a source-shift test. A held-out split drawn from a single collection measures interpolation within one acquisition and post-processing regime. It does not measure what happens when scanner, sequence, reconstruction or export pipeline changes, which is precisely the condition under which a deployed classifier would operate.

Neither problem is addressed by reporting a higher accuracy. Both require the evaluation itself to be constructed differently. The work described here therefore treats the evaluation design as part of the contribution rather than as a reporting formality.

---

## Method details

### Motivation for the architecture

An earlier version of the model in this project concatenated the final 1,280-dimensional EfficientNet representation with the 768-dimensional ViT class token, producing a 2,048-dimensional vector that fed a small fusion head, with the convolutional backbone and most transformer layers frozen. Inspecting that design exposed four specific weaknesses.

First, the convolutional pathway was only partially fine-tuned even though a fully fine-tuned convolutional control scored higher in isolation, so the branch was handicapped before fusion began. Second, only the terminal convolutional output was used, discarding the intermediate scales at which small lesions are often best represented. Third, concatenating a 1,280-dimensional vector with a 768-dimensional one gives the wider branch more influence on the downstream linear layer purely because it is wider, with no mechanism to correct that. Fourth, neither branch carried an independent training objective, so nothing prevented one pathway from contributing little while the head learned to rely on the other.

AGMF-Net addresses these four points directly: full fine-tuning at a reduced backbone learning rate, pooling from multiple convolutional depths, projection of both branches to a common width before they are combined, a learned gate that sets the mixing per image, and auxiliary classifiers that require each branch to remain discriminative on its own.

### AGMF-Net architecture

Figure 1 shows the forward path. An input image of 224 × 224 × 3 is passed to both branches.

The convolutional branch uses EfficientNet-B0. Activations are taken from stage 3 (40 channels), stage 5 (112 channels) and the final stage (1,280 channels). Each tap is reduced by global average pooling over its spatial dimensions and the three pooled vectors are concatenated, giving 1,432 dimensions. A linear layer maps this to 512 dimensions, followed by layer normalisation and a GELU non-linearity.

The transformer branch uses ViT-B/16 in the `google/vit-base-patch16-224` configuration. The 768-dimensional class-token embedding is projected to 512 dimensions through a linear layer, layer normalisation and GELU. Projecting both branches to the same width is deliberate: it removes the width asymmetry that would otherwise bias the fusion, so the gate compares comparable quantities.

Writing **c** and **v** for the two projected 512-dimensional vectors, the gate receives their concatenation and produces two scalars:

$$[g_c,\; g_v] = \mathrm{softmax}\big(W_2\,\phi(W_1[\mathbf{c};\mathbf{v}] + b_1) + b_2\big)$$

where $W_1 \in \mathbb{R}^{256\times 1024}$, $W_2 \in \mathbb{R}^{2\times 256}$ and $\phi$ is GELU. By construction $g_c + g_v = 1$. The fused representation is

$$\mathbf{z} = [\,2 g_c\,\mathbf{c}\;;\;2 g_v\,\mathbf{v}\,] \in \mathbb{R}^{1024}$$

The factor of two keeps the expected scale of **z** comparable to plain concatenation, since the two gate values sum to one; at $g_c = g_v = 0.5$ the gated and ungated forms coincide exactly. Classification applies layer normalisation, dropout of 0.2 and a linear map to four logits.

During training two auxiliary linear classifiers, each mapping 512 dimensions to four classes, are attached to **c** and **v** respectively. The training objective is

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda\big[\mathcal{L}_{\text{CE}}(\hat{y}_{c}, y) + \mathcal{L}_{\text{CE}}(\hat{y}_{v}, y)\big], \qquad \lambda = 0.3$$

with label smoothing of 0.1 applied to every term. The auxiliary heads are discarded at inference.

The complete network has 91,211,918 parameters. The gate module accounts for 262,914 of these. Two properties of the gate should be stated precisely, because they are easy to overstate. It is a learned module whose output varies with the input, which we verify empirically in the Results. It is not a fixed or manually chosen ratio. Equally, the gate mass measures how much of each branch's projected representation enters the fused vector — that is, branch utilisation — and does not by itself establish that a branch is causally responsible for a prediction.

### Dataset and duplicate control

The pooled image set comprises 7,023 JPEG images in four classes: glioma, meningioma, no-tumour and pituitary. The class order is fixed as (glioma, meningioma, notumor, pituitary) throughout all artifacts and code.

The files are 2D exports and contain no DICOM headers and no EXIF patient, study or series identifiers. This has a consequence that constrains the entire study and we state it once, plainly: **no patient, study or series grouping key is available, so no split in this work can be verified as patient-independent, and none is claimed.**

Because pooled public collections commonly contain re-encoded copies of the same slice, we audited the pool for near-duplicates before partitioning. Each image was reduced to a 256-bit difference hash and image pairs within a Hamming distance of 8 were treated as near-duplicates. Connected components of this relation were formed, yielding 5,648 clusters from 7,023 images, and one representative was retained per cluster. This removed 1,375 redundant images. The retained images were then repartitioned with split seed 42 into 4,800 training and 848 test images. The test class counts are glioma 243, meningioma 228, no-tumour 116 and pituitary 261.

This procedure controls *image* redundancy. It does not identify patients, and two distinct slices from the same patient will not be linked by it. Duplicate cleaning therefore reduces one identifiable leakage pathway without eliminating leakage in general.

A second, separate collection of 499 images was used as an external set. Applying the same hash criterion against the training pool showed that 126 of these images have a near-duplicate in training and 373 do not. The 373 duplicate-disjoint images constitute the cross-source stress test. The remaining 126 are reported only as part of the "external, all images" figure and are never presented as unseen data. Figure 2 shows which images each reported number is computed on.

External labels were never used for training, validation, early stopping, threshold selection or model selection.

### Training and evaluation protocol

Images are resized to 224 × 224 and normalised with channel mean 0.5 and standard deviation 0.5. Training augmentation is random resized cropping with scale 0.8–1.0, horizontal flipping, rotation up to 15°, and brightness and contrast jitter of 0.2. No contrast-limited adaptive histogram equalisation is applied; this differs from the reference pipeline and is noted in the comparison section.

Optimisation uses AdamW with a base learning rate of 1e-4, a backbone learning-rate multiplier of 0.1, weight decay of 0.05, batch size 16 and label smoothing of 0.1. Training runs for at most 15 epochs with early stopping on validation accuracy and patience 4. A validation partition of 15% is carved from the training set with a fixed seed. In the full configuration an exponential moving average of the weights is maintained and evaluated, and horizontal-flip test-time augmentation is applied at inference.

Every experiment was run with seeds 42, 43 and 44. All reported ± values are sample standard deviations (ddof = 1) across those three seeds. Three seeds provide limited information about variability, and we do not treat the seed spread as a confidence interval or use it in a significance test.

### Statistical analysis

Metrics were recomputed from the per-image prediction files rather than read from training logs. For each model and seed we computed accuracy, macro precision, macro recall, macro-F1, balanced accuracy, per-class precision, recall, specificity and F1, the confusion matrix, negative log-likelihood, the multiclass Brier score, expected calibration error over 15 equal-width bins applied to the maximum predicted probability, one-vs-rest ROC-AUC, and one-vs-rest average precision. Average precision is the step-wise mean of precision at each positive; it is not the trapezoidal area under an interpolated precision–recall curve, and the two are not interchangeable. Where a quantity is mathematically undefined for a class it is reported as unavailable rather than as zero.

Component comparisons are paired. Each ablation is compared with the full model at the same seed on the same 848 images, matched by exact filename, with the requirement that the true label agree with the split manifest for every image. This produces 7 variants × 3 seeds = 21 comparisons. For each we computed the 2 × 2 table of joint correctness and an exact two-sided McNemar test on the discordant pairs, using the binomial form rather than the chi-square approximation because the discordant counts are small. Holm correction was applied across all 21 tests as a single family. We did not correct across seven while selecting or pooling one p-value per variant, because that would be a smaller and more permissive family obtained by discarding evidence.

Uncertainty on each paired difference was estimated by a bootstrap over images, stratified by true class, with 10,000 iterations at a fixed recorded seed. One resampling index set per iteration was applied to both models so that the pairing is preserved. The resampling unit is an image. Because no patient identifiers exist, no patient-level uncertainty can be estimated and none is reported. The seed standard deviation and the bootstrap interval measure different quantities and are never combined.

The 2,544 predictions that arise from 848 images scored under three seeds are repeated measurements of the same images. They are never treated as 2,544 independent observations.

---

## Method validation

### Internal performance

Table 1 reports the primary results. Figure 3 shows the same values with individual seeds visible.

**Table 1.** AGMF-Net performance by evaluation setting. Values are mean ± sample standard deviation across seeds 42, 43 and 44.

| Evaluation setting | Images | Accuracy (%) | Macro-F1 (%) | Balanced accuracy (%) |
|---|---:|---:|---:|---:|
| Original held-out split | 1,052 | 99.14 ± 0.10 | 99.12 ± 0.09 | — |
| **Duplicate-clean held-out split** | **848** | **98.27 ± 0.07** | **98.28 ± 0.04** | **98.20** |
| External collection, all images | 499 | 76.49 ± 1.52 | 70.61 ± 2.15 | — |
| Duplicate-disjoint cross-source subset | 373 | 69.53 ± 1.98 | 68.69 ± 2.21 | 74.49 |

The duplicate-clean result is the primary internal figure. The original-split and duplicate-clean numbers are computed on different image sets under different partitioning procedures and are not two measurements of the same quantity; the 0.87-point difference between them is consistent with the removal of near-duplicates but we did not run the controlled experiment that would be needed to attribute it.

Aggregate probabilistic quality on the duplicate-clean test set was: macro ROC-AUC 0.9995, macro average precision 0.9985, negative log-likelihood 0.1188, Brier score 0.0335 and expected calibration error 0.0752 over 15 bins.

### Cross-source performance

Accuracy on the 373 duplicate-disjoint images was 69.53 ± 1.98%, which is 28.74 percentage points below the duplicate-clean result. Probabilistic quality degraded in the same direction: negative log-likelihood rose from 0.1188 to 0.7493, the Brier score from 0.0335 to 0.4192, and expected calibration error from 0.0752 to 0.1618. Macro ROC-AUC remained high at 0.9849, which is worth noting: ranking quality degraded far less than decision quality, so a threshold or calibration adjusted on the new source might recover part of the loss. We did not attempt that, because doing so would require the external labels we deliberately withheld.

The degradation is not uniform across classes (Figure 8). Mean recall on the duplicate-disjoint subset was 98.08% for glioma, 86.07% for meningioma, 100% for no-tumour and 13.81% for pituitary. Across the three seeds the 111 duplicate-disjoint pituitary images produced 333 predictions, of which 260 (78.1%) were assigned to the no-tumour class. These 333 values are repeated predictions of the same 111 images and must not be read as 333 independent cases.

The direction of this error is the one that matters most for any triage application: the dominant failure is a tumour-bearing class being called no-tumour. We report it as an observed property of this model on this subset, not as a general property of the architecture.

### Component ablation

Seven variants were trained under the identical protocol, each altering exactly one component, and compared against the full model on the same images at the same seed. Table 2 and Figure 5 give the results.

**Table 2.** Ablation results on the duplicate-clean test set (848 images). Differences are observed means across three seeds. Holm correction is across all 21 seed-specific tests.

| Variant | Accuracy (%) | Macro-F1 (%) | Δ accuracy (pp) | Adverse at every seed | Holm-significant | Verdict |
|---|---:|---:|---:|---|---:|---|
| Full AGMF-Net | 98.27 ± 0.07 | 98.28 ± 0.04 | reference | — | — | reference |
| − ViT branch (CNN only) | 95.68 ± 0.34 | 95.90 ± 0.40 | −2.59 | yes | 3 / 3 | supported by consistent accuracy evidence |
| − auxiliary losses | 97.72 ± 0.48 | 97.73 ± 0.39 | −0.55 | yes | 0 / 3 | suggestive but inconclusive |
| − CNN branch (ViT only) | 98.03 ± 0.14 | 97.96 ± 0.16 | −0.24 | yes | 0 / 3 | suggestive but inconclusive |
| − test-time augmentation | 98.19 ± 0.25 | 98.21 ± 0.19 | −0.08 | no | 0 / 3 | no detectable accuracy benefit |
| − multi-scale taps | 98.23 ± 0.20 | 98.19 ± 0.21 | −0.04 | no | 0 / 3 | no detectable accuracy benefit |
| − adaptive gate | 98.27 ± 0.18 | 98.23 ± 0.13 | +0.00 | no | 0 / 3 | no detectable accuracy benefit |
| − exponential moving average | 98.31 ± 0.30 | 98.35 ± 0.24 | +0.04 | no | 0 / 3 | no detectable accuracy benefit |

Of the 21 seed-specific tests, four were significant before correction and three after. All three surviving tests belong to the CNN-only variant. Removing the transformer branch cost 2.59 percentage points, adverse at every seed, with all three paired bootstrap intervals excluding zero. This is the one component whose contribution is resolved by the present evidence.

Removing the auxiliary losses lowered accuracy at all three seeds with an observed mean of −0.55 pp, and the smallest unadjusted p-value was 0.0386; after Holm correction across the 21-test family this became 0.6943. Removing the convolutional branch was likewise adverse at all three seeds (−0.24 pp) without surviving correction. We record both as suggestive but inconclusive. A consistent direction across three seeds is worth reporting and is not sufficient evidence of benefit.

The remaining four components — gate, multi-scale taps, EMA and TTA — produced no detectable accuracy benefit under this protocol. The gate is the most consequential of these for the paper's own framing, and we state the result plainly: the observed mean accuracy difference on removing the gate was +0.00 pp, with per-seed differences of −0.12, 0.00 and +0.12 pp, discordant prediction counts of 2, 7 and 3 out of 848, all three Holm-adjusted p-values at 1.0, and no paired interval excluding zero.

This must be read correctly in both directions. It does not show that the gate is useless, that it has zero effect, or that it should be removed. An absence of detectable evidence on 848 images with three seeds is compatible with a real effect smaller than this design can resolve. Equally, it does not license describing adaptive gating as an empirically demonstrated accuracy improvement on this benchmark, and we do not describe it that way. The honest statement is that the benchmark used here is too close to saturation, and the seed count too small, to resolve differences of the size these components appear to produce.

### Gate behaviour

Figure 7 shows the distribution of CNN gate mass over the 848 clean-test images. Mean $g_c$ was 0.100, 0.085 and 0.115 for seeds 42, 43 and 44, with per-image values ranging from 0.027 to 0.754 across the three seeds. The distribution is unimodal, well away from 0.5, and clearly not degenerate at a point.

Two conclusions follow, and only two. The gate is genuinely input-dependent: its output varies substantially from image to image, so the fusion is not equivalent to a fixed weighting. And the transformer branch receives the larger share of the mass on most images, which is consistent with the ablation finding that removing the transformer branch is the costlier deletion.

What does not follow is any causal statement. A branch can receive high gate mass without being responsible for the decision, and the ablation is the more direct test of contribution. We therefore do not present gate mass as evidence of branch importance, and we do not present it as an explanation of individual predictions.

Separately, the gate's status in the reference configuration was verified rather than assumed: the trained checkpoints load into the gated architecture with all 578 state-dictionary entries matched and no missing, unexpected or mis-shaped tensors, and evaluating through the gated path reproduces every one of the 848 recorded reference predictions with a maximum probability difference of 4.8 × 10⁻⁶. Because the gated and ungated forward paths coincide only at a gate mass of exactly 0.5, and the observed probe mass is far from 0.5, this confirms that the reported reference results were produced with the gate active.

### Calibration

Figure 6 shows reliability curves for both regimes with the bin populations plotted beneath, since a reliability curve without bin counts can make a bin holding three images look like a trend.

Internally the model is overconfident in the mid-confidence range, with an expected calibration error of 0.0752. Under source shift this worsens to 0.1618, and negative log-likelihood and Brier score rise substantially. In practical terms, a high softmax value from this model on out-of-source data is a weaker indicator of correctness than the same value on in-source data.

These calibration comparisons are descriptive. No paired inferential procedure or multiplicity control was prespecified for NLL, Brier or ECE across models, and we did not add one after the fact, so we do not rank models on calibration or claim that any variant is better calibrated than another.

### Explainability demonstrator

Grad-CAM++ maps were generated from the final convolutional stage of the EfficientNet branch. Example images were selected deterministically as the first filename-ordered clean-test image of each class, before any heatmap was inspected, so the figure is not a selection of attractive cases.

The scope of this visualisation needs stating carefully. It reflects the sensitivity of the score to the final convolutional feature map of one branch. It does not represent the transformer pathway, which contributes the larger share of the gate mass on most images; it does not represent the test-time-augmentation average that produces the reported predictions; and it has not been compared against any expert annotation, because no verified lesion masks exist for this dataset. It is therefore a demonstrator of where the convolutional path is locally sensitive, not validated tumour localisation and not an account of the model's reasoning.

### Computational cost

[AUTHOR INPUT REQUIRED — training and inference wall-clock timings were not recorded in a form we could verify from the retained artifacts. We decline to state timings we cannot substantiate. If the target venue requires a computation-time table, these measurements should be re-run and recorded.]

Parameter count is verified: 91,211,918 total parameters, of which the gate accounts for 262,914.

---

## Comparison with the reference method

BRAIN-META (Napa et al., MethodsX 2026) reports 97.10% accuracy and 96.94% F1 on its own held-out partition of 1,311 images, using ten CNN–ViT base learners whose softmax outputs are stacked and combined by logistic-regression and XGBoost meta-learners.

AGMF-Net differs from that design in kind rather than degree. It is a single model that fuses at the feature level with a learned gate, and it contains no meta-learner and no prediction stacking; it should not be described as a stacking ensemble. The preprocessing pipelines also differ — BRAIN-META reports CLAHE contrast enhancement, which we do not apply.

Table 3 sets out the comparison with the protocol columns that decide whether the accuracy figures can be compared at all.

**Table 3.** Protocol-aware comparison. See `tables/BASE_PAPER_COMPARISON.csv` for the machine-readable version.

| Characteristic | BRAIN-META | AGMF-Net |
|---|---|---|
| Design | 10 CNN–ViT base learners + meta-learners | single feature-level hybrid with adaptive gate |
| Fusion level | softmax prediction stacking | projected feature fusion |
| Preprocessing | resize, normalise, CLAHE | resize, normalise (mean/std 0.5), no CLAHE |
| Test partition | 1,311 images | 1,052 (original) / 848 (duplicate-clean) |
| Duplicate control | not reported in the inspected paper | perceptual-hash audit, 1,375 images removed |
| Patient-level grouping | not reported in the inspected paper | not available; not claimed |
| Cross-source evaluation | not reported in the inspected paper | 373 duplicate-disjoint images |
| Multi-seed evaluation | not reported in the inspected paper | seeds 42, 43, 44 |
| Calibration reporting | not a headline analysis | NLL, Brier, ECE, reliability curves |
| Component ablations | base-learner comparisons reported | seven protocol-matched component controls |
| Explainability | Grad-CAM on CNN components | Grad-CAM++ on the CNN path |
| Accuracy | 97.10% | 99.14 ± 0.10% (original); 98.27 ± 0.07% (duplicate-clean) |
| Macro-F1 | 96.94% | 99.12 ± 0.09% (original); 98.28 ± 0.04% (duplicate-clean) |
| **Comparability status** | — | **contextual only** |

The comparability status is the operative field. AGMF-Net obtained 99.14 ± 0.10% accuracy under the original internal image-level protocol and 98.27 ± 0.07% after duplicate cleaning. BRAIN-META reports 97.10% accuracy under its own reported protocol. Because the datasets, partition construction, duplication controls and evaluation conditions are not established as equivalent, these figures provide context rather than a controlled head-to-head ranking.

We therefore make no claim of statistical superiority over BRAIN-META. Establishing one would require its per-image predictions, or a reproduction of that pipeline on the identical split; neither was available to us. Where entries above read "not reported in the inspected paper", that is a statement about what we could verify in the published article, not an assertion that the authors did not perform the analysis.

---

## Theoretical and methodological implications

The architectural result is narrower than the architecture's design intent, and saying so is more useful than obscuring it. Projecting both branches to a common width before fusion is well motivated, and the gate does behave as designed — it is learned, it varies per image, and it allocates most mass to the transformer branch. But on this benchmark, at this level of performance, removing the gate cost nothing measurable. The component that demonstrably matters is the transformer branch itself.

The methodological result is the more transferable one. A duplicate-clean partition, three seeds, protocol-matched ablations with multiplicity control, and a duplicate-disjoint cross-source subset together produce a picture that no single accuracy figure conveys: a model at 98.27% internally and 69.53% on 373 images from another source, with one class collapsing almost entirely while another is untouched. Reporting only the first number would not have been false, but it would have been misleading about deployment behaviour.

This also bears on how saturated benchmarks should be used for architecture research. With the full model at 98.27% and the reference seed spread at 0.07 pp, an ablation would need a large effect to be resolvable at three seeds after correcting across 21 tests. Four of our seven components produced no detectable effect. The reasonable inference is not that those components are worthless, but that this benchmark cannot adjudicate them, and that component claims made on similar benchmarks without paired testing and multiplicity control should be treated with caution.

---

## Assumptions, limitations and threats to validity

A structured version is provided in `tables/ASSUMPTIONS_LIMITATIONS.csv`. The principal items are as follows.

**Addressed or partly addressed.** Near-duplicate image leakage was reduced by a perceptual-hash audit and a duplicate-clean repartition, though this controls image redundancy and not patient identity. Single-run variability was partly addressed with three seeds. Component attribution was addressed with protocol-matched ablations and multiplicity control, subject to limited statistical power. Fixed branch weighting was replaced by a learned input-dependent gate. Accuracy-only reporting was expanded to macro and per-class metrics, calibration and paired uncertainty. Source-shift blindness was exposed — not solved — by the duplicate-disjoint evaluation.

**Not addressed.** Patient, study and series identifiers are unavailable, so patient-independent splitting cannot be verified and is not claimed. Scanner, sequence, institution and demographic metadata are unavailable, so no stratified analysis was possible. The data are 2D JPEG exports rather than complete DICOM studies, so volumetric and acquisition context is absent. No multi-institutional, prospective or reader-study validation was performed. No verified lesion masks exist, so localisation cannot be quantified. Cross-source generalisation remains weak, including the pituitary-to-no-tumour failure. Only three seeds and a single additional source of 373 images were evaluated. BRAIN-META was not reproduced on an identical split.

**Threats to validity.** The duplicate threshold (256-bit hash, Hamming ≤ 8) is an operational choice; a stricter or looser threshold would change the partition, and we did not test sensitivity to it. The cross-source subset comes from one collection, so its failure pattern may not generalise to other sources. Non-significant ablations reflect limited power as much as small effects. Finally, a softmax value from this model is not a probability of diagnostic correctness, and is measurably less trustworthy under source shift.

---

## Conclusion and future work

AGMF-Net combines multi-scale convolutional features with a transformer representation through a learned, image-dependent gate, reaching 98.27 ± 0.07% accuracy and 98.28 ± 0.04% macro-F1 on a duplicate-cleaned held-out set of 848 images across three seeds. Under controlled ablation, only removal of the transformer branch produced an accuracy loss that survived Holm correction at every seed; the gate, multi-scale taps, EMA and TTA produced no detectable accuracy benefit under this protocol, which we report as an absence of detectable evidence rather than as evidence of absence.

The finding we consider most important is the second one. On 373 duplicate-disjoint images from another source, accuracy fell to 69.53 ± 1.98%, calibration error more than doubled, and pituitary recall fell to 13.81% with 78.1% of pituitary predictions assigned to no-tumour — while no-tumour recall stayed at 100%. High internal accuracy on this dataset family therefore does not imply usable behaviour on data from elsewhere.

Useful next steps follow from the specific gaps above rather than from general ambition. Obtaining data with patient and study identifiers would permit a patient-level split and would test whether the duplicate-clean result is itself optimistic. Evaluating across several additional sources would show whether the pituitary failure is a property of this particular external collection. Because ranking quality (macro ROC-AUC 0.9849) degraded far less than decision quality under shift, source-specific recalibration is a concrete and testable avenue. Finally, resolving the effect of components such as the gate will require either a harder benchmark or many more seeds; it cannot be settled on this one.

This work is an image-level research demonstrator. It is not a diagnostic device, its splits are not patient-independent, and no claim of clinical readiness, fairness or deployment safety is made or supported.

---

## Declarations

**Ethics.** [AUTHOR INPUT REQUIRED — institutional ethics/IRB determination. The images used are from a publicly available, de-identified collection; the authors must confirm the determination applicable at their institution.]

**Data availability.** [AUTHOR INPUT REQUIRED — statement of dataset source and redistribution terms, repository URL and archival DOI.]

**Funding.** [AUTHOR INPUT REQUIRED]

**Competing interests.** [AUTHOR INPUT REQUIRED]

**CRediT author statement.** [AUTHOR INPUT REQUIRED]

**Declaration of generative AI.** [AUTHOR INPUT REQUIRED — the target venue's disclosure policy applies. The authors are responsible for reviewing and verifying all content prior to submission.]

**Acknowledgements.** [AUTHOR INPUT REQUIRED]

---

## Figures

| Figure | Content |
|---|---|
| 1 | AGMF-Net forward path with verified layer widths |
| 2 | Evaluation map: which images each reported number is computed on |
| 3 | Performance by evaluation regime, individual seeds shown |
| 4 | Confusion matrices, duplicate-clean and duplicate-disjoint cross-source |
| 5 | Ablation forest plot with paired image-level bootstrap intervals |
| 6 | Reliability curves with bin populations, both regimes |
| 7 | Gate-mass distribution overall and by true class |
| 8 | Class-specific cross-source failure and destination of pituitary predictions |

---

## References

1. Napa KK, Murugan S, Murugan JS, Jayanthi A. BRAIN-META: a reproducible CNN–vision transformer meta-ensemble pipeline for explainable brain tumor classification. *MethodsX*. 2026;16:103769. https://doi.org/10.1016/j.mex.2025.103769

2. Nickparvar M. Brain tumor MRI dataset. Kaggle; 2022. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

3. Tan M, Le QV. EfficientNet: rethinking model scaling for convolutional neural networks. *Proc. 36th Int. Conf. Machine Learning (ICML)*; 2019. [REFERENCE VERIFICATION REQUIRED — page range and PMLR volume to be confirmed against the published proceedings]

4. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16×16 words: transformers for image recognition at scale. *Int. Conf. Learning Representations (ICLR)*; 2021. [REFERENCE VERIFICATION REQUIRED — confirm final ICLR citation form]

5. Loshchilov I, Hutter F. Decoupled weight decay regularization. *Int. Conf. Learning Representations (ICLR)*; 2019. [REFERENCE VERIFICATION REQUIRED]

6. Chattopadhay A, Sarkar A, Howlader P, Balasubramanian VN. Grad-CAM++: generalized gradient-based visual explanations for deep convolutional networks. *IEEE Winter Conf. Applications of Computer Vision (WACV)*; 2018:839–847. https://doi.org/10.1109/WACV.2018.00097

7. Holm S. A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*. 1979;6(2):65–70. [REFERENCE VERIFICATION REQUIRED — confirm against publisher record]

8. McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*. 1947;12(2):153–157. https://doi.org/10.1007/BF02295996

9. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *Proc. 34th Int. Conf. Machine Learning (ICML)*; 2017. [REFERENCE VERIFICATION REQUIRED]

10. Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the inception architecture for computer vision. *IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*; 2016:2818–2826. [REFERENCE VERIFICATION REQUIRED — cited for label smoothing]

> **Reference status.** Entries 1, 2, 6 and 8 carry identifiers verified from the supplied article or the resource itself. Entries 3, 4, 5, 7, 9 and 10 are standard method citations whose bibliographic details were **not** verified against a publisher record in this environment, which had no network access. They are marked accordingly and must be checked before submission. No DOI, page range, volume or year has been invented; fields that could not be verified are absent rather than filled.
