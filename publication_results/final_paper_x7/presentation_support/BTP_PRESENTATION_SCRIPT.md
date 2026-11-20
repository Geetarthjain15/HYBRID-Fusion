# Complete Presentation Script

## Multi-Scale EfficientNet–ViT Feature Fusion for Duplicate-Aware and Explainable Brain-Tumour MRI Classification

**Designed for:** the supplied 14-slide End-Term Review presentation  
**Suggested speaking time:** approximately 18–20 minutes for the complete script, excluding questions  
**Scientific position:** emphasize the integrated fusion system and the rigor of its evaluation; do not claim clinical readiness, verified tumour localisation, universal generalisation, or statistical superiority over BRAIN-META.

## Critical correction before presenting

Slide 10 currently contains altered values for six ablations. The verified prediction-derived values are:

| Configuration | Verified accuracy (%) | Verified change from full model (pp) | Evidence-based interpretation |
|---|---:|---:|---|
| Full FusionV2 | 98.27 ± 0.07 | Reference | Complete model |
| ViT removed (CNN only) | 95.68 ± 0.34 | −2.59 | Holm-significant loss at all three seeds |
| Auxiliary losses removed | 97.72 ± 0.48 | −0.55 | Consistent adverse direction; inconclusive after Holm correction |
| CNN removed (ViT only) | 98.03 ± 0.14 | −0.24 | Higher full-model mean; inconclusive after Holm correction |
| TTA removed | 98.19 ± 0.25 | −0.08 | No detectable accuracy benefit under this protocol |
| Multi-scale taps removed | 98.23 ± 0.20 | −0.04 | No detectable accuracy benefit under this protocol |
| Adaptive gate removed | 98.27 ± 0.18 | 0.00 | No detectable accuracy benefit under this protocol |
| EMA removed | 98.31 ± 0.30 | +0.04 | No detectable accuracy benefit under this protocol |

Do not say that six of seven ablations prove a contribution. Only removal of the ViT branch produced a Holm-significant accuracy loss at every seed. This does not weaken the project: it makes the main contribution more precise and defensible.

---

## Slide 1 — Title

**Suggested time: 35 seconds**

Good morning/afternoon. We are presenting our project, titled **“Multi-Scale EfficientNet–ViT Feature Fusion for Duplicate-Aware and Explainable Brain-Tumour MRI Classification.”**

The work addresses four-class classification of brain MRI images into glioma, meningioma, pituitary tumour and no tumour. The central idea is not simply to combine two popular backbones. We developed a single feature-level network that brings together the local spatial representation of EfficientNet-B0 and the global representation of ViT-B/16. We then evaluated this network using duplicate-aware data preparation, three independent training seeds, controlled paired ablations, cross-source testing and a quantitatively audited explainability pipeline.

The main message of the project is that a strong architecture should be supported not only by high internal accuracy, but also by evidence showing which components matter, how reproducible the result is, and how behaviour changes when the source distribution changes.

**Transition:** I will first explain the research gap, then the proposed fusion method, and finally the evidence produced by our evaluation.

---

## Slide 2 — Contents

**Suggested time: 20 seconds**

The presentation begins with the motivation and literature survey. We then identify the gaps that shaped our objectives and problem formulation. After that, we describe the duplicate-aware evaluation pipeline, the proposed architecture and the training protocol. The final part presents the internal, ablation, cross-source and explainability results, followed by future work and expected outcomes.

---

## Slide 3 — Introduction

**Suggested time: 60 seconds**

Magnetic resonance imaging is routinely used to examine intracranial abnormalities, but interpreting a large number of scans is demanding. Tumour categories may also share visual characteristics, making consistent image-level classification difficult. A computer-aided model can therefore serve as a research tool for studying how local and global image evidence may be combined. It is not presented here as a replacement for clinical judgement.

Convolutional networks and vision transformers offer complementary representations. EfficientNet is effective at extracting local texture, edge and boundary information through spatial feature maps. A ViT divides the image into patches and models broader relationships through transformer layers. Our research question was whether these representations could be integrated inside one trainable feature-fusion network rather than combined only after separate models had made their predictions.

The base paper, BRAIN-META, reported 97.10 percent accuracy and 96.94 percent F1 through a ten-learner CNN–ViT meta-ensemble. That study provides an important reference point, but its result is contextual because its split and pipeline differ from ours. Our contribution is directed toward a single integrated fusion network, together with duplicate control, repeated-seed evidence, controlled ablation, source-shift evaluation and explanation auditing.

---

## Slide 4 — Literature Survey

**Suggested time: 65 seconds**

The surveyed work shows a progression from prediction-level ensembles to increasingly integrated CNN–transformer systems. BRAIN-META stacks the outputs of multiple CNN and ViT learners using higher-level meta-learners. Other recent approaches introduce attention gates, multi-backbone ensembles, local cross-attention, domain-aware training, uncertainty estimates or quantitative explanation assessment.

These studies establish that hybrid representations are promising, but they also reveal that a high headline accuracy is not sufficient by itself. Evaluation protocols differ substantially: some use one image-level split, some use cross-validation, and only a smaller group combines duplicate screening or external testing. This makes numerical rankings across papers unsafe unless the models are reproduced on the same data and split.

Our work is positioned at the intersection of these ideas. Architecturally, it uses multi-scale EfficientNet features and a ViT class-token representation within one gated feature-fusion model. Experimentally, it combines duplicate-aware splitting, three-seed training, paired statistical ablation and duplicate-disjoint external evaluation. Explainability is treated as a testable model property rather than as a visually persuasive heatmap alone.

**Presenter note:** describe the literature as motivation and context. Do not claim that the survey proves worldwide uniqueness or statistical superiority.

---

## Slide 5 — Gaps Identified

**Suggested time: 55 seconds**

Four gaps directly motivated the project.

First, prediction-level stacking combines final decisions, but it does not directly learn how local CNN features and global transformer features should interact inside a single representation.

Second, random image-level splitting can give an overly optimistic impression when visually similar or near-duplicate images occur across partitions. A duplicate-aware analysis is therefore required alongside the conventional split.

Third, complex hybrid models are often presented without testing whether their individual components produce reproducible gains. We address this through seven protocol-matched ablations, filename-aligned predictions, exact paired McNemar testing and Holm correction across seed-specific comparisons.

Fourth, attractive heatmaps do not establish faithful or anatomically correct reasoning. Our explainability work therefore includes branch-specific extraction, numerical-neutrality checks, sanity checks, deletion and insertion tests, transformation stability and a locked cross-source analysis.

These gaps lead to the novelty of the work: **an integrated fusion architecture evaluated through a linked chain of architectural, statistical, robustness and explanation evidence.**

---

## Slide 6 — Aims and Objectives

**Suggested time: 45 seconds**

The first objective was to develop a single EfficientNet-B0 and ViT-B/16 feature-level fusion model for four-class MRI classification.

The second was to establish whether its strong performance remained reproducible after duplicate cleaning and across seeds 42, 43 and 44.

The third was to isolate the contribution of the principal components using controlled ablations and paired statistical testing rather than relying only on the full-model accuracy.

The fourth was to evaluate the model on duplicate-disjoint images from a separate source, without using those final test labels for training or model selection.

The fifth was to build a dual-path explainability pipeline in which CNN and ViT sensitivity maps correspond to the same image, selected class and forward pass, and where extraction is verified not to change the prediction.

Together, these objectives move the work from model construction toward a reproducible evaluation framework.

---

## Slide 7 — Problem Statement and Formulation

**Suggested time: 75 seconds**

The problem is to classify an MRI image into one of four classes while testing whether a single CNN–ViT network can retain strong duplicate-clean performance, demonstrate reproducible component contributions and reveal its limitations under source shift.

Let the CNN branch produce a projected feature vector, **c**, and the ViT branch produce a projected vector, **v**, each with 512 dimensions. The gate receives the concatenated representation and produces two non-negative weights, one for the CNN branch and one for the ViT branch, which sum to one. Each projected feature is scaled by twice its gate weight, and the two weighted vectors are concatenated into a 1,024-dimensional fused representation. This scaling preserves the approximate feature magnitude when both weights are near one half.

The training objective combines the fused cross-entropy loss with auxiliary CNN-branch and ViT-branch losses, weighted by 0.3, with label smoothing of 0.1. The auxiliary heads are used during training to encourage each branch to remain individually discriminative.

For explanation analysis, the CNN Grad-CAM++ map and the ViT Gradient-times-Input patch map can be combined using the image-specific gate weights. However, the gate describes branch utilisation; it must not be interpreted as causal or anatomical proof. The composite map is therefore evaluated alongside each branch map and simple baselines.

---

## Slide 8 — Duplicate-Aware Evaluation Pipeline

**Suggested time: 70 seconds**

The starting collection contained 7,023 JPEG MRI images from the four target classes. Images were resized to 224 by 224 pixels and normalized using a mean and standard deviation of 0.5, matching the established inference pipeline.

To reduce the influence of highly similar images, we applied a 256-bit perceptual difference hash. Images within a Hamming-distance threshold of eight were grouped into clusters. This produced 5,648 representative images after 1,375 redundant images were removed. Using the fixed clean split, 4,080 images were assigned to training, 720 to validation and 848 to the test set. We retained the conventional 1,052-image test split as a secondary reference so that the effect of duplicate cleaning could be measured.

For source-shift evaluation, a separate 499-image collection was compared against the training pool using the same duplicate-screening logic. The 373 images that were duplicate-disjoint from the training data formed the external evaluation subset. These labels were used only for final scoring, never for model selection or tuning.

This pipeline is central to the project because it separates strong internal performance from robustness to an altered source distribution.

---

## Slide 9 — Architecture and Experimental Protocol

**Suggested time: 75 seconds**

The CNN pathway uses ImageNet-pretrained EfficientNet-B0. Features are collected from stage three, stage five and the final stage. After spatial pooling, these features form a 1,432-dimensional multi-scale vector, which is projected to 512 dimensions.

In parallel, the ViT-B/16 pathway provides a 768-dimensional class-token representation, which is also projected to 512 dimensions. The learned two-way gate weights these projected features before concatenation. The fused representation is passed to the final four-class classifier. Two auxiliary heads are active only during training. The complete network contains approximately 91.2 million parameters; the gate itself contains approximately 262,914 parameters.

Training used AdamW with a learning rate of 1 times 10 to the minus 4, weight decay 0.05, batch size 16, a maximum of 15 epochs, early stopping, exponential moving average and flip-based test-time augmentation. The entire primary experiment was repeated for seeds 42, 43 and 44.

Seven ablations were trained or evaluated under the same protocol. Explainability used CNN Grad-CAM++ at a native 7-by-7 resolution and ViT Gradient-times-Input over a 14-by-14 patch grid. The attribution layer was implemented as a read-only publication wrapper, leaving the verified classifier unchanged.

---

## Slide 10 — Performance and Controlled Ablation

**Suggested time: 105 seconds**

The model achieved **99.14 plus or minus 0.10 percent accuracy** and **99.12 plus or minus 0.09 percent macro-F1** on the conventional test split. On the primary duplicate-clean test set of 848 images, it retained **98.27 plus or minus 0.07 percent accuracy** and **98.28 plus or minus 0.04 percent macro-F1**. The reduction in mean accuracy after duplicate cleaning was 0.87 percentage points, while variation across the three seeds remained small.

The controlled ablation provides the most important evidence about the architecture. Removing the ViT branch reduced accuracy from 98.27 to **95.68 percent**, a mean loss of **2.59 percentage points**. This was the only tested deletion whose adverse effect remained significant after Holm correction at every seed. Therefore, the transformer pathway makes a reproducible contribution and the model is not operating as a CNN-only system.

Removing auxiliary supervision reduced the mean to **97.72 percent**, a 0.55-point loss with the same adverse direction at all seeds. This supports auxiliary supervision as a promising training mechanism, although the comparison was inconclusive after multiplicity correction. Removing the CNN branch produced **98.03 percent**, 0.24 points below the full model; the higher full-model mean is compatible with complementary local information, but the present sample does not statistically resolve that difference.

The effects of removing TTA, multi-scale taps or the adaptive gate were small under this protocol, and removing EMA slightly increased the mean by 0.04 points. We therefore avoid claiming that every mechanism independently improves accuracy. The defensible conclusion is narrower and stronger: **the integrated system achieves high duplicate-clean performance, and the ViT pathway is the component with clear reproducible accuracy evidence, while auxiliary supervision shows a consistent but not multiplicity-corrected trend.**

On the full external set, accuracy was **76.49 plus or minus 1.52 percent**. On the stricter 373-image duplicate-disjoint subset, it was **69.53 plus or minus 1.98 percent**, with macro-F1 of **68.69 plus or minus 2.21 percent**. This external result is not a failure to hide; it is evidence that the internal score alone cannot establish cross-source robustness.

---

## Slide 11 — Cross-Source and Explainability Results

**Suggested time: 110 seconds**

The class-wise external analysis shows where the source shift is concentrated. Mean recall remained **98.08 percent for glioma**, **86.07 percent for meningioma** and **100 percent for no tumour**, but fell to **13.81 percent for pituitary images**. This indicates a class-specific vulnerability rather than a uniform degradation, and it provides a concrete direction for future source-diverse training.

For explainability, we produced **5,088 branch-specific maps**: CNN Grad-CAM++ and ViT Gradient-times-Input maps for all 848 clean test images across three seeds. Both maps were generated for the same fused target class and synchronized forward pass. Instrumentation caused exactly zero probability difference and zero label changes relative to ordinary inference. All nine prespecified sanity gates passed.

Under the locked faithfulness protocol, lower deletion AUC and higher insertion AUC were preferred. CNN Grad-CAM++ achieved a deletion AUC of **0.3649** and insertion AUC of **0.7186**. ViT Gradient-times-Input achieved **0.5182** and **0.5814**, respectively. The equal composite achieved **0.3846** and **0.7115**, while the gate-weighted composite achieved **0.4858** and **0.6203**.

The most important scientific safeguard is the centre-prior control, which scored **0.3525** for deletion and **0.7148** for insertion. Its similarity to the strongest learned maps shows that central image structure can strongly influence these perturbation metrics. Consequently, we present the maps as branch-specific sensitivity evidence—not as tumour segmentation or clinically verified localisation.

The same locked protocol was applied without retuning to the 373 duplicate-disjoint external images. This extends source-shift analysis beyond accuracy to confidence, branch behaviour and explanation measurements, but it does not establish a single causal mechanism for the performance drop.

Compared with BRAIN-META's reported 97.10 percent accuracy, our internal values are numerically higher and our evaluation adds duplicate control, repeated seeds, paired ablation, external testing and quantitative explanation auditing. This remains a **contextual comparison**, not a controlled claim of statistical superiority, because the studies use different splits and pipelines.

---

## Slide 12 — Future Work and Expected Outcomes

**Suggested time: 65 seconds**

The external pituitary result gives the first priority for future work: training with more source-diverse data, particularly pituitary examples collected under varied acquisition conditions. Patient-level evaluation should also be performed when reliable patient or study identifiers become available; it cannot be inferred from the current metadata.

Calibration or prior-correction methods should be fitted only with a separate validation source, never on final external-test labels. Architectural extensions such as spatial CNN–ViT cross-attention may allow feature interaction before global pooling, but they are future hypotheses and must be compared with the current model under the same controlled protocol. Lighter transformer backbones or knowledge distillation may improve efficiency.

For explainability, expert tumour masks would enable localisation-specific measures such as Dice, overlap or pointing-game accuracy. Until those annotations exist, our current evidence is limited to model faithfulness and stability.

The expected outcome is therefore not simply a more complex model. It is a source-diverse, patient-aware and directly comparable validation framework in which any new architectural change must preserve the same duplicate-aware, three-seed and paired statistical standards established here.

---

## Slide 13 — References

**Suggested time: 20 seconds**

These are the principal studies used to position the project. BRAIN-META is the base paper and motivates comparison with prediction-level stacking. The remaining studies represent related directions in gated fusion, attention-based CNN–transformer integration, source-shift evaluation, uncertainty and quantitative explanation analysis. The references support the research gap and methodology, while all numerical claims about our model come from our stored prediction and analysis artifacts.

---

## Slide 14 — Conclusion and Thank You

**Suggested time: 45 seconds**

To conclude, this project contributes a single multi-scale EfficientNet-B0 and ViT-B/16 feature-fusion system and, equally importantly, a rigorous way of evaluating it.

The model retained **98.27 percent duplicate-clean accuracy**, and controlled ablation established a reproducible contribution from the ViT pathway. Duplicate-disjoint testing then revealed a substantial cross-source gap, concentrated particularly in the pituitary class. Finally, the dual-path explanation pipeline was shown to be numerically neutral, sanity-tested and quantitatively evaluated, while the centre-prior result prevented us from overstating heatmaps as localisation evidence.

Therefore, the novelty is not based on a single accuracy value. It lies in connecting **integrated feature fusion, duplicate-aware reproducibility, statistically controlled component analysis, source-shift evaluation and audited dual-path explainability** within one research framework.

Thank you. We welcome your questions.

---

# Short Defence Answers for Likely Questions

## 1. What is the main novelty?

The main novelty is the combination of a single multi-scale EfficientNet-B0–ViT-B/16 feature-fusion model with a linked evidence framework: duplicate-aware splitting, three-seed evaluation, paired ablation with Holm correction, duplicate-disjoint external testing, and numerically neutral dual-path explanation auditing. No one element alone should be presented as the entire novelty.

## 2. Does the adaptive gate improve accuracy?

The gate is implemented and produces image-dependent branch weights, but its removal changed mean clean accuracy by 0.00 percentage points under the tested protocol. Therefore, it demonstrates dynamic branch utilisation, not a proven independent accuracy gain.

## 3. Which component has the clearest ablation evidence?

The ViT pathway. Removing it reduced mean clean accuracy by 2.59 percentage points, and the loss remained significant after Holm correction at all three seeds. Auxiliary supervision showed a consistent 0.55-point adverse change when removed, but this was not Holm-significant across seeds.

## 4. Is FusionV2 statistically better than BRAIN-META?

That claim cannot be made. FusionV2 has higher reported internal numerical values, but BRAIN-META used a different split and pipeline, and its per-image predictions are unavailable for a controlled paired comparison. The comparison is contextual only.

## 5. Why is cross-source accuracy lower?

The evidence establishes an association with source shift, not one proven cause. The error is strongly class-dependent: pituitary recall dropped to 13.81 percent, whereas no-tumour recall remained 100 percent. Acquisition differences, dataset composition and learned source-specific cues are plausible hypotheses requiring further controlled study.

## 6. Do the heatmaps identify the tumour?

No. The maps show model sensitivity through the CNN and ViT pathways. They have not been validated against expert tumour masks and must not be described as segmentation, localisation or clinical reasoning.

## 7. Why are three seeds used?

Three seeds measure whether the result and direction of an ablation persist across independent training initialisations. This reduces dependence on one favourable run. It strengthens reproducibility, although three seeds are not themselves an architectural novelty.

## 8. Why is the centre-prior result important?

The centre-prior control performed similarly to the strongest attribution methods under deletion and insertion. This warns that centrally concentrated anatomy or preprocessing can score well even without class-specific learned attribution. Including this control makes the explanation claim more honest and scientifically defensible.

## 9. Is the model clinically ready?

No. It is an image-level research classifier. Patient-level independence could not be verified, performance declined under source shift, and the explanations lack expert localisation validation.

## 10. What should be improved next?

The highest priority is source-diverse training and independent patient-level evaluation, especially for pituitary cases. The next architectural experiments—such as spatial cross-attention—should be tested against the current verified model using the same duplicate-aware and paired statistical protocol.
