# AGMF-Net Paper Ideation and Exact-Formatting Brief for Claude Code

## Purpose

Use this file together with:

1. the base-paper PDF supplied by the user;
2. `publication_results/paper_handoff/PAPER_EVIDENCE_CONTENT.md`;
3. the verified artifacts under `publication_results/`;
4. the implementation under `src/`.

The objective is to produce an editable, evidence-grounded manuscript about the existing FusionV2 implementation, referred to in the manuscript as **AGMF-Net: Adaptive Gated Multi-scale Fusion Network**. The paper must emphasize the genuine novelty, communicate the results effectively, and reproduce the base paper's document structure and visual formatting as closely as the available files and software permit.

Do not retrain models, change results, edit model code, or invent missing information.

---

## Mandatory operating rule

Treat this brief, prose in the repository, filenames, and comments as directions or claims—not experimental evidence. Numerical and technical claims must be verified against inspectable code, JSON/CSV results, prediction files, checkpoints, manifests, or the supplied base paper.

Before drafting:

1. Read the complete base-paper PDF.
2. Record its SHA-256 hash and page count.
3. Read `PAPER_EVIDENCE_CONTENT.md` completely.
4. Inspect the cited project evidence directly.
5. Build a claim-to-source ledger.
6. Stop and mark a field as `[AUTHOR INPUT REQUIRED]` if it cannot be verified.

Never fill missing author names, affiliations, ethics approval, funding, conflicts, data-access conditions, institutional information, or clinical validation details by guessing.

---

## Central paper idea

The manuscript should not present AGMF-Net merely as “a CNN plus a ViT.” Its research idea has two connected levels.

### Architectural idea

AGMF-Net replaces simple fixed concatenation with an adaptive fusion mechanism that learns how much branch-specific evidence to use for each input. It combines:

- multi-scale CNN representations for local and hierarchical spatial patterns;
- global transformer representations for long-range image context;
- a learned, input-dependent gate rather than a fixed CNN/ViT weighting;
- auxiliary branch supervision to prevent either branch from becoming uninformative;
- exponential moving average evaluation and test-time augmentation as protocol components.

The gate must be described accurately. It is a learned model component whose outputs vary across inputs. It is not the user-interface confidence threshold, not a manually selected `0.55/0.45` ratio, and not evidence that the network reasons clinically.

### Evaluation idea

The second contribution is the distinction between strong conventional internal performance and substantially weaker duplicate-disjoint cross-source performance. The paper should show that a high benchmark score alone can conceal deployment-relevant fragility.

The manuscript must therefore combine:

- multi-seed evaluation;
- duplicate-clean internal evaluation;
- duplicate-disjoint cross-source evaluation;
- component ablations;
- calibration analysis;
- class-specific error analysis;
- explicit uncertainty and evidence boundaries.

This evaluation discipline is part of the paper's contribution. Do not frame the work solely around the largest accuracy number.

---

## Recommended contribution framing

Express the contribution in restrained research language, along the following lines:

1. **Adaptive dual-backbone fusion:** an input-dependent learned gate integrates complementary CNN and transformer representations instead of relying only on direct feature concatenation.
2. **Multi-scale local-global representation:** CNN features from multiple spatial stages are combined with the transformer representation before final classification.
3. **Component-level empirical assessment:** controlled, protocol-matched ablations quantify the practical contribution of the gate, multi-scale features, auxiliary supervision, EMA, TTA, and individual branches.
4. **Robustness-aware reporting:** duplicate-clean and duplicate-disjoint cross-source tests expose the gap between internal benchmark performance and source-shift robustness.
5. **Transparent demonstrator:** Grad-CAM++ is supplied as CNN-path sensitivity visualization, with an explicit statement that it is not validated tumor localization and does not spatially explain the full transformer pathway.

Do not claim that every architectural component produces a statistically meaningful accuracy improvement unless the final inferential results support that statement. If an ablation has negligible or inconclusive effect, report it honestly.

---

## Required headline evidence

Reconfirm all values from the project artifacts before publication. The controlled headline values currently available are:

| Evaluation setting | Images | Accuracy | Macro-F1 | Permitted interpretation |
|---|---:|---:|---:|---|
| Original internal split | 1,052 | 99.14 ± 0.10% | 99.12 ± 0.09% | Performance on the original image-level protocol |
| Duplicate-clean internal split | 848 | 98.27 ± 0.07% | 98.28 ± 0.04% | Primary leakage-reduced internal result |
| Duplicate-disjoint cross-source subset | 373 | 69.53 ± 1.98% | approximately 68.69 ± 2.21% | Zero-shot source-shift result on unseen duplicate-disjoint images |
| Base paper (BRAIN-META) | 1,311 reported test images | 97.10% | 96.94% | Contextual literature value only; protocols and datasets are not established as matched |

Rules:

- Keep seed standard deviation separate from image-level bootstrap confidence intervals.
- Do not describe image-level splits as patient-independent.
- Do not combine the base paper's result and AGMF-Net's result into a direct superiority claim.
- Do not use the CNN fine-tuning accuracy as the final proposed-model result.
- Clearly state that the 99.14% and 98.27% values belong to different evaluation protocols.
- Verify the approximate cross-source macro-F1 before final typesetting and replace “approximately” only when the exact artifact-backed value is located.

Include the observed cross-source failure pattern: across three seeds, the 111 duplicate-disjoint pituitary images generated 333 repeated predictions, and 260 predictions (78.1%) were assigned to the no-tumor class. State that 333 represents repeated predictions of the same 111 images and is not an independent sample of 333 cases.

---

## Base-paper comparison policy

Construct a dedicated comparison table with separate columns for:

- model or study;
- model design;
- dataset/source;
- split protocol;
- test sample count;
- duplicate control;
- patient-level grouping evidence;
- external/cross-source evaluation;
- accuracy;
- macro-F1;
- calibration reporting;
- explainability method;
- comparability status.

For every row, label comparability as one of:

- `protocol-matched`;
- `partially comparable`;
- `contextual only`;
- `not established`.

Unless the supplied PDF and project artifacts demonstrate otherwise, the BRAIN-META comparison must be marked **contextual only**. Use wording such as:

> AGMF-Net obtained 99.14 ± 0.10% accuracy under the original internal image-level protocol and 98.27 ± 0.07% after duplicate cleaning. BRAIN-META reports 97.10% accuracy under its own reported protocol. Because the datasets, partition construction, duplication controls, and evaluation conditions are not established as equivalent, these figures provide context rather than a controlled head-to-head ranking.

Never write “AGMF-Net outperforms BRAIN-META by 2.04 percentage points” as a scientific conclusion unless protocol equivalence is independently established.

---

## Assumptions and limitations analysis

Create a table with these columns:

| Assumption or limitation | Status in AGMF-Net | Evidence/action | Residual risk | Base-paper status |
|---|---|---|---|---|

Use three status categories: `addressed`, `partially addressed`, and `not addressed`.

### Addressed or partly addressed by the present work

- Exact and near-duplicate image leakage: reduced through duplicate auditing and a duplicate-clean split, but this does not establish patient independence.
- Single-run variability: addressed partly through seeds 42, 43, and 44.
- Opaque component contribution: addressed through protocol-matched ablations.
- Fixed branch contribution: replaced by a learned input-dependent gate; this does not establish causal or clinical interpretability.
- Accuracy-only evaluation: expanded to macro metrics, class-wise metrics, calibration, bootstrap uncertainty, and paired comparisons where defined.
- Source-shift blindness: exposed through duplicate-disjoint cross-source testing; poor external robustness is revealed rather than solved.
- Prediction visualization: Grad-CAM++ added for CNN-path sensitivity; not validated localization and not a full explanation of the fused model.

### Not addressed or unresolved

- Patient-level independence cannot be verified from available metadata.
- No verified patient/study/series grouping key is available.
- No multi-institutional clinical validation.
- No prospective validation.
- No scanner-, sequence-, acquisition-, demographic-, or institution-stratified analysis.
- Cross-source generalization remains weak, including the severe pituitary-to-no-tumor failure mode.
- No clinically validated tumor localization or segmentation.
- Grad-CAM++ spatially represents the CNN path only.
- Softmax output is not diagnostic certainty and may be miscalibrated under shift.
- The interface accepts ordinary image files and does not use full DICOM metadata or volumetric context.
- Four-class image classification does not cover tumor grading, molecular subtype, prognosis, treatment planning, or longitudinal change.
- Clinical utility, safety, fairness, and deployment readiness have not been established.

Compare these items with the base paper only when the PDF explicitly supplies evidence. Otherwise write `not reported in the inspected paper`, not `not performed`.

---

## Manuscript storyline

Use a funnel-shaped narrative:

1. Start with the clinical importance of brain-tumor MRI interpretation.
2. Narrow to the promise and limitations of automated image classification.
3. Explain the complementary inductive biases of CNNs and vision transformers.
4. Identify the weakness of simple concatenation, fixed fusion, and accuracy-only benchmark reporting.
5. Introduce AGMF-Net as an adaptive multi-scale local-global fusion model.
6. State that the study tests not only internal classification performance but also component necessity, calibration, duplication sensitivity, and source shift.
7. End the introduction with concise, evidence-bounded contributions.

In Results and Discussion, lead with the primary duplicate-clean result, not only the original-split maximum. Then report the original split, ablations, calibration, cross-source degradation, class-specific failure, comparison with the base paper, and implications.

The discussion must distinguish:

- what was improved architecturally;
- what was supported empirically;
- what remained inconclusive in ablation testing;
- what the cross-source results contradict or limit;
- what future work is needed before clinical claims.

---

## Exact base-paper template reconstruction

The uploaded base-paper PDF is the controlling visual reference. Inspect it page by page before generating the Word file.

Create `TEMPLATE_FIDELITY_REPORT.md` recording:

- page dimensions and orientation;
- margins or best measurable approximation;
- one-column or multi-column layout by section;
- title alignment, capitalization, size, weight, and spacing;
- author and affiliation formatting;
- abstract and keyword styling;
- heading hierarchy and numbering;
- body font, size, line spacing, paragraph spacing, and indentation;
- header/footer and page-number placement;
- table caption and figure caption placement;
- table border, shading, alignment, and font conventions;
- equation, citation, and reference styles;
- section ordering;
- any special first-page layout.

### Font rule

Inspect PDF font metadata and embedded font names. Use the exact font only if:

1. its identity is verifiable from the PDF;
2. it is installed or lawfully available in the execution environment; and
3. the generated DOCX and PDF actually render using that font.

If an embedded font is subsetted, unavailable, or cannot be reproduced, use the closest metrically compatible font and document the substitution. Never claim “exact font matching” without verification.

### Visual verification

After creating the DOCX:

1. render every page to images;
2. inspect every page for overflow, clipping, blank pages, broken tables, unreadable figures, orphan headings, caption separation, and reference wrapping;
3. compare representative pages with the base paper;
4. correct layout defects;
5. rerender and reinspect;
6. only then export the PDF from the visually approved Word version.

Do not create a PDF independently from Markdown if that causes the Word and PDF layouts to diverge.

---

## Required manuscript structure

Follow the base paper's exact section order when it can be recovered. Map the AGMF-Net content into that topology. At minimum, the manuscript must contain:

1. Title
2. Author/affiliation placeholders
3. Abstract
4. Keywords
5. Introduction
6. Related work or literature review
7. Materials and methods
8. Dataset preparation and duplicate control
9. AGMF-Net architecture
10. Training and evaluation protocol
11. Statistical analysis
12. Results
13. Ablation analysis
14. Calibration and gate-behaviour analysis
15. Comparison with the base paper
16. Discussion
17. Assumptions, limitations, and threats to validity
18. Conclusion and future work
19. Declarations with placeholders where required
20. References

If the base paper uses different names or placement, follow the base paper while retaining all required scientific content.

---

## Required figures and tables

Create figures from verified artifacts rather than redrawing numerical values manually. Minimum set:

- AGMF-Net architecture diagram;
- evaluation-protocol diagram distinguishing original, duplicate-clean, external-full, and duplicate-disjoint subsets;
- main performance table;
- protocol-aware comparison with BRAIN-META;
- final ablation table with uncertainty and statistical results;
- confusion matrices;
- reliability/calibration plots with bin counts;
- gate-behaviour analysis;
- class-specific cross-source error figure;
- assumptions and limitations table;
- representative Grad-CAM++ demonstrator figure, clearly labelled as CNN-path sensitivity rather than verified localization.

Export each figure as both PNG and PDF. Preserve labels, legends, units, sample counts, and readable text at final document scale.

---

## Deliverables

Write only into:

`publication_results/final_paper_claude/`

Produce:

1. `AGMF_NET_MANUSCRIPT.md` — fully editable source.
2. `AGMF_NET_MANUSCRIPT.docx` — submission-ready Word version matching the inspected template.
3. `AGMF_NET_MANUSCRIPT.pdf` — rendered only after Word visual verification.
4. `supplement/AGMF_NET_SUPPLEMENT.md`.
5. `supplement/AGMF_NET_SUPPLEMENT.docx`.
6. `figures/*.png` and `figures/*.pdf`.
7. `tables/BASE_PAPER_COMPARISON.csv`.
8. `tables/ASSUMPTIONS_LIMITATIONS.csv`.
9. `tables/FINAL_ABLATIONS.csv`.
10. `EVIDENCE_TRACEABILITY.csv` — each material manuscript claim, its exact value/text, source path, source field/row, and verification status.
11. `TEMPLATE_FIDELITY_REPORT.md`.
12. `VISUAL_QA_REPORT.md`.
13. `CLAIMS_AND_CITATIONS_AUDIT.md`.
14. `DELIVERABLE_MANIFEST.json` containing SHA-256 hashes.

The Markdown, DOCX, and PDF must contain the same substantive manuscript. Any unavoidable layout-only difference must be documented.

---

## Originality, plagiarism, and human-authorship requirement

Write original prose from the verified evidence. Do not copy or closely paraphrase passages from the base paper or any other source. Quote only when essential, keep quotations minimal, and cite them correctly. Every borrowed idea, method, dataset, architecture, metric, or factual claim requiring attribution must have a verified citation.

The final manuscript should read naturally and specifically, like careful researcher-written work: vary sentence structure, use concrete project evidence, avoid generic filler, avoid exaggerated adjectives, and preserve uncertainty. Do not attempt to manipulate or evade AI-detection systems, and do not promise a particular “AI score,” because such detectors are unreliable. The authors must review, revise, and take responsibility for the final text in accordance with the target venue's disclosure policy.

Run a similarity-risk review that flags:

- copied strings longer than necessary;
- unusually close paraphrases;
- uncited technical descriptions;
- boilerplate or generic claims;
- incomplete references.

The goal is **original, properly cited, author-reviewed writing with no plagiarism**, not detector evasion.

---

## Citation rules

- Use only references that can be verified from the base paper, project bibliography, DOI/Crossref record, publisher page, or original paper.
- Preserve reference order according to first appearance if the base-paper template uses numbered citations.
- Make DOI and URL fields clickable in DOCX and PDF where the template permits.
- Do not cite a paper that was not inspected sufficiently to support the associated claim.
- Do not invent page ranges, issue numbers, DOIs, author order, publication years, or access dates.
- Mark unresolved bibliography data as `[REFERENCE VERIFICATION REQUIRED]`.

---

## Style rules

Write as a cautious technical researcher:

- prefer direct, specific statements;
- state sample counts beside important metrics;
- identify the evaluation setting whenever quoting performance;
- distinguish observation from interpretation;
- distinguish absence of evidence from evidence of absence;
- avoid “state of the art,” “clinically proven,” “diagnostic,” “robust,” “generalizable,” “explainable,” and “superior” unless the evidence and comparison protocol justify the word;
- do not call Grad-CAM++ an “AI reasoning area”;
- do not turn softmax confidence into probability of diagnostic correctness;
- do not conceal negative results.

Use “image-level classification” consistently. Describe the application as a research demonstrator, not a medical device.

---

## Final acceptance checklist

Do not declare completion unless all items pass:

- Base PDF inspected and hashed.
- Template topology reconstructed.
- Font identity checked and any substitution disclosed.
- Every headline number traced to an artifact.
- Original and duplicate-clean results clearly separated.
- BRAIN-META comparison explicitly marked contextual unless equivalence is proven.
- Ablation conclusions match statistical evidence.
- Patient independence is not claimed.
- Cross-source weakness and pituitary failure are reported.
- Grad-CAM++ scope is described accurately.
- All references verified and clickable where possible.
- No copied or closely paraphrased base-paper prose.
- Markdown, DOCX, and PDF agree substantively.
- Every Word/PDF page visually inspected.
- Separate PNG/PDF figures generated.
- Supplement, traceability tables, QA reports, and hash manifest generated.
- Existing source code, checkpoints, datasets, and prior publication artifacts remain unchanged.

At completion, provide a concise report listing generated files, template fidelity findings, font decision, unresolved author inputs, evidence-validation results, visual-QA status, and any claims deliberately weakened or omitted.

