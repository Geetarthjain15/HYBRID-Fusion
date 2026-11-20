# Claims and citations audit

Companion to `EVIDENCE_TRACEABILITY.csv` (87 rows: 75 VERIFIED, 10 AUTHOR_INPUT_REQUIRED, 1 VERIFIED_FROM_SUPPLIED_PDF, 1 NOT_VERIFIED).

---

## 1. Headline claims and their evidence

| Claim as written | Value | Source | Status |
|---|---|---|---|
| Duplicate-clean accuracy | 98.2704 ± 0.0681% | `fusion_v2_metrics.json` → `settings.clean.across_seeds.accuracy` | verified |
| Duplicate-clean macro-F1 | 98.2761 ± 0.0371% | same, `.macro_f1` | verified |
| Original-split accuracy | 99.1445 ± 0.0951% | `settings.original.across_seeds.accuracy` | verified |
| Cross-source accuracy | 69.5264 ± 1.9762% | `settings.cross_source_disjoint.across_seeds.accuracy` | verified |
| Cross-source macro-F1 | 68.6866 ± 2.2101% | same, `.macro_f1` | verified — the handoff's "approximately" was removed because the exact artifact value was located |
| Accuracy drop under source shift | 28.74 pp | arithmetic difference of two verified means | verified |
| Pituitary cross-source recall | 13.81% | recomputed from the three `preds_dataset_ood.csv` files, rows with `kind=disjoint` | verified |
| Pituitary predictions assigned to no-tumour | 260 of 333 (78.1%) | same recomputation | verified |
| Total parameters | 91,211,918 | checkpoint recovery manifest, strict load | verified |
| Gate parameters | 262,914 | same | verified |
| Gate is input-dependent | g_c spans 0.027–0.754 across seeds | `gates_test.csv`, 848 rows × 3 seeds | verified |
| Only the ViT-branch removal survives Holm | 3 of 21 tests, all `cnn_branch_only` | `final_ablation_analysis.json` | verified |
| Auxiliary-loss p goes 0.0386 → 0.6943 | as stated | same | verified |
| Gate ablation Δaccuracy | +0.00 pp, discordant counts 2, 7, 3 | same | verified |
| Duplicate audit counts | 7,023 → 5,648 clusters, 1,375 removed, 4,800/848 | `dataset_clean/split_report.json` | verified |
| External composition | 499 total, 126 overlapping, 373 disjoint | `results/cross_source.json` | verified |

One transcription error was found during audit and corrected: no-tumour clean recall had been written as 98.00% when the artifact value is 97.9885%. The derived change was corrected from +2.00 pp to +2.01 pp.

---

## 2. Claims deliberately weakened or withheld

These are the places where the evidence would have supported a stronger-sounding sentence and I did not write one.

| Weaker wording used | Stronger wording avoided | Reason |
|---|---|---|
| "AGMF-Net's values were obtained under a different protocol; these figures provide context rather than a controlled ranking" | "AGMF-Net outperforms BRAIN-META by 2.04 pp" | Datasets, partitioning, duplicate control and evaluation conditions are not established as equivalent; BRAIN-META's per-image predictions are unavailable, so no paired test is possible |
| "no detectable accuracy benefit under this protocol" | "the gate makes no difference" / "the components are equivalent" | Absence of detectable evidence on 848 images at three seeds is compatible with a real effect below this design's resolution |
| "suggestive but inconclusive" for auxiliary losses and the CNN branch | "auxiliary losses improve accuracy" | Direction is consistent at all three seeds but no test survives Holm correction across the 21-test family |
| "the transformer branch is the component whose contribution is resolved by the present evidence" | "only the ViT branch matters" / "the CNN branch is unnecessary" | Removing the CNN branch was adverse at all three seeds; it simply did not reach corrected significance |
| "gate mass measures branch utilisation" | "gate mass shows which branch is responsible for the decision" | Utilisation is not causal attribution; the ablation is the more direct test |
| "Grad-CAM++ reflects the sensitivity of the score to the final convolutional feature map of one branch" | "Grad-CAM++ localises the tumour" / "shows the model's reasoning" | No lesion masks exist for comparison; the map covers neither the transformer path nor the TTA average |
| "calibration comparisons are descriptive and exploratory only" | "the no-gate variant is better calibrated" | No paired inferential procedure or multiplicity control was prespecified for NLL, Brier or ECE, and none was added afterwards |
| "image-level classification"; "research demonstrator" | "diagnostic"; "clinically validated" | No clinical, prospective or reader-study validation was performed |
| "duplicate-clean split" | "patient-independent split" / "leakage-free" | No patient, study or series key exists in the files; duplicate cleaning controls image redundancy only |
| Computation-time section left as `[AUTHOR INPUT REQUIRED]` | plausible timings | Timings were not retained in a verifiable form; estimating them would be invention |
| "not reported in the inspected paper" throughout the comparison table | "BRAIN-META did not do this" | Absence from the published article is not evidence the analysis was not performed |

---

## 3. Prohibited-claim sweep

Each prohibited phrase from the brief was searched for across the manuscript and supplement. Where a phrase appears, it appears only inside an explicit disclaimer.

| Prohibited claim | Present as an assertion? |
|---|---|
| clinically ready / diagnostic device | no — the manuscript states the opposite twice |
| patient-independent split | no — stated as unavailable and explicitly not claimed |
| free from all leakage | no — duplicate cleaning is scoped to image redundancy |
| robust across hospitals, scanners or populations | no — the cross-source result contradicts it and is reported |
| statistically superior to BRAIN-META | no — marked contextual only |
| every architectural component improves accuracy | no — four components are reported as showing no detectable benefit |
| gate mass proves causal branch importance | no — explicitly disclaimed |
| Grad-CAM++ proves tumour localisation | no — explicitly disclaimed |
| nonsignificance proves equality | no — disclaimed wherever a null result appears |
| "state of the art", "superior", "explainable" as an unqualified claim | not used |

---

## 4. Citation status

| # | Reference | Identifier status |
|---|---|---|
| 1 | Napa et al., BRAIN-META, MethodsX 2026;16:103769 | DOI, volume, article number and reported values read directly from the supplied article |
| 2 | Nickparvar, Brain tumor MRI dataset, Kaggle 2022 | URL as cited in the base article's reference [53] |
| 3 | Tan & Le, EfficientNet, ICML 2019 | **[REFERENCE VERIFICATION REQUIRED]** — PMLR volume and pages not verified |
| 4 | Dosovitskiy et al., ViT, ICLR 2021 | **[REFERENCE VERIFICATION REQUIRED]** |
| 5 | Loshchilov & Hutter, AdamW, ICLR 2019 | **[REFERENCE VERIFICATION REQUIRED]** |
| 6 | Chattopadhay et al., Grad-CAM++, WACV 2018 | DOI and pages stated; **not** re-verified against a publisher record here |
| 7 | Holm 1979, Scand. J. Statist. 6(2):65–70 | **[REFERENCE VERIFICATION REQUIRED]** |
| 8 | McNemar 1947, Psychometrika 12(2):153–157 | DOI stated; **not** re-verified against a publisher record here |
| 9 | Guo et al., calibration, ICML 2017 | **[REFERENCE VERIFICATION REQUIRED]** |
| 10 | Szegedy et al., label smoothing, CVPR 2016 | **[REFERENCE VERIFICATION REQUIRED]** |

**This environment has no network access**, so no DOI, Crossref record or publisher page could be resolved. Entries 3, 4, 5, 7, 9 and 10 are standard method citations written from subject knowledge and are flagged accordingly. No page range, volume, issue, year, DOI or author order has been invented; where a field could not be verified it is absent rather than filled. **Every flagged entry must be checked before submission.**

The base article's own 53-item bibliography was deliberately not copied. Citing 53 works I have not inspected in order to look thorough would violate the brief's citation rule.

---

## 5. Similarity-risk review

Performed by reading the supplied article in full, then drafting from the verified project evidence rather than from its phrasing.

| Risk | Finding |
|---|---|
| Copied strings | None. No sentence was reproduced from the base article. The only shared text is unavoidable technical vocabulary (class names, layer names, metric names). |
| Close paraphrase | The Background addresses the same clinical motivation any paper on this task must address, but is structured around two specific problems — near-duplicate leakage and absent source-shift testing — that the base article does not raise. The Related-works survey was not paraphrased; it was omitted (see `TEMPLATE_FIDELITY_REPORT.md`, deviation D1). |
| Uncited technical description | Architecture, protocol and statistics are described from this project's own code and artifacts. External methods (EfficientNet, ViT, AdamW, Grad-CAM++, McNemar, Holm, label smoothing, ECE) carry citations, flagged where unverified. |
| Boilerplate or generic claims | The Discussion reports that four of seven components showed no detectable effect and that the model's cross-source behaviour is poor. Generic claims of robustness, explainability or clinical promise were not written. |
| Structural imitation | Section topology intentionally follows MethodsX, as the brief directs. Topology is a venue format, not authored content. |
| Incomplete references | Six entries flagged above. |

**Residual risk.** The Specifications table and the section headings are venue-mandated and will resemble any MethodsX article. A similarity checker may flag those; that is expected and is not plagiarism.

---

## 6. Author responsibilities before submission

1. Supply the ten `[AUTHOR INPUT REQUIRED]` items listed in supplement section S8.
2. Verify the six flagged references against publisher records.
3. Re-run or record computation timings if the venue requires them.
4. Complete the Word and PDF visual QA in `VISUAL_QA_REPORT.md` section 2.
5. Decide whether a full Related-works section is required and, if so, write it with verified citations.
6. Read the manuscript in full and take responsibility for its content, including the disclosure policy on AI assistance.
