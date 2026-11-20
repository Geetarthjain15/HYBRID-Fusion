# Claims and citations audit

## Status

The manuscript's quantitative claims are linked to machine-readable project artifacts in `EVIDENCE_TRACEABILITY.csv`. Bibliographic statements are linked to the supplied BRAIN-META PDF or stable proceedings/DOI records. No missing result was inferred.

## Allowed claims used

- The proposed model is a single multi-scale EfficientNet-B0–ViT-B/16 feature-fusion network with auxiliary branch supervision and an input-dependent two-way gate.
- Original-split performance is 99.14 ± 0.10% accuracy and 99.12 ± 0.09% macro-F1 across seeds 42, 43 and 44.
- Duplicate-clean performance is 98.27 ± 0.07% accuracy and 98.28 ± 0.04% macro-F1 on 848 images.
- Duplicate-disjoint cross-source performance is 69.53 ± 1.98% accuracy and 68.69 ± 2.21% macro-F1 on 373 images.
- Internal and cross-source paired ablations use exact two-sided McNemar tests with Holm correction over 21 seed-specific comparisons.
- Removing the ViT branch is the only tested deletion with a Holm-significant accuracy loss at every seed in both evaluation regimes.
- The gate is learned and input-dependent; its mass measures branch utilisation, not causal or anatomical explanation.
- Grad-CAM++ visualises sensitivity through the CNN path only and has not been validated as lesion localisation.
- BRAIN-META's 97.10% accuracy and 96.94% F1 are contextual values because its split and pipeline differ.

## Prohibited or unsupported claims excluded

- Clinical readiness, diagnostic safety or medical-device status.
- Elimination of all leakage or proof of patient-independent splitting.
- Universal hospital, scanner or acquisition generalisation.
- Statistical superiority over BRAIN-META.
- Equivalence or uselessness inferred from a non-significant ablation.
- Causal branch attribution inferred from gate mass.
- Tumour localisation inferred from Grad-CAM++.
- Zero-shot generalisation attributed to any target-label calibration or adaptation.

## Author-supplied information still required

Author names, ORCIDs, affiliations, corresponding-author details, target venue, institutional ethics determination, funding, competing interests, CRediT roles, repository URL, archival DOI, acknowledgements, dataset-licensing confirmation and venue-required generative-AI disclosure remain explicit placeholders.

## X3–X7 explainability evidence audit

- Separate CNN Grad-CAM++ and ViT Gradient × Input maps were produced for 848 images across three seeds (5,088 maps); extraction caused zero probability change and zero label change.
- X4 passed all nine prespecified sanity gates. Parameter-randomisation and similarity-to-prior measurements without prespecified decision thresholds remain descriptive.
- Under the locked X5 protocol, CNN, ViT, equal-composite and gate-composite maps were evaluated by deletion, insertion and transformation stability; the strong centre-prior baseline prevents treating visual concentration alone as evidence of method-specific faithfulness.
- X6 applied the unchanged X5 protocol to 373 duplicate-disjoint cross-source images. Cross-source changes in prediction confidence, branch disagreement and attribution measurements are reported descriptively and do not establish one causal failure mechanism.
- Original attention-export X2 remains a recorded failure of the prespecified numerical-neutrality gate. X2b replaced it with an SDPA-preserving token-attribution method; the failed method is not presented as accepted evidence.
- No explanation is described as a segmentation, anatomical localisation, clinical rationale or attention map.
