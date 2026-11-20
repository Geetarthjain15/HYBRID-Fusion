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
