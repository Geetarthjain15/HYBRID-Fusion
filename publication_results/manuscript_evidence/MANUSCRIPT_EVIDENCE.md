# Phase 3D manuscript evidence

Base paper: Kumar Napa et al., *MethodsX* 16 (2026) 103769, DOI 10.1016/j.mex.2025.103769. PDF SHA-256: `ea8fac3fa57c03419f1ad3a160ddab86ad181b1d2440b2af5f185c80249b4aee`.

## Contextual comparison

| Model/evaluation | n | Accuracy | F1 | Comparability |
|---|---:|---:|---:|---|
| FusionV2 — conventional/original held-out image split | 1052 | 99.14 ± 0.10 | 99.12 ± 0.09 | contextual only versus BRAIN-META |
| FusionV2 — duplicate-clean held-out image split | 848 | 98.27 ± 0.07 | 98.28 ± 0.04 | primary internal protocol; not identical to BRAIN-META |
| FusionV2 — duplicate-disjoint cross-source image subset | 373 | 69.53 ± 1.98 | available in machine-readable metrics; not used for base-paper headline comparison | different source and subset; not comparable to BRAIN-META test split |
| BRAIN-META (XGBoost meta-learner) — paper-reported 80/20 split; 5712 training and 1311 testing images | 1311 | 97.10 | 96.94 | contextual only; different preprocessing, split and ten-model stacking pipeline |

The BRAIN-META values are verified from the supplied paper: its algorithm states an 80/20 split on PDF page 5; the dataset section reports 5,712 training and 1,311 testing images on page 5; and the XGBoost result reports accuracy 0.9710 and F1 0.9694 on page 8. This is not a controlled head-to-head comparison.

## What this study does not establish

- Patient-independent evaluation.
- Clinical readiness or diagnostic safety.
- Generalization across hospitals, scanners, populations, or acquisition protocols.
- Statistical superiority over BRAIN-META.
- Equivalence or zero utility for components with nonsignificant ablations.
- Causal or anatomical explanation from gate mass.

## Evidence-controlled component conclusions

- **learned gate**: no detectable accuracy benefit under this protocol (observed Δ accuracy +0.0000 pp).
- **multi-scale taps**: no detectable accuracy benefit under this protocol (observed Δ accuracy -0.0393 pp).
- **auxiliary losses**: suggestive but inconclusive (observed Δ accuracy -0.5503 pp).
- **EMA**: no detectable accuracy benefit under this protocol (observed Δ accuracy +0.0393 pp).
- **ViT branch contribution**: supported by consistent accuracy evidence (observed Δ accuracy -2.5943 pp).
- **CNN branch contribution**: suggestive but inconclusive (observed Δ accuracy -0.2358 pp).
- **TTA**: no detectable accuracy benefit under this protocol (observed Δ accuracy -0.0786 pp).

See the accompanying CSV matrices for manuscript-ready assumptions, limitations, contribution interpretations, and allowed/prohibited claims.
