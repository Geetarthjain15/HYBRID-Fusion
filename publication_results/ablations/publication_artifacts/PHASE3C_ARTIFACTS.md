# Phase 3C publication artifacts

All values and figures are generated from the accepted Phase 3B machine-readable evidence. No training, tuning, or model modification occurred.

## Primary ablation table

| Model | Accuracy mean ± seed SD (%) | Macro-F1 mean ± seed SD (%) | Observed Δ accuracy (pp) | Holm-significant seeds | Interpretation |
|---|---:|---:|---:|---:|---|
| Full FusionV2 | 98.2704 ± 0.0681 | 98.2761 ± 0.0371 | +0.0000 | reference | reference |
| No gate | 98.2704 ± 0.1801 | 98.2270 ± 0.1255 | +0.0000 | 0 | no detectable accuracy benefit under this protocol |
| No multi-scale | 98.2311 ± 0.2043 | 98.1875 ± 0.2136 | -0.0393 | 0 | no detectable accuracy benefit under this protocol |
| No auxiliary loss | 97.7201 ± 0.4766 | 97.7334 ± 0.3873 | -0.5503 | 0 | suggestive but inconclusive |
| No EMA | 98.3097 ± 0.2968 | 98.3517 ± 0.2406 | +0.0393 | 0 | no detectable accuracy benefit under this protocol |
| CNN only | 95.6761 ± 0.3404 | 95.8981 ± 0.4026 | -2.5943 | 3 | supported by consistent accuracy evidence |
| ViT only | 98.0346 ± 0.1362 | 97.9598 ± 0.1599 | -0.2358 | 0 | suggestive but inconclusive |
| No TTA | 98.1918 ± 0.2455 | 98.2056 ± 0.1922 | -0.0786 | 0 | no detectable accuracy benefit under this protocol |

Seed SD describes spread across three training seeds; it is not a confidence interval. Observed differences are not labelled important or negligible because no prespecified margin exists.

## Figures

- `ablation_accuracy`: seed means, seed SD, and individual seed results.
- `reference_confusion`: summed prediction counts across seeds; repeated images are explicitly labelled.
- `ablation_confusion_comparison`: normalized class-error patterns for informative controls.
- `paired_disagreements`: paired reference wins and losses by seed.
- `reference_reliability`: seed-specific calibration curves with bin counts.
- `gate_mass_by_class`: descriptive gate utilization only.

## Interpretation guardrails

Calibration and gate comparisons are descriptive. Gate mass indicates utilization, not causal contribution, anatomical attention, or clinical explanation. Non-significance is not equivalence. These results concern image-level evaluation on one duplicate-clean benchmark partition and do not establish patient independence, clinical readiness, or cross-institutional generalization.
