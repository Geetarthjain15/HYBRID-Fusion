# Duplicate-disjoint cross-source ablation analysis

**Status: PASS.** Existing predictions only; no training, tuning, selection, or thresholding.

## Performance on 373 duplicate-disjoint images

| Variant | Accuracy mean ± seed SD (%) | Macro-F1 mean ± seed SD (%) |
|---|---:|---:|
| `full_fusion_v2` | 69.53 ± 1.98 | 68.69 ± 2.21 |
| `no_gate` | 69.26 ± 1.88 | 68.52 ± 1.54 |
| `no_multiscale` | 67.92 ± 3.20 | 66.25 ± 3.07 |
| `no_auxiliary_loss` | 70.15 ± 1.57 | 69.52 ± 2.08 |
| `no_ema` | 67.92 ± 1.48 | 66.60 ± 1.21 |
| `cnn_branch_only` | 44.50 ± 3.09 | 45.07 ± 3.24 |
| `vit_branch_only` | 68.54 ± 1.21 | 67.23 ± 1.75 |
| `no_tta` | 69.71 ± 1.93 | 68.93 ± 2.28 |

## Statistical scope

Exactly 21 seed-matched McNemar tests were Holm-corrected together; 3 remained significant at 0.05. Paired image-level bootstrap intervals use 10,000 class-stratified resamples per comparison. Repeated predictions of the same 373 images are never treated as independent cases.

## Interpretation boundaries

- These results evaluate frozen outputs on one duplicate-disjoint external collection.
- External labels were not used to choose a component or tune any setting.
- Seed SD is not a confidence interval; bootstrap intervals are image-level, not patient-level.
- A nonsignificant difference is not equivalence or proof that a component is unnecessary.
