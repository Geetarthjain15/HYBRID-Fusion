# Phase X4 - Explanation Sanity Checks

**Overall: PASS**

Mode `full`, 16 images, checkpoint seed 42, device cpu, attention `sdpa`.

## Gates (1-8)

| check | result | measured |
|---|---|---|
| C1_deterministic_repeat | PASS | max abs diff cnn 0.000e+00 / vit 0.000e+00 |
| C2_batch_vs_single | PASS | rel cnn 2.485e-06 / vit 8.481e-07, target classes identical: True |
| C3_target_class_sensitivity | PASS | all images differ from the non-predicted class map |
| C4_finite_and_nonconstant | PASS | all maps finite, non-constant, non-zero |
| C5_native_shapes | PASS | cnn 7x7, vit 14x14 |
| C6_orientation_preserved | PASS | raster order, peak correspondence, upsample orientation |
| C7_hook_and_gradient_cleanup | PASS | 0 hooks after exit, grads clean True, state_dict unchanged True |
| C8_cnn_equivalence_le_1e-5 | PASS | max abs diff 0.000e+00 (tolerance 1e-5) |

## Descriptive baselines (9-11) - no threshold, no verdict

These are observed similarity values between the real maps and three
reference maps. No decision rule was prespecified for them, so they are
reported as measurements only. They are not evidence of faithfulness,
which is measured solely by the Phase X5 protocol.

| branch | comparison | Pearson mean | Spearman mean | normalised L1 mean |
|---|---|---|---|---|
| cnn | parameter randomized | 0.0552 | 0.0847 | 0.4759 |
| cnn | random map | -0.0002 | 0.0400 | 0.4343 |
| cnn | centre prior | 0.6700 | 0.6519 | 0.2066 |
| vit | parameter randomized | 0.0384 | 0.0223 | 0.2020 |
| vit | random map | 0.0059 | -0.0008 | 0.2800 |
| vit | centre prior | 0.0201 | 0.0479 | 0.3330 |

## Scope

X4 tests technical validity only. It does not compute deletion,
insertion, confidence drop, stability, CNN-ViT agreement, composites or
external-source results, and it makes no claim about anatomical
localisation, tumour detection or clinical validity. No region of any
map may be described as tumour. No attention matrix was read.
