# Phase X4 - Explanation Sanity Checks

**Overall: PASS**

Mode `smoke`, 4 images, checkpoint seed 42, device cpu, attention `sdpa`.

## Gates (1-8)

| check | result | measured |
|---|---|---|
| C1_deterministic_repeat | PASS | max abs diff cnn 0.000e+00 / vit 0.000e+00 |
| C2_batch_vs_single | PASS | rel cnn 3.690e-06 / vit 6.221e-07, target classes identical: True |
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
| cnn | parameter randomized | 0.1101 | 0.1679 | 0.4940 |
| cnn | random map | 0.1101 | 0.0485 | 0.4277 |
| cnn | centre prior | 0.8101 | 0.7890 | 0.1956 |
| vit | parameter randomized | -0.0172 | 0.0290 | 0.2648 |
| vit | random map | -0.0289 | -0.0143 | 0.2871 |
| vit | centre prior | 0.0320 | 0.0539 | 0.3133 |

## Scope

X4 tests technical validity only. It does not compute deletion,
insertion, confidence drop, stability, CNN-ViT agreement, composites or
external-source results, and it makes no claim about anatomical
localisation, tumour detection or clinical validity. No region of any
map may be described as tumour. No attention matrix was read.
