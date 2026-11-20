# Phase X5 - Faithfulness and Stability

Protocol lock `a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5`.

Mode `full`, 848 images x 3 seeds, device `cuda`, 194.7 min.

## Deletion AUC (lower indicates the ranking removes evidence faster)

| method | mean | 95% CI |
|---|---|---|
| `cnn_gradcampp` | 0.3649 | [0.3611, 0.3687] |
| `vit_grad_x_input` | 0.5182 | [0.5119, 0.5246] |
| `composite_equal` | 0.3846 | [0.3801, 0.3891] |
| `composite_gate_weighted` | 0.4858 | [0.4797, 0.4918] |
| `baseline_random` | 0.5625 | [0.5578, 0.5671] |
| `baseline_centre_prior` | 0.3525 | [0.3490, 0.3558] |

## Insertion AUC (higher indicates the ranking restores evidence faster)

| method | mean | 95% CI |
|---|---|---|
| `cnn_gradcampp` | 0.7186 | [0.7146, 0.7227] |
| `vit_grad_x_input` | 0.5814 | [0.5760, 0.5867] |
| `composite_equal` | 0.7115 | [0.7072, 0.7157] |
| `composite_gate_weighted` | 0.6203 | [0.6153, 0.6253] |
| `baseline_random` | 0.5662 | [0.5615, 0.5709] |
| `baseline_centre_prior` | 0.7148 | [0.7107, 0.7187] |

Bootstrap: image-level, stratified by true class, 10000 iterations, seed 20260927.

## Interpretation limits

These are measures of how well each ranking predicts this model's own
behaviour under masking. They are not anatomical localisation, not tumour
detection, and not clinical validity. CNN-ViT agreement is descriptive and
is not evidence that either map is correct. Gate mass is branch
utilisation, not causal importance. No attention matrix was read.
