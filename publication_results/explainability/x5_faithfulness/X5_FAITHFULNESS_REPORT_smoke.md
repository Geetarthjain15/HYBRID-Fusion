# Phase X5 - Faithfulness and Stability

Protocol lock `a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5`.

Mode `smoke`, 8 images x 1 seeds, device `cuda`, 0.6 min.

## Deletion AUC (lower indicates the ranking removes evidence faster)

| method | mean | 95% CI |
|---|---|---|
| `cnn_gradcampp` | 0.4211 | [0.3583, 0.4836] |
| `vit_grad_x_input` | 0.6112 | [0.5673, 0.6553] |
| `composite_equal` | 0.4411 | [0.3696, 0.5124] |
| `composite_gate_weighted` | 0.5680 | [0.5120, 0.6241] |
| `baseline_random` | 0.6474 | [0.6037, 0.6910] |
| `baseline_centre_prior` | 0.3903 | [0.3503, 0.4301] |

## Insertion AUC (higher indicates the ranking restores evidence faster)

| method | mean | 95% CI |
|---|---|---|
| `cnn_gradcampp` | 0.7230 | [0.7048, 0.7411] |
| `vit_grad_x_input` | 0.6876 | [0.6676, 0.7075] |
| `composite_equal` | 0.7233 | [0.7081, 0.7385] |
| `composite_gate_weighted` | 0.6883 | [0.6707, 0.7059] |
| `baseline_random` | 0.6337 | [0.5830, 0.6844] |
| `baseline_centre_prior` | 0.7018 | [0.6698, 0.7339] |

Bootstrap: image-level, stratified by true class, 10000 iterations, seed 20260927.

## Interpretation limits

These are measures of how well each ranking predicts this model's own
behaviour under masking. They are not anatomical localisation, not tumour
detection, and not clinical validity. CNN-ViT agreement is descriptive and
is not evidence that either map is correct. Gate mass is branch
utilisation, not causal importance. No attention matrix was read.
