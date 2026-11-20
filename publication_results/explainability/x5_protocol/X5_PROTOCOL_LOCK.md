# Phase X5 Protocol Lock

**Written before any faithfulness value was computed.**

Version `1.0`. Every parameter below is fixed. Changing any of them after a result is observed invalidates this lock, and the SHA-256 recorded alongside it is what makes that detectable.

## Methods

| id | native grid | source |
|---|---|---|
| `cnn_gradcampp` | 7x7 | X3 CNN Grad-CAM++ at native stage-8 resolution |
| `vit_grad_x_input` | 14x14 | X3 ViT patch-embedding Gradient x Input (the X2b method) |
| `composite_equal` | 14x14 | 0.5 * norm(cnn) + 0.5 * norm(vit), both on the common grid |
| `composite_gate_weighted` | 14x14 | gate[:,0] * norm(cnn) + gate[:,1] * norm(vit) |
| `baseline_random` | 14x14 | deterministic uniform random map, null baseline |
| `baseline_centre_prior` | 14x14 | deterministic isotropic Gaussian centred on the grid, sigma = grid/4, null baseline |

> The adopted plan's X5 section lists ViT attention rollout and gradient-weighted ViT attention. Both were removed by the approved X3 amendment because they require eager attention, which failed the X2 neutrality gate. The ViT method evaluated here is the SDPA-safe Gradient x Input method verified in X2b and X3.

## Grid, resampling, normalisation

| parameter | value |
|---|---|
| common grid | 14x14 |
| CNN 7 -> 14 | bilinear, align_corners=False |
| order | resample FIRST, normalise SECOND |
| normalisation | per-map min-max to [0,1] after resampling |

## Masking

| parameter | value |
|---|---|
| mask value (tensor space) | 0.0 |
| mask value (raw pixel) | 0.5 |
| mask upsample 14 -> 224 | nearest-neighbour, exact 16x16 pixel blocks |

Eval_transform normalises with mean 0.5 and std 0.5, so a raw mid-grey pixel of 0.5 maps to exactly 0.0 in tensor space; masking therefore replaces a region with the dataset channel mean.

## Deletion and insertion

| parameter | deletion | insertion |
|---|---|---|
| start | unmodified image | fully masked image (all cells at the mask value) |
| order | descending map value; ties broken by ascending flat grid index | descending map value; ties broken by ascending flat grid index |
| steps | 20 | 20 |
| score | fused softmax probability of the locked target class | fused softmax probability of the locked target class |

AUC: trapezoidal over the 21 scored points including both endpoints, x axis scaled to [0,1].

Top-k percentages: [1, 5, 10, 20, 50]. Metrics: confidence_drop, prediction_change_rate.

## Stability

| transformation | specification |
|---|---|
| `hflip` | horizontal flip |
| `rot_plus10` | +10 degrees, bilinear, zero fill |
| `rot_minus10` | -10 degrees, bilinear, zero fill |
| `scale_plus5` | +5% scale about centre, bilinear |
| `scale_minus5` | -5% scale about centre, bilinear |
| `noise` | additive Gaussian, sigma=0.01 in normalised tensor space, fixed seed 20260927 |

Map comparison: Spearman rank correlation on the common grid, after inverting the geometric transform on the map.

## Bootstrap

| parameter | value |
|---|---|
| unit | image |
| stratification | by true class |
| iterations | 10000 |
| seed | 20260927 |
| quantities | every headline quantity, reported with 95% percentile CIs |
| paired | methods compared on the same resampled image sets |

## Subset decision

**Status at lock time: OPEN at lock time - to be fixed by the dry run**

taken from measured runtime and storage ALONE. Faithfulness values are not computed in the dry run, so the subset cannot be a function of the answer.

If a subset is required: stratified by true class, proportional allocation, sampling seed 20260927, selected before any faithfulness value is computed.

The decision is appended as a separate, separately hashed amendment; the original lock hash is never rewritten.

## Prohibited in X5

- any anatomical, tumour, lesion or clinical claim
- any attention-based claim (no attention matrix is read)
- treating CNN-ViT agreement as evidence of correctness
- treating gate mass as causal importance
- altering any locked parameter after observing a result
- selecting the evaluated subset using faithfulness values
- external or cross-source evaluation (that is Phase X6)

## Seeds

| purpose | seed |
|---|---|
| model_seeds | [42, 43, 44] |
| random_baseline | 20260927 |
| noise_transform | 20260927 |
| bootstrap | 20260927 |
| subset_sampling_if_needed | 20260927 |
