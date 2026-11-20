# Phase X2b - SDPA-Preserving ViT Token Attribution

**Overall: PASS**

| field | value |
|---|---|
| phase | `X2b` |
| mode | `neutrality_full` |
| created_utc | `2026-09-26T16:23:36.553230+00:00` |
| attn_implementation | `sdpa` |
| attribution_method | `gradient_x_input` |
| attribution_site | `vit.vit.embeddings output, patch tokens 1:197` |
| target_rule | `predicted class from recorded inference` |
| batch_size | `16` |
| n_images | `848` |
| neutrality_tolerance | `1e-06` |
| class order | `glioma, meningioma, notumor, pituitary` |

## Why the embedding site

Measured gradient magnitude at the two candidate sites, seed 42:

| site | sum of abs gradient | nonzero elements |
|---|---|---|
| final `last_hidden_state[:, 0]` (CLS) | 7.017653e-01 | - |
| final `last_hidden_state[:, 1:]` (196 patches) | 0.000000e+00 | 0 / 150528 |

`fusion_v2.py:91` reads only the CLS token, so the final patch
tokens feed no downstream output and their gradient is identically
zero. Attribution is therefore taken at the patch-embedding output.

## Per-seed neutrality

| seed | max abs diff single | max abs diff TTA | over tol | label changes | acc instr | acc recorded | acc diff pp | maps valid | s |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.000000e+00 | 0.000000e+00 | 0 | 0 | 98.2311% | 98.2311% | 0.0000 | 848/848 | 628 |
| 43 | 0.000000e+00 | 0.000000e+00 | 0 | 0 | 98.2311% | 98.2311% | 0.0000 | 848/848 | 684 |
| 44 | 0.000000e+00 | 0.000000e+00 | 0 | 0 | 98.3491% | 98.3491% | 0.0000 | 848/848 | 667 |

## Acceptance checks

| check | result |
|---|---|
| A1_sdpa_active_throughout | PASS |
| A2_eager_route_unreachable | PASS |
| A3_zero_pred_label_changes | PASS |
| A4_accuracy_diff_zero_pp | PASS |
| A5_max_prob_diff_within_tol | PASS |
| A6_all_maps_valid | PASS |
| A7_no_retained_state | PASS |
| A8_exact_restoration | PASS |
| A10_identity | PASS |
| site_correction_still_required | PASS |
| A9_preservation | PASS |

Preservation: 150 baseline artifacts re-hashed, 0 altered.

## Interpretation limits

These 14x14 maps are a local, first-order measure of how the fused
logit for the selected class responds to perturbation of the ViT
patch-embedding representation. They are model sensitivity only.

- No attention matrix was read; no attention-based claim is made.
- The maps are not segmentation, detection or anatomical
  localisation, and no region of them may be described as tumour.
- They are not a causal, exact or complete explanation of any
  prediction, and they do not demonstrate clinically correct
  reasoning.
- Faithfulness is not established here. It is measured only by the
  prespecified deletion/insertion and stability protocol of Phase
  X5, which has not been run.
- The Phase X2 eager-attention failure remains recorded and is not
  repaired, superseded or invalidated by this phase.
- FusionV2 architecture, weights, gate, training, preprocessing,
  TTA policy and all published metrics are unchanged.
