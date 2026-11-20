# Phase X3 (amended) - smoke

**Overall: PASS (PRESERVATION NOT VERIFIED ON THIS HOST)**

| check | result |
|---|---|
| S1_sdpa_throughout | PASS |
| S2_eager_route_unreachable | PASS |
| S3_cnn_maps_valid_7x7 | PASS |
| S4_vit_maps_valid_14x14 | PASS |
| S5_gradcam_equivalence_le_1e-5 | PASS |
| S6_neutrality | PASS |
| S7_zero_label_changes | PASS |
| S8_targets_synchronized | PASS |
| S9_identity | PASS |
| no_retained_state | PASS |
| preservation | FAIL |

| seed | dp single | dp TTA | label changes | CNN valid | ViT valid | Grad-CAM++ equiv max | s/img |
|---|---|---|---|---|---|---|---|
| 42 | 0.000e+00 | 0.000e+00 | 0/0 | 8/8 | 8/8 | 3.517e-06 | 0.036 |

Separate CNN (7x7) and ViT (14x14) maps only. No composite. No attention
matrix read. Maps are model sensitivity, not localisation; no region may be
described as tumour. Faithfulness, stability, CNN-ViT agreement and external
results are not computed in X3.
