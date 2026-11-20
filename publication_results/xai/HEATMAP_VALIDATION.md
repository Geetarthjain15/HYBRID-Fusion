# FusionV2 Grad-CAM++ faithfulness audit

## Result

The implementation targets the selected fused-model logit and the final spatial EfficientNet stage. A red region on a no-tumor prediction is expected because every nonconstant map is normalized independently: red is the maximum support for the selected **no-tumor** class within that image. It is not evidence that a tumor exists at that location.

## Deterministic examples

| Source class | Prediction | Probability | Deletion AUC CAM/random | Insertion AUC CAM/random | Flip correlation |
|---|---|---:|---:|---:|---:|
| glioma | glioma | 0.9362 | 0.3981/0.7949 | 0.7645/0.6324 | 0.9778 |
| meningioma | meningioma | 0.9288 | 0.3045/0.6960 | 0.8529/0.7129 | 0.9361 |
| notumor | notumor | 0.9413 | 0.5165/0.4972 | 0.7311/0.6700 | 0.9411 |
| pituitary | pituitary | 0.9143 | 0.1753/0.2656 | 0.5811/0.3589 | 0.9821 |

## Interpretation

Deletion and insertion compare the Grad-CAM++ ranking with a fixed random control. Lower deletion AUC and higher insertion AUC are directionally consistent with a more faithful ranking, but four examples do not establish general validity.

## Limits

- Localization accuracy: **not available** because no verified lesion masks or boxes exist.
- The map is CNN-path sensitivity, not a complete explanation of the gated CNN-ViT model.
- The map does not explain horizontal-flip TTA probability averaging.
- Per-image min-max normalization prevents color intensity from being compared across images.
- No-tumor must be labelled as selected-class support, not as a tumor heatmap.
