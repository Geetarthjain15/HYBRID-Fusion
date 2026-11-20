# FusionV2 Grad-CAM++ demo validation

## Scope

This extension adds an isolated research-demonstration application. It does not
modify or retrain FusionV2, alter a checkpoint, or change any existing result.

## Evidence and checks

- The seed-42 recovered checkpoint SHA-256 was verified against
  `publication_results/ablations/checkpoint_recovery/import_manifest.json`.
- The checkpoint loaded strictly into an architecture-identical offline
  FusionV2 constructor.
- A clean-test image was evaluated with the verified horizontal-flip TTA
  prediction path. The predicted label matched the stored seed-42 prediction.
- Grad-CAM++ completed for each of the four possible class targets.
- Every map was finite, had shape 224 x 224, and was normalized to [0, 1].
- One recorded clean-test image from each ground-truth class was then checked
  end to end with seed 42. All four predicted labels matched the stored
  prediction CSV, all learned gate rows summed to one, and all Grad-CAM++ maps
  were finite and normalized. The maximum absolute probability difference was
  below `5e-7` for every checked image (well inside the established `1e-4`
  checkpoint-reproduction tolerance).
- The Streamlit server started successfully in headless mode.
- Four lightweight preprocessing/shape/error-handling tests passed.

## Interpretation boundary

The prediction is produced by the complete learned-gate FusionV2 model. The
Grad-CAM++ target is a selected fused output logit, but its spatial activation
source is the final EfficientNet feature stage. Accordingly, the visualization
is described as **CNN-path Grad-CAM++**. It does not spatially explain the ViT
branch, establish tumor localization, validate anatomical reasoning, or support
clinical use. No localization mask or clinician annotation was used.

## Status

PASS for demonstration use under the stated interpretation boundary.

## Local interface extension

The local Streamlit interface was extended and revalidated on 2026-09-25. It
now provides the uploaded image, exact four-class softmax outputs, top-two
margin, normalized predictive entropy, learned CNN/ViT gate mass, on-demand
Grad-CAM++, a downloadable inference record, three-seed aggregate evaluation
metrics, per-class metrics, selected-seed clean-split metrics and a confusion
matrix. Dataset-level values are loaded from the accepted Phase 1 JSON rather
than retyped as application constants. At the user's explicit request, the
top warning, confidence explanatory caption, cross-source warning banner and
separate interpretation/limitations tab were subsequently removed from the
demonstration UI; the underlying metrics remain available in the metrics table
and no inference behavior changed. Five lightweight application tests passed
after this UI-only revision.
