# Phase X5 post-change report

```text
FILES CREATED:
  publication_results/explainability/x5_faithfulness/ (12 immutable Kaggle result files)
  publication_results/explainability/x5_faithfulness/KAGGLE_OUTPUT_SHA256.json
  publication_results/explainability/x5_faithfulness/X5_LANDING_PROVENANCE.json
  publication_results/explainability/x5_faithfulness/X5_POST_CHANGE_REPORT.md

FILES MODIFIED:
  none

COMMANDS RUN:
  Re-hashed all 16 files listed in KAGGLE_OUTPUT_SHA256.json.
  Compared the downloaded and local protocol-lock and scale-amendment hashes.
  Ran src/publication/explainability/test_faithfulness.py.
  Validated full-result CSV schemas, row counts, seed/image alignment, finite values,
  curve lengths, transformation membership, and bootstrap table membership.
  Ran the Phase 0 preservation check on the project host.
  Copied the 12 Kaggle result files into a previously absent destination and
  re-hashed every copied file against its incoming source.

TEST RESULTS:
  Kaggle output hashes: 16/16 PASS.
  X5 unit tests: 40/40 PASS.
  Full scope: 848 images x 3 seeds x 6 methods; 2,544 aligned image-seed rows PASS.
  Stability table: 15,264 rows = 2,544 x 6 transformations PASS.
  Bootstrap table: 12 rows = 2 quantities x 6 methods PASS.
  Curves: 12 curve columns, each containing 21 locked points PASS.
  Preservation: 150 checked, 0 altered PASS.
  Landing copy hashes: 12/12 PASS.

REPRODUCED BASELINE RESULTS:
  Not applicable. X5 evaluates explanation faithfulness and stability and does not
  recompute classification accuracy or macro-F1. Previously verified FusionV2
  performance remains unchanged; the preservation audit found 0 altered files.

NEW RESULTS:
  Runtime: 11,680.004 s (3.244 h) on a Tesla T4.
  Deletion AUC (lower is better): CNN Grad-CAM++ 0.3649 [0.3611, 0.3687];
  ViT Gradient x Input 0.5182 [0.5119, 0.5246]; equal composite 0.3846
  [0.3801, 0.3891]; gate-weighted composite 0.4858 [0.4797, 0.4918];
  random 0.5625 [0.5578, 0.5671]; centre prior 0.3525 [0.3490, 0.3558].
  Insertion AUC (higher is better): CNN Grad-CAM++ 0.7186 [0.7146, 0.7227];
  ViT Gradient x Input 0.5814 [0.5760, 0.5867]; equal composite 0.7115
  [0.7072, 0.7157]; gate-weighted composite 0.6203 [0.6153, 0.6253];
  random 0.5662 [0.5615, 0.5709]; centre prior 0.7148 [0.7107, 0.7187].

WARNINGS/LIMITATIONS:
  These metrics measure the model's response to masking, not anatomical
  localisation, tumour boundaries, clinical validity, or explanation correctness.
  The centre-prior baseline achieved the lowest deletion AUC and nearly matched
  CNN Grad-CAM++ on insertion AUC. The gate-weighted composite did not outperform
  CNN Grad-CAM++ or the equal composite on either headline metric. Therefore X5
  does not support claiming that gate weighting produces the best explanation.
  CNN-ViT agreement remains descriptive; gate mass is branch utilisation rather
  than causal importance. The run used transformers 5.0.0 on Kaggle while the
  locked local environment recorded 4.57.6; this environment difference remains
  disclosed. Cross-source explanation evaluation belongs to X6 and was not run.

ACCEPTANCE STATUS: PASS

NEXT PHASE: not started
```
