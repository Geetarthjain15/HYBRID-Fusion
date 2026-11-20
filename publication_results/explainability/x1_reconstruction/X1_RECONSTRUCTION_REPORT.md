# Phase X1 - clean-test reconstruction and identity verification

Generated 2026-09-26T11:33:23.027936+00:00.

**Status: PASS.** No training, no model change, no modification to any dataset, checkpoint or existing prediction file.

## Why reconstruction was necessary

`dataset_clean/Testing` does not exist on this machine. The 848 duplicate-clean test images live under `dataset/`, distributed across both original splits by the duplicate-clean repartition - which is why searching `dataset/Testing` alone finds only a fraction of them. They were rebuilt by manifest lookup, and their identity was then proven by outcome rather than assumed from filenames.

## Reconstruction

| Field | Value |
|---|---|
| Source manifest | `dataset_clean/split_manifest.json` |
| Manifest SHA-256 | `a02bcb62847ce1b9146a3e769f41f88ec1afce5a46d03fb5aaffb0ef43a0e6d2` |
| Testing entries | 848 |
| Files indexed under `dataset/` | 7023 |
| Ambiguous filenames | 0 |
| Images reconstructed | 848 |
| Content-duplicate groups | 0 |
| Destination | `publication_runs/explainability/reconstructed_clean_test` |

Per-class counts: {'glioma': 243, 'meningioma': 228, 'notumor': 116, 'pituitary': 261}.

Per-file SHA-256, byte size, source path and destination path for all 848 images are recorded in `RECONSTRUCTION_MANIFEST.json`.

## Identity verification

Each reconstructed image was scored under every verified checkpoint using the authoritative evaluation transform imported from `src/train_ablation.py` and horizontal-flip TTA from `src/fusion_v2.py`, then compared filename-by-filename against the recorded predictions. The recorded CSVs were read only.

| Seed | n | Filename mm | True-label mm | Pred-label mm | Acc recomputed | Acc recorded | Diff (pp) | max abs dp | mean abs dp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 848 | 0 | 0 | 0 | 0.982311 | 0.982311 | 0.0000 | 4.82e-06 | 2.82e-07 |
| 43 | 848 | 0 | 0 | 0 | 0.982311 | 0.982311 | 0.0000 | 3.93e-06 | 2.74e-07 |
| 44 | 848 | 0 | 0 | 0 | 0.983491 | 0.983491 | 0.0000 | 3.99e-06 | 2.87e-07 |

Checkpoints used, digests re-verified against the recovery manifest before loading:

- seed 42: `9b328ba83df5d44b542e19faae68fd1493fee61e0c5b80e493c2f8a387ed592e`
- seed 43: `9f97b6aa0c9ec3f1124e593c8c29e11d31f4aef8bc08e20a40e7ed1f29e80f25`
- seed 44: `4155a2b71df99a6159e1e4a5dbe635f4e9ea7852a6704d804dd1fb5d43aca434`

Reference prediction files (read-only):

- seed 42: `runs_v2_clean/fusion_v2_seed42/preds_test.csv` (`247e6494d7315856a5fbb2a26b2eed78`)
- seed 43: `runs_v2_clean/fusion_v2_seed43/preds_test.csv` (`bb142d3d33528fdb0d5ad915af384010`)
- seed 44: `runs_v2_clean/fusion_v2_seed44/preds_test.csv` (`fe00252503bf19b210540d84ba788f3c`)

## Floating-point differences

Probability differences arise only from non-deterministic reduction order in CPU kernels between the original run and this one. They are bounded by 1e-04, far below any value that could change a predicted label, and every predicted label matched exactly.

## Protocol

| Field | Value |
|---|---|
| Transform | imported from `train_ablation.eval_transform` |
| TTA | horizontal flip, `fusion_v2.tta_logits` |
| Class order | glioma, meningioma, notumor, pituitary |
| Device | cpu |
| Probability tolerance | 1e-04 |
| Accuracy tolerance | 0.00 pp |

## Acceptance checks

| Check | Result |
|---|---|
| manifest 848 unique filenames | **PASS** |
| no ambiguous source filenames | **PASS** |
| every scored image present in reference | **PASS** |
| all seeds zero true label mismatch | **PASS** |
| all seeds zero pred label mismatch | **PASS** |
| all seeds accuracy diff zero pp | **PASS** |
| all seeds prob diff within tolerance | **PASS** |
| expected image count compared | **PASS** |
| class order matches documented | **PASS** |
| reconstructed 848 | **PASS** |
| no content duplicates | **PASS** |
| per class counts match manifest | **PASS** |
| all seeds covered | **PASS** |
| all seeds zero filename mismatch | **PASS** |

**Overall: PASS.** Elapsed 1066.2 s.

## Scope

This phase establishes that the reconstructed images are the inputs behind the published clean-test results. It establishes nothing about patient independence, about the external set, or about any explanation method. Phase X2 has not started.
