# Phase X6 implementation and Kaggle hand-off

## Status

The Phase X6 implementation, exact 373-image reconstruction, local smoke test,
and self-contained Kaggle package are complete. The full 373 × 3 × 6 evaluation
has **not** been run and no full-grid X6 scientific result is claimed.

## Locked scope

- X5 protocol SHA-256: `a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5`
- Selector: `results/cross_source.json`, entries whose `kind` is `disjoint`
- Selector SHA-256: `e8a27a92495e2f44c2ef7af655128a8ea83574cbb271e6b09ce7d4f99b4c3cbb`
- Images: 373 (glioma 52; meningioma 122; no-tumour 88; pituitary 111)
- Seeds: 42, 43, 44
- Methods: the six methods locked in X5, unchanged
- Full image-seed pairs: 1,119
- External labels: scoring and stratified uncertainty only; never selection or tuning

## Local validation

- Structural/unit tests: 24/24 passed.
- CPU smoke: eight images (two per class), seed 42, all six methods.
- Instrumented-versus-plain single-forward maximum probability difference: `0.0`.
- Recomputed-TTA-versus-recorded maximum probability difference: `1.4541e-06`.
- Recorded-label changes: 0.
- Preservation: 150 checked, 0 altered.
- Smoke runtime: 507.34 seconds on CPU.

The smoke sample is a pipeline check only and is not a scientific subset result.

## Compute decision

The measured X5 runtime projects the full X6 grid to approximately 1.43 hours on
one Tesla T4. CPU scaling projects approximately 22.94 hours. The full run is
therefore packaged for Kaggle and was not started locally.

## Final Kaggle package

- Archive: `publication_runs/explainability/kaggle_x6_v2/fusion_v2_x6_upload_v2.zip`
- Archive SHA-256: `362f7b9a39f102c2622ffee7b6f8a285db4422173972562fd3587210e5735e05`
- Archive entries: 401 (400 hash-pinned inputs plus `UPLOAD_MANIFEST.json`)
- Reconstructed images: 373
- Verified checkpoints: 3
- Notebook: `publication_runs/explainability/kaggle_x6_v2/kaggle_x6_notebook.ipynb`
- Instructions: `publication_runs/explainability/kaggle_x6_v2/KAGGLE_X6_STEPS.md`

The notebook verifies every input hash, installs `grad-cam` only if absent, runs
the 24 tests, runs one GPU smoke test, runs the full grid once, checks the
recorded lock/scope/neutrality, and creates `x6_results.zip` with per-file
SHA-256 values.

## Outputs expected after Kaggle

- `x6_cross_source.json`
- `X6_CROSS_SOURCE_REPORT.md`
- `per_image_curves.csv`
- `stability.csv`
- `shift_effects.csv`
- smoke counterparts and `KAGGLE_OUTPUT_SHA256.json`

Shift confidence intervals use unique images as the bootstrap unit after
averaging the three seed measurements for each filename. Clean and external
images remain separate and are never pooled into one evaluation set.

## Interpretation limits

X6 may establish measured associations between source shift and explanation
behaviour. It cannot establish that an explanation change caused the accuracy
drop, anatomically localise tumours, or demonstrate robustness across hospitals,
scanners, or patient populations.

**Full-run acceptance status: not available — full Kaggle run not started.**
