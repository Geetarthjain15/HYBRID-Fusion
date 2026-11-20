# X5 Scale-Decision Amendment

Amends `x5_protocol_lock.json` **without modifying it**. The lock file is
byte-identical to the version hashed before the dry run; this amendment is
a separate document with its own SHA-256.

| field | value |
|---|---|
| protocol lock SHA-256 | `a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5` |
| protocol lock modified | no |
| protocol parameters changed | none |
| decided | 2026-09-26T23:54:01.789907+00:00 |
| approved by | user, explicitly, for the full-set Kaggle GPU evaluation |

## Decision

| field | value |
|---|---|
| scope | **FULL SET** |
| images | 848 |
| seeds | [42, 43, 44] |
| image x seed pairs | 2544 |
| methods | 6 (all six locked) |
| subset | NONE - no subsetting, so no selection decision arises and no faithfulness value can influence which images are evaluated |
| compute | Kaggle GPU (Tesla T4) |

Methods: `cnn_gradcampp`, `vit_grad_x_input`, `composite_equal`, `composite_gate_weighted`, `baseline_random`, `baseline_centre_prior`.

## Basis for the decision

Taken from **measured runtime and storage only**. No faithfulness value
existed when it was made, so the evaluated set cannot be a function of the
answer. Because the decision is the full set, no selection arises at all.

| quantity | value |
|---|---|
| measured CPU, per image, all methods + stability | 73.8 s |
| projected forward passes | 717,408 |
| projected CPU | 52.2 h |
| **projected Kaggle T4** | **2.97 h** |
| projected storage | 17.2 MB |

> the GPU figure is EXTRAPOLATED from the X3 measured CPU-to-T4 ratio on a different workload (attribution, not masked forward passes). It is a projection, not a measurement, and the actual runtime is recorded when the run completes.

The locked protocol records batch size rather than fixing it. the runner batches the 47 masked variants of one image into a single forward, which is a scheduling choice, not a protocol change; the batch size actually used is recorded with the results..

## Output paths

`publication_results/explainability/x5_faithfulness/`

- `x5_faithfulness.json`
- `X5_FAITHFULNESS_REPORT.md`
- `per_image_curves.csv`
- `stability.csv`
- `bootstrap_effects.csv`
- `sha256_manifest.json`

The runner refuses to overwrite any existing path.

## Checksum verification steps

- 1. Before the run: faithfulness.py verifies x5_protocol_lock.json hashes to a553c93b... and halts if it does not.
- 2. Before the run: all 10 code files and 3 checkpoints are verified against UPLOAD_MANIFEST.json in the notebook, and the X1 manifest is verified with its 848 / 243-228-116-261 counts.
- 3. After the run: KAGGLE_OUTPUT_SHA256.json records a digest for every produced file.
- 4. On return to the project host: every downloaded file is re-hashed against KAGGLE_OUTPUT_SHA256.json.
- 5. On return: the 150-file Phase 0 preservation baseline is re-checked, which Kaggle cannot do (--skip-preservation is passed there).
- 6. On return: the protocol lock hash recorded in the results is compared against a553c93b... to prove the protocol did not change between lock and run.

## Prohibited

- changing any locked protocol parameter
- selecting a subset using faithfulness values
- reporting these results as external or cross-source evidence (Phase X6)
- any anatomical, tumour or clinical claim

**Status: APPROVED - full X5 run authorised but NOT yet executed**
