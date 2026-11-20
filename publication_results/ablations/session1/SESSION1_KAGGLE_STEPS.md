# Session 1 — Kaggle steps

Nothing here has been uploaded or launched. Follow in order.

## 0. Prerequisites

- Kaggle account phone-verified (Settings → Phone Verification). Without it,
  Internet is off in notebooks and the pretrained backbones cannot download.
- The data dataset from the earlier phases already exists as
  `brain-tumor-mri-data`, containing `dataset/` and `dataset_ood/`. Reuse it.
  **Do not** upload `dataset_clean/` — the driver rebuilds and fingerprints it.

## 1. Upload the code (3 min)

The existing `brain-tumor-code` dataset predates Session 1 and lacks the driver.

1. kaggle.com/datasets → **brain-tumor-code** → ⋮ → **New Version**
2. Remove the existing files in the upload panel
3. Drag in `kaggle/upload/fusionv2_code_session1_v1.zip`
4. Version notes: `session 1 ablation driver`
5. **Create**, and wait for processing

Verify the upload before spending GPU hours:

```
sha256sum kaggle/upload/fusionv2_code_session1_v1.zip
# expect: 90ceaa12e28fdfe94e8f38d4352e6a55994422199bade333b46272aa195c708c
```

That hash is also in `kaggle/upload/fusionv2_code_session1_v1.zip.sha256`.

## 2. Create the notebook (2 min)

1. **Code → New Notebook → File → Import Notebook**
2. Upload `kaggle/publication_session1.ipynb`
3. Right panel:
   - **Add Input** → `brain-tumor-code` **and** `brain-tumor-mri-data`
   - **Accelerator** → GPU T4 ×2
   - **Internet** → On
   - **Persistence** → Files only

Confirm `brain-tumor-code` is attached at the **new version** (⋮ → Change
version → latest). Cell 1 asserts the driver is present and fails immediately if
an old version is attached — that assert exists so you find out in 30 seconds
rather than after the grid dies.

## 3. Dry run first (1 min)

Run cells 1–3. Expected from cell 1:

```
code OK: 12 required files present
torch <version> | cuda True | Tesla T4
dataset: ['dataset', 'dataset_ood']
```

Cell 3 lists the nine cells and their output paths, and trains nothing. If the
list is not exactly `no_gate`, `no_multiscale`, `no_auxiliary_loss` × seeds
42/43/44, stop and re-check the attached code version.

## 4. Launch (~7 h unattended)

**Save Version → Save & Run All (Commit) → Save.** Close the tab; Kaggle emails
you on completion.

What you should see in the log, in order:

```
PREFLIGHT
  n_pooled                     PASS
  n_clusters                   PASS
  n_train                      PASS
  n_test                       PASS
  split_manifest_sha256        PASS
  images_on_disk_training      PASS
  images_on_disk_testing       PASS
  17/17 passed -> PASS
SMOKE TEST (2 epochs, isolated directory)
  smoke run COMPLETE
GRID - 9 cells
  [1/9] RUN no_gate seed 42
  Epoch 1/15 loss=... train=... val=... gap=+0.0xxx
  ...
  [1/9] OK no_gate seed 42 in 45.2 min | best epoch 8 | max gap 1.71 pp | within expected range
```

Per-epoch lines carry the train–validation gap, so overfitting is visible while
it runs rather than only afterwards.

**If preflight fails**, the session stops before training. That is the intended
behaviour: a mismatched split would produce numbers that look fine and mean
nothing. Read the failing check, fix the input, commit again.

## 5. After an interruption

Commit again. A cell is skipped only when its output **passes the completeness
validator**, not because `metrics.json` exists. An incomplete directory is
cleared and re-run. Nothing complete is ever overwritten.

## 6. Collect results

Output tab → download:

- **`fusionv2_session1_results.zip`** — the nine runs plus `run_manifest.json`
  with a SHA-256 for every file. Large: nine 365 MB checkpoints.
- `fusionv2_session1_results.zip.sha256` — verify after download.

Locally:

```bash
sha256sum -c fusionv2_session1_results.zip.sha256
unzip fusionv2_session1_results.zip -d publication_runs/session_1/

env_win/Scripts/python.exe src/publication/validate_run_outputs.py \
    --root publication_runs/session_1 --all \
    --json-out publication_results/ablations/session1/validation_report.json
```

## 7. What to check before believing the numbers

1. **All nine validated COMPLETE.** Anything less means a partial matrix.
2. **Overfitting flags.** The validator flags a max train–validation gap above
   5.0 pp, a best epoch that is the last epoch, or best validation below 0.90.
   Reference behaviour is a 1.62 pp peak with early stopping at epoch 12. A
   flagged run needs a human look, not a quiet acceptance.
3. **Preflight passed in the log.** If the split fingerprint did not match, the
   results are not comparable to the reference and must not enter the table.
4. **`no_gate` has no gate CSVs.** That is correct, not a missing artifact.

## Not in this session

`no_ema`, `cnn_branch_only`, `vit_branch_only` are Session 2. `no_tta` is
evaluation-only against the recovered checkpoints. `full_fusion_v2` is reused.
The driver refuses all of them by name.
