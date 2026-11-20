# Session 2 — Kaggle steps

Nothing has been uploaded or launched. Follow in order.

## 0. Prerequisites

- Kaggle account phone-verified (Settings → Phone Verification). Without it,
  Internet is off in notebooks and the pretrained backbones cannot download.
- Session 1 complete. Session 2 does not depend on its *outputs*, but running
  them out of order means the ablation table stays incomplete either way.

## 1. Upload two archives (~10 min, mostly the checkpoints)

Session 2 needs **three** Kaggle inputs. You already have the data.

### a. Code — `kaggle/upload/fusionv2_code_session2_v1.zip` (56.2 KB)

Add it as a **New Version** of your existing `session1` dataset, or create a new
`session2` dataset. Either works — the notebook discovers paths by searching
for `publication/session2_driver.py`, not by dataset name.

```
sha256sum kaggle/upload/fusionv2_code_session2_v1.zip
# expect 33bae5e3b79bb84c25cea27bde6b1a7a3dda850798b86560b3cefa9bc1fb26a0
```

### b. Checkpoints — `kaggle/upload/fusionv2_recovered_checkpoints_v1.zip` (1.02 GB)

Create a **separate dataset**, e.g. `fusionv2-recovered-checkpoints`. Keep it
apart from the code: 1 GB of weights has no business in a 56 KB code package,
and separate datasets version independently.

```
sha256sum kaggle/upload/fusionv2_recovered_checkpoints_v1.zip
# expect 321affcdcef48f6ff257ee79e4eeb81d19f0313aaea5028bbce05b29f4e397c3
```

Wait for Kaggle to finish processing both before continuing.

### c. Data — already uploaded

Reuse whichever dataset holds `dataset/` and `dataset_ood/`.

## 2. Create the notebook (2 min)

**Code → New Notebook → File → Import Notebook** →
`kaggle/publication_session2.ipynb`

Right panel:

| Setting | Value |
|---|---|
| **Add Input** | the code dataset, the checkpoint dataset, and the data dataset |
| **Accelerator** | GPU T4 ×2 |
| **Internet** | On |
| **Persistence** | Files only |

Confirm the code input is on the **Session 2 version** — if it still points at
the Session 1 package, cell 1's assert fails immediately, which is the point.

## 3. Cells 1–4 first (3 min, do not skip)

Run them individually.

**Cell 1** must print three resolved paths and:
```
code OK: 14 required files present
torch ... | cuda True | Tesla T4 x2
dataset: ['dataset', 'dataset_ood']
```

**Cell 3** verifies the checkpoints — this is the gate that matters:
```
  seed 42 PASS 9b328ba83df5d44b 365236755 bytes
  seed 43 PASS 9f97b6aa0c9ec3f1 365236755 bytes
  seed 44 PASS 4155a2b71df99a61 365236755 bytes
all 3 recovered checkpoints verified
```
A `FAIL` here means the archive is corrupt or the wrong version is attached.
Stop and re-upload; do not proceed.

**Cell 4** must list exactly twelve cells — nine `train`, three `eval_only`.

## 4. Launch (~5 h)

**Save Version → Save & Run All (Commit) → Save.** Close the tab.

## 5. The 10-minute checkpoint

**Versions → the running version → Logs.** You want, in order:

```
PREFLIGHT
  python ... | torch ... | transformers 4.57.x
  cuda True | gpu Tesla T4 (x2)
  code package: 15 files, combined digest ...
  n_pooled                          PASS
  n_clusters                        PASS
  split_manifest_sha256             PASS
  images_on_disk_training           PASS
  n_model_train_after_carve_out     PASS
  seed 42 PASS ...
  seed 43 PASS ...
  seed 44 PASS ...
  all 3 recovered checkpoints verified
  17/17 passed -> PASS
SMOKE TEST (2 epochs, isolated directory)
  smoke run COMPLETE
TRAINING GRID - 9 cells
  [1/9] TRAIN no_ema seed 42
```

**`smoke run COMPLETE` is the go signal.** After that it runs unattended and
Kaggle emails you.

## 6. What to watch

Per epoch: `Epoch 4/15 loss=0.64 train=0.9882 val=0.9792 gap=+0.0091`

Per cell: `[1/9] OK no_ema seed 42 in 45.2 min | best epoch 8 | max gap 1.71 pp | within expected range`

**`no_ema` is the variant to watch.** Removing weight averaging is exactly the
change that can widen the train–validation gap, so a flag there may well be the
finding rather than a fault. The reference peaks at +1.62 pp.

The three `no_tta` cells run last and take ~3 minutes each. They print
`(no training, no checkpoint written)`.

## 7. Collect results

Output tab → `fusionv2_session2_results.zip` and `.sha256`.

```bash
cd "D:/Research Model/Brain_Tumor_Detection_ViT2"
sha256sum -c fusionv2_session2_results.zip.sha256
unzip fusionv2_session2_results.zip -d publication_runs/session_2/

env_win/Scripts/python.exe src/publication/validate_run_outputs.py \
    --root publication_runs/session_2 --all \
    --json-out publication_results/ablations/session2/validation_report.json
```

## 8. Before believing the numbers

1. **All twelve validated COMPLETE.** Fewer means a partial matrix.
2. **The three `no_tta` cells contain no `checkpoint.pth`.** If one does,
   something trained and the cell is invalid.
3. **Preflight passed**, including all three checkpoint digests. If the split
   fingerprint or a digest differed, the results are not comparable to the
   reference and must not enter the table.
4. **Overfitting flags**, especially on `no_ema`.
5. **Branch-only cells have no gate CSVs.** Correct — they have no gate.

## If a session times out

Commit again. A cell is skipped only when its artifact set validates, so it
resumes rather than restarting, and never overwrites a finished run.

## Not in this session

Session 1 variants (`no_gate`, `no_multiscale`, `no_auxiliary_loss`),
`full_fusion_v2` retraining, original-split runs, fixed-weight gates, and
target-source stacking are all refused by name in code.
