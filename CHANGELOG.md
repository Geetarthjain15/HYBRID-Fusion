# Brain Tumor Detection AI — Project Changelog

Running log of setup, fixes, and changes made to this project. Newest entries at the top.

---

## 2026-09-22 — Ablation harness, corrected leakage audit, Kaggle grid

**Findings that change the paper's claims**

1. **The shipped ViT checkpoint is untrained.** `vit_brain_tumor_5class.pth` scores 26.05% on the
   test set (macro ROC-AUC 0.503) and predicts `pituitary` for 99.2% of inputs. It is not a baseline.
   The hybrid model warm-starts from it, so the "ViT branch" was effectively initialised from noise.
2. **CNN-only ablation (local CPU, seed 42): 91.83%** vs fusion 98.76%. Paired McNemar p = 2.4e-17.
   Gain is concentrated in meningioma (+19.1 recall points). Holds on the leakage-free subset
   (93.32% vs 98.63%, n = 659). The fusion contribution is real.
3. **The 81% test-contamination figure was over-counted.** The audit's 64-bit dHash at Hamming ≤ 4
   collides on unrelated MRI scans of the same class. Re-measured with a 256-bit dHash at Hamming ≤ 8
   (validated against the exact-MD5 matches): **393 / 1052 = 37%**, almost entirely `notumor`
   (279/300) and `meningioma` (109/246); `glioma` and `pituitary` are essentially clean.
   New manifest: `clean_test_manifest_v2.txt` (659 images, was 196).
4. **`dataset_ood` is partially, not wholly, overlapping.** Under the 256-bit hash: `glioma` is
   ~95% resized copies of training images (explains its 100% accuracy); `meningioma` and `pituitary`
   are mostly genuinely new images. The pituitary collapse to 26% recall is a true cross-source
   generalisation failure, not a resize artefact. Label it "partial cross-source" in the paper and
   filter overlapping images before reporting.

**Files added**

| File | Purpose |
|---|---|
| `src/train_ablation.py` | One harness for `cnn` / `vit` / `fusion` arms, identical split, schedule, augmentation and freezing policy; `--unfreeze_all_cnn` capacity control; `--extra_eval` dumps predictions on additional sets; GPU-ready (`--workers`, pinned memory). |
| `src/make_clean_split.py` | Dedup-aware re-split: MD5 + 256-bit dHash clusters, one representative per cluster, stratified 85/15. 7023 → 5648 unique, 4800 train / 848 test. |
| `src/aggregate_runs.py` | Mean ± std over seeds per arm, paired McNemar vs fusion, optional clean-subset column. |
| `kaggle/ablation_grid.ipynb` + `kaggle/README.md` | Runs the full grid (4 arms × seeds × 2 splits) on a Kaggle T4, resumable, packages results. |
| `runs/cnn_seed42/` | The local CNN-only run (checkpoint, history, metrics, predictions). |
| `preds_cnn_test.csv`, `clean_test_manifest_v2.txt` | Audit trail for finding 2 and 3. |

**Files changed**

- `src/streamlit_app.py` — removed two hard-coded metric tables (94.01% / 91.2% for ViT; 97.98% /
  78.17% fallback for hybrid) that did not correspond to any measurement. ViT panel now shows the
  real 26.05% with a warning.

**Still open:** ViT-only and same-harness fusion arms, seed sweep, and the clean-split retrain — all
queued in the Kaggle notebook. Local CPU cost was measured at 5–6 h per ViT/fusion run, which is why
this moves to GPU.

---

## 2026-09-17 — Architecture documentation

- Added a detailed architecture flow chart covering both models (preprocessing → Original ViT / Hybrid CNN+ViT → softmax → confidence gate → Grad-CAM++), the training regimen, and the current checkpoint's metrics.
- Started this changelog to track project updates going forward.

## 2026-09-17 — Got the app running on Windows

**Starting point:** only documentation files (`.gitignore`, `HOW_TO_START.txt`, `requirements.txt`, `START_GUIDE.md`, an old run log) were available — no source code or model weights. Located the real project on the user's Windows PC at:

```
D:\Research Model\Brain_Tumor_Detection_ViT2
```

(the folder that first got connected, `__MACOSX`, was just macOS zip-extraction metadata, not the project).

**Issues found and fixed:**

1. **`env/` is a Mac virtual environment.** It ships `bin/` (Unix-style), which can't run on Windows.
   *Fix:* created a fresh Windows venv, `env_win/`, and installed dependencies there instead.

2. **`transformers` installed as an unpinned dependency pulled in v5.17**, which renamed ViT's internal module structure (`vit.encoder.layer.*.attention.attention.query` → `vit.layers.*.attention.q_proj`, etc). The saved checkpoint (`vit_brain_tumor_5class.pth`) was trained against the old naming, so `model.load_state_dict(...)` failed with hundreds of `Missing key(s)` / `Unexpected key(s)` errors on startup.
   *Fix:* pinned `transformers>=4.40,<5` in `requirements.txt` (resolved to `4.57.6`), which restored the matching module names. Both the **Original ViT** and **Hybrid CNN+ViT** models now load and predict correctly.

3. **Port 8501 got stuck held by the first (failed) run.** Windows blocks a remote assistant from typing into a terminal for security reasons, so the stuck process couldn't be cleanly killed via automation.
   *Fix:* app now runs on **port 8502** instead. `RUN_8502.bat` was added as a ready-made fallback launcher for whenever 8501 is unavailable.

**Files added to the project folder:**

| File | Purpose |
|---|---|
| `RUN_APP.bat` | One-click setup + run: creates `env_win`, installs `requirements.txt`, launches the app. Self-heals the `transformers` version on future runs since the pin is now in `requirements.txt`. |
| `FIX_AND_RUN.bat` | One-off script used to downgrade `transformers` in the already-created `env_win` and relaunch. Kept for reference; `RUN_APP.bat` now does this automatically on a fresh install. |
| `RUN_8502.bat` | Launches the app on port 8502 — use this if 8501 is ever occupied by a stuck earlier run. |
| `CHANGELOG.md` | This file. |

**Files changed:**

- `requirements.txt` — pinned `transformers>=4.40,<5` (was unpinned `transformers`).

**Result:** app verified working end-to-end in-browser for both models (Original ViT and Hybrid CNN+ViT both load, predict, and show correct metrics with no errors). User has since tested it themselves and confirmed it works.

---

## Known follow-ups (not yet done)

- A stray console window from the first failed run may still be open on the PC, holding port 8501 — safe to close manually at any time.
- `env/` (the original Mac venv folder) is now unused; can be deleted to save space if it's not needed for reference.
- Parameter counts cited in the architecture doc (~86M for ViT-B/16, ~5.3M for EfficientNet-B0) are the published sizes of the public backbones, not a from-source count of this specific checkpoint.
