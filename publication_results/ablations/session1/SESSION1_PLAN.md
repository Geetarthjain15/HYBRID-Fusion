# Session 1 Plan

Nine clean-split ablation cells. Nothing is trained locally; nothing has been
uploaded. This document is the reviewed contract the Kaggle driver enforces in
code.

## Scope

| Variant | Seeds | Question | Est. |
|---|---|---|---|
| `no_gate` | 42, 43, 44 | Does the learned softmax gate help, versus plain concatenation? | 45 min each |
| `no_multiscale` | 42, 43, 44 | Do the intermediate CNN taps (stages 3 and 5) contribute? | 45 min each |
| `no_auxiliary_loss` | 42, 43, 44 | Do the per-branch auxiliary losses prevent branch collapse? | 45 min each |

**9 cells, ~6.8 GPU-hours** on a T4 — inside the 12-hour session cap with margin.

### Explicitly out of scope

The driver refuses these by name rather than relying on anyone reading this table:

| Excluded | Why |
|---|---|
| `no_ema` | Session 2. Training-time ablation: the pre-EMA weights were never saved. |
| `cnn_branch_only` | Session 2. Branch-only control, needs training. |
| `vit_branch_only` | Session 2. Branch-only control, needs training. |
| `no_tta` | Evaluation-only against the recovered checkpoints. Not a training cell. |
| `full_fusion_v2` | Reused. Predictions and verified checkpoints already exist. |
| original-split runs | Clean split is primary for Phase 3. |

## Enforced order

1. **Rebuild and fingerprint the split** — before any gradient step.
2. **17-test Phase 3 suite** — refuses to train if anything fails.
3. **Smoke test** — 2 epochs, isolated directory.
4. **Grid** — nine cells, halting on the first failure.
5. **Package** — zip plus a SHA-256 manifest of every file.

## Preflight gate

The session stops unless every one of these matches. A silently different split
produces numbers that look fine and mean nothing.

| Check | Expected |
|---|---|
| pooled source images | 7023 |
| unique clusters | 5648 |
| redundant dropped | 1375 |
| clean Training | 4800 |
| clean Testing | 848 |
| dHash Hamming threshold | 8 |
| split seed | 42 |
| `split_manifest.json` SHA-256 | `a02bcb62847ce1b9146a3e769f41f88ec1afce5a46d03fb5aaffb0ef43a0e6d2` |
| class counts | Training 1377/1290/655/1478, Testing 243/228/116/261 |
| images actually on disk | must equal 4800 and 848 |

The last row matters on its own: the report states intent, the filesystem states
reality, and only the filesystem trains.

## Hyperparameters — unchanged from the reviewed reference

Read from `runs_v2_clean/fusion_v2_seed42/metrics.json`, not retyped:
batch 16, LR 1e-4, backbone LR ×0.1, weight decay 0.05, label smoothing 0.1,
cosine schedule, 15 epochs max, patience 4, validation 15% carved from the
training partition with seed 42, `aux_weight` 0.3, EMA on, hflip TTA on.

Each variant alters **exactly one** switch. Verified by test, by diffing the
resolved configuration against the reference.

## Overfitting

You asked to make sure we don't overfit. Here is the honest position.

**What I did not do, and why.** I did not add regularisation, shorten the
schedule, or change augmentation. These are controlled ablations: if the recipe
changes between the reference and a variant, any accuracy difference confounds
the component under test with a change in how hard the model was allowed to fit,
and the ablation stops answering its question. Requirement 8 fixes the
hyperparameters for the same reason.

**Guards already in the recipe, all active:**

- validation is a seeded 15% carve-out of the **training** partition only — the
  clean test set and the external set never participate in selection;
- early stopping on validation accuracy, patience 4;
- **the saved checkpoint is the best-validation epoch, not the last** — this is
  the single most important guard, and it is what stops a widening late-epoch
  gap from reaching the reported numbers;
- label smoothing 0.1, weight decay 0.05;
- RandomResizedCrop(0.8–1.0), horizontal flip, rotation 15°, colour jitter;
- EMA of weights, where the variant keeps it;
- the duplicate-clean split itself removes near-duplicate leakage.

**What I added: measurement.** Every epoch now records `train_val_gap`. Every
run's `metrics.json` carries best epoch, gap at the best epoch, maximum gap,
final gap, and whether early stopping fired. The validator flags a run for review
when the maximum gap exceeds **5.0 pp**, or when the best epoch is the last one
(meaning the epoch budget bound training rather than validation), or when best
validation accuracy is below 0.90.

**Reference behaviour on this split** (`runs_v2_clean/fusion_v2_seed42`): gap
peaks at **+1.62 pp** at epoch 12; best epoch is 8; early stopping fires at 12.
A variant departing markedly from that gets flagged — for a human to look at, not
for the script to quietly correct.

**What this does not prove.** The split controls near-duplicate *images*. It
cannot control same-subject leakage, because Phase 2 established that patient
identity is unverifiable from the available metadata. A low train–validation gap
is evidence against memorisation within this split; it is not evidence of
patient-level independence.

## Completion, resume and failure

A run counts as complete only when the validator passes: every required artifact
present, parseable, and of the expected row count.

Required per run: `checkpoint.pth`, `metrics.json`, `history.json`,
`preds_test.csv` (848 rows), `preds_dataset_ood.csv` (499 rows), `config.json`,
`environment.json`, `run.log`. Plus `gates_test.csv` and `gates_dataset_ood.csv`
for `no_multiscale` and `no_auxiliary_loss` — **not** for `no_gate`, which has no
gate by definition.

`metrics.json` alone is deliberately **not** a completion signal. A process
killed between writing metrics and writing predictions leaves a directory that
looks finished; resume logic trusting it would skip that cell permanently. On
resume, an incomplete directory is cleared and re-run.

The grid **halts on the first failure**. A partial matrix that reads as complete
is worse than an obvious stop.

## A note on `no_gate` parameter counts

`FusionV2.__init__` always constructs `self.gate`; `forward()` consults it only
when `gated=True`. So `no_gate` reports the same 91,211,918 trainable parameters
as the reference, but 262,914 of them are inert — outside the forward graph,
receiving no gradient. The dry-run output reports
`gate_used_in_forward`, `inert_gate_parameters` and
`params_excluding_inert_gate` separately so the comparison is stated honestly:
**90,949,004 effective parameters** against the reference's 91,211,918.

## Outputs

```
/kaggle/working/publication_runs/session_1/clean/<variant>/seed_<seed>/
/kaggle/working/publication_runs/session_1/_smoke_test/      (isolated)
/kaggle/working/publication_runs/session_1/run_manifest.json (SHA-256 per file)
/kaggle/working/fusionv2_session1_results.zip
/kaggle/working/fusionv2_session1_results.zip.sha256
```
