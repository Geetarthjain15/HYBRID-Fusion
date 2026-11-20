# Session 2 Plan

Twelve clean-split cells: **nine training**, **three evaluation-only**. Nothing
is trained locally; nothing has been uploaded. This document is the reviewed
contract the Kaggle driver enforces in code.

## Scope

### Training (9 cells)

| Variant | Seeds | Question | Est. |
|---|---|---|---|
| `no_ema` | 42, 43, 44 | Does weight averaging explain the gain? | 45 min each |
| `cnn_branch_only` | 42, 43, 44 | What does the local (CNN) branch contribute alone? | 10 min each |
| `vit_branch_only` | 42, 43, 44 | What does the global (ViT) branch contribute alone? | 38 min each |

### Evaluation-only (3 cells)

| Variant | Seeds | Question | Est. |
|---|---|---|---|
| `no_tta` | 42, 43, 44 | Does horizontal-flip TTA explain the gain? | 3 min each |

**~4.8 GPU-hours total**, comfortably inside the 12-hour session cap.

### Explicitly out of scope

Refused by name in code, not left to a human reading this table:

| Excluded | Why |
|---|---|
| `no_gate`, `no_multiscale`, `no_auxiliary_loss` | Session 1 |
| `full_fusion_v2` | Reused — retraining would replace a *verified* reference with an unverified one |
| original-split runs | Clean split is the only primary split |
| fixed-weight gates | The gate is learned; hardcoding 0.15/0.85 or 0.55/0.45 would fabricate the result |
| target-source logistic stacking | Supervised adaptation, not a FusionV2 ablation |

## Why `no_tta` is evaluation-only and `no_ema` is not

Both look like they might skip training. Only one can.

**`no_tta` genuinely can.** It differs from the reference only at inference, so
the honest experiment loads the *same weights* and scores them without TTA.
Retraining would answer a different question — "a model trained again, evaluated
without TTA" — and seed-to-seed training noise (s.d. 0.07 pp) is the same order
as the effect being measured, so that substitution would swamp the signal.

**`no_ema` cannot.** The pre-EMA weights were never saved: the trainer
checkpoints `eval_model`, which *is* the EMA copy when EMA is on, so the raw
weights were discarded at save time. Validation-based selection also used the
EMA copy, so a no_ema run would have picked a different epoch. This was verified
by reading the trainer, not assumed.

## Checkpoint dependency

`no_tta` scores the three recovered FusionV2 checkpoints verified in the Phase 3
addendum — strict load (578/578 keys, 0 missing, 0 unexpected, 0 shape
mismatches), learned gate intact and input-dependent, and stored predictions
reproduced exactly on all 848 clean-test images.

| Seed | Bytes | SHA-256 (first 16) | Expected clean-test acc |
|---|---:|---|---:|
| 42 | 365,236,755 | `9b328ba83df5d44b` | 0.9823 |
| 43 | 365,236,755 | `9f97b6aa0c9ec3f1` | 0.9823 |
| 44 | 365,236,755 | `4155a2b71df99a61` | 0.9835 |

The notebook re-checks every digest **before** `no_tta` runs, and the trainer
checks again at load. A checkpoint differing by one byte is a different model;
scoring it would silently answer a question about weights nobody verified.

Checkpoints ship as a **separate 1.02 GB archive**, not in the code ZIP.

## Enforced order

1. **Environment** — CUDA, GPU model, dependency versions
2. **Code integrity** — SHA-256 of every uploaded source file
3. **Split** — rebuilt and fingerprinted before any gradient step
4. **Checkpoints** — all three digests verified before anything runs
5. **Tests** — all 17 Phase 3 tests
6. **Smoke test** — one short run, isolated directory
7. **Training grid** — nine cells, halting on first failure
8. **Evaluation** — three `no_tta` cells
9. **Packaging** — zip plus SHA-256 manifest

Checkpoint verification sits at step 4 deliberately: a hash failure surfaces
before seven hours of training, not after.

## Preflight gate

| Check | Expected |
|---|---|
| pooled source images | 7023 |
| perceptual clusters | 5648 |
| clean Training (on disk) | 4800 |
| clean Testing (on disk) | 848 |
| model train after carve-out | 4080 |
| validation after carve-out | 720 |
| `split_manifest.json` SHA-256 | `a02bcb62847ce1b9146a3e769f41f88ec1afce5a46d03fb5aaffb0ef43a0e6d2` |
| class counts | Training 1377/1290/655/1478, Testing 243/228/116/261 |

The on-disk counts matter independently of the report: the report states intent,
the filesystem states reality, and only the filesystem trains.

## Hyperparameters — unchanged

LR 1e-4, backbone multiplier 0.1, batch 16, max 15 epochs, patience 4, label
smoothing 0.1, weight decay 0.05, cosine schedule, validation a seeded 15%
carve-out of the training partition. Each variant alters **exactly one** switch,
verified by diffing the resolved configuration against the reference.

## Overfitting

The recipe is unchanged, for the same reason as Session 1: altering
regularisation would confound the component under test with a change in fitting
capacity. Overfitting is **measured**, not newly defended against.

Active guards: validation from the training partition only; early stopping on
validation, patience 4; **the checkpoint saved is the best-validation epoch, not
the last**; label smoothing; weight decay; augmentation; the duplicate-clean
split itself.

`no_ema` is the one to watch. Removing weight averaging is exactly the change
that can widen the train–validation gap, and it is the only Session 2 variant
where that is a live risk. Reference behaviour: gap peaks at **+1.62 pp**, best
epoch 8, early stopping at 12. The validator flags any run above **5.0 pp**, or
that never early-stops, or whose best validation falls below 0.90 — flagged for
a human, never auto-corrected.

Note that `no_ema` legitimately *should* differ here: EMA smooths the weights
that get evaluated, so its removal may show a wider gap. That is the finding,
not a fault.

## Artifacts

**Training cells** — `checkpoint.pth`, `metrics.json`, `history.json`,
`preds_test.csv` (848), `preds_dataset_ood.csv` (499), `config.json`,
`environment.json`, `run.log`. Plus `gates_test.csv` / `gates_dataset_ood.csv`
for `no_ema` only — the branch-only variants have no gate by construction.

**Evaluation cells** — `metrics.json`, `preds_test.csv`, `preds_dataset_ood.csv`,
`config.json`, `environment.json`, `run.log`,
`source_checkpoint_sha256.txt`, plus gate CSVs (the reference is gated).
**`checkpoint.pth` must NOT exist** — the validator fails the cell if it does,
because writing one would mean something trained.

`metrics.json` alone is not a completion signal. A process killed between
writing metrics and writing predictions leaves a directory that looks finished;
resume logic trusting it would skip that cell permanently.

## Outputs

```
/kaggle/working/publication_runs/session_2/clean/<variant>/seed_<seed>/
/kaggle/working/publication_runs/session_2/_smoke_test/        (isolated)
/kaggle/working/publication_runs/session_2/run_manifest.json   (SHA-256 per file)
/kaggle/working/fusionv2_session2_results.zip
/kaggle/working/fusionv2_session2_results.zip.sha256
```

## What Session 2 completes

With Session 1, this finishes all eight Phase 3 variants. The analysis that
follows is a paired McNemar of each variant against `full_fusion_v2`
(98.27 ± 0.07 clean test) at matching seeds — seed s.d. is 0.07 pp, so a
difference under ~0.15 pp is not a result, whatever the means suggest.
