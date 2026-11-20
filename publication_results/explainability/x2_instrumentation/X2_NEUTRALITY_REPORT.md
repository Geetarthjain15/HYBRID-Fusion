# Phase X2 - read-only dual-path instrumentation

Generated 2026-09-26T12:05:32.652535+00:00.

**Status: FAIL.** No training, no model change, no modification to any dataset, checkpoint or existing prediction file. Raw tensors were validated in flight and not persisted.

## What was measured, and why two comparisons

Instrumenting the model requires two intrusive changes: forward hooks on the CNN stages, and switching the ViT from `sdpa` to `eager` attention, because under `sdpa` the flag `output_attentions=True` silently returns `None`. Both are confined to a context manager that restores the model on exit, including on the exception path.

Two comparisons are reported separately because they mean different things:

| | Compares | Tolerance | Purpose |
|---|---|---|---|
| **A. Neutrality** | instrumented vs uninstrumented, same machine, same checkpoint | max abs dp <= 1e-06 | measures what THIS phase does |
| **B. Identity** | instrumented vs the authoritative recorded predictions | 0 label mismatches, 0.00 pp | confirms the instrumented model still reproduces the published result |

B's probability residual is the cross-machine floating-point difference that Phase X1 already documented at around 4e-06. It is reported but is **not** held to the neutrality tolerance, which would fail X2 on a residual it did not cause.

## Per-seed results

### A. Neutrality

| Seed | n | max abs dp (single pass) | max abs dp (TTA) | images over tolerance | predicted-class changes |
|---|---:|---:|---:|---:|---:|
| 42 | 848 | 1.609e-06 | 8.941e-07 | 3 | 0 |
| 43 | 848 | 1.013e-06 | 9.537e-07 | 1 | 0 |
| 44 | 848 | 8.643e-07 | 5.364e-07 | 0 | 0 |

### B. Identity against the recorded predictions

| Seed | n | Filename mm | True-label mm | Pred-label mm | Acc instrumented | Acc recorded | Diff (pp) | max abs dp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 848 | 0 | 0 | 0 | 0.982311 | 0.982311 | 0.0000 | 4.94e-06 |
| 43 | 848 | 0 | 0 | 0 | 0.982311 | 0.982311 | 0.0000 | 3.63e-06 |
| 44 | 848 | 0 | 0 | 0 | 0.983491 | 0.983491 | 0.0000 | 4.31e-06 |

### Structural validation

| Seed | gate row-sum error | attention row-sum error | hooks after exit | attn impl before -> after | restore max abs dp |
|---|---:|---:|---:|---|---:|
| 42 | 1.19e-07 | 8.34e-07 | 0 | sdpa -> sdpa | 0.00e+00 |
| 43 | 1.19e-07 | 8.34e-07 | 0 | sdpa -> sdpa | 0.00e+00 |
| 44 | 1.19e-07 | 8.34e-07 | 0 | sdpa -> sdpa | 0.00e+00 |

## Tensor schema exposed from one synchronized pass

| Tensor | Shape observed |
|---|---|
| `fused_logits` | `[16, 4]` |
| `fused_probs` | `[16, 4]` |
| `tta_probs` | `[16, 4]` |
| `aux_cnn_logits` | `[16, 4]` |
| `aux_cnn_probs` | `[16, 4]` |
| `aux_vit_logits` | `[16, 4]` |
| `aux_vit_probs` | `[16, 4]` |
| `gate` | `[16, 2]` |
| `target_index` | `[16]` |
| `cnn_maps[3]` | `[16, 40, 28, 28]` |
| `cnn_maps[5]` | `[16, 112, 14, 14]` |
| `cnn_maps[8]` | `[16, 1280, 7, 7]` |
| `vit_attention` | `12 x [16, 12, 197, 197]` |

Raw tensors were **not** persisted. The attention stack alone is 12 x [B,12,197,197] per batch, which across 848 images and three seeds would be gigabytes for no benefit at this phase; it is validated in flight and discarded.

## Acceptance checks

| Check | Result |
|---|---|
| A neutrality within 1e-6 all images | **FAIL** |
| A zero predicted class changes | **PASS** |
| B zero filename mismatch | **PASS** |
| B zero true label mismatch | **PASS** |
| B zero pred label mismatch | **PASS** |
| B accuracy diff zero pp | **PASS** |
| gate finite nonneg sums to one | **PASS** |
| attention shape and row sums valid | **PASS** |
| cnn stage shapes valid | **PASS** |
| all tensor groups present | **PASS** |
| target matches fused argmax | **PASS** |
| no hooks retained | **PASS** |
| no gradients retained | **PASS** |
| attention implementation restored | **PASS** |
| restore reproduces exactly | **PASS** |
| raw tensors not persisted | **PASS** |
| all seeds covered | **PASS** |
| all 848 images per seed | **PASS** |

**Overall: FAIL.** Elapsed 3868.4 s.

## Scope

This phase establishes that the instrumented model is the verified model and that every tensor the later phases need can be obtained from one synchronized pass. It computes no attribution, makes no claim about what any tensor means, and establishes nothing about the external set. Attention was exposed; that does not make attention an explanation. Phase X3 has not started.
