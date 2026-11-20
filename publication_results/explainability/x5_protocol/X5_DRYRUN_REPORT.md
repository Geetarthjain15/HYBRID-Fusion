# Phase X5 Dry Run

**No faithfulness value was computed or retained.** The deletion and
insertion loops were executed for realistic timing and every scored
value was discarded, so the scale decision below cannot be a function
of the answer.

Protocol lock SHA-256: `a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5`

## Measured on 16 images (4 per class, seed 42), CPU

| quantity | value |
|---|---|
| map production | 0.763 s/image |
| forward passes per image per method | 47 |

| method | s/image |
|---|---|
| `composite_gate_weighted` | 11.552 |
| `cnn_gradcampp` | 11.503 |
| `baseline_random` | 11.396 |
| `composite_equal` | 11.368 |
| `baseline_centre_prior` | 11.351 |
| `vit_grad_x_input` | 11.289 |

## Projected full run (848 images x 3 seeds x 6 methods)

| quantity | value |
|---|---|
| seconds per image, all methods | 68.46 |
| seconds per image, stability | 4.58 |
| seconds per image, total | 73.80 |
| total forward passes | 717,408 |
| **CPU** | **52.2 hours** |
| **Kaggle T4 (projected)** | **3.0 hours** |
| curve storage | 17.2 MB |

GPU projection basis: measured in X3 on this code path: CPU 0.404 s/img vs Tesla T4 0.023 s/img = 17.6x.

## Scale recommendation

**Kaggle GPU, full 848 x 3**

Infeasible on this cpu-only host but comfortable on a t4; no subset needed, so no selection decision arises.

Decided from runtime and storage only. Requires approval before the
full X5 run begins.
