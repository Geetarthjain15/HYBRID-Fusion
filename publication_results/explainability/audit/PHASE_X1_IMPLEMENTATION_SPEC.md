# Phase X1 — implementation specification (PROPOSAL ONLY)

**Status: not implemented. Nothing in this document has been built.** It
specifies what Phase X1 would create, so that the design can be approved or
rejected before any code exists. Phase X1 must not begin without explicit
approval.

Everything below rests on the Phase X0 findings in
`EXPLAINABILITY_ARCHITECTURE_AUDIT.md` and the recorded probe outputs. Where a
design choice is not determined by the model, that is stated rather than hidden
behind a default.

---

## 1. What Phase X1 would and would not do

**Would:** build a read-only instrumentation wrapper around an
already-constructed, digest-verified FusionV2 instance; extract per-image CNN
spatial maps, ViT attention tensors, branch logits and the gate row in one
synchronized pass; and write per-image attribution artifacts to a new directory.

**Would not:** modify `src/fusion_v2.py`, modify any training, evaluation,
diagnostic or Streamlit script, retrain or re-export any checkpoint, alter any
existing prediction file, or change any published metric.

---

## 2. Proposed modules

All new code under `src/publication/explainability/`. No existing file is edited.

### 2.1 `dual_path_hooks.py`

Instruments an existing model instance and restores it exactly on exit.

```python
class DualPathRecorder:
    def __init__(self, model: FusionV2, cnn_stages: tuple[int, ...] = (3, 5, 8),
                 enable_vit_attention: bool = True) -> None: ...
    def __enter__(self) -> "DualPathRecorder": ...
    def __exit__(self, *exc) -> None: ...          # removes hooks, restores sdpa
    def run(self, x: torch.Tensor) -> "DualPathOutputs": ...
```

```python
@dataclass(frozen=True)
class DualPathOutputs:
    fused_logits:  Tensor   # [B, 4]
    fused_probs:   Tensor   # [B, 4]
    aux_cnn_logits: Tensor  # [B, 4]
    aux_vit_logits: Tensor  # [B, 4]
    gate:          Tensor   # [B, 2]   columns (cnn, vit), rows sum to 1
    cnn_maps:      dict[int, Tensor]   # {3: [B,40,28,28], 5: [B,112,14,14], 8: [B,1280,7,7]}
    vit_attention: tuple[Tensor, ...]  # 12 x [B, 12, 197, 197]
    attn_impl_used: str
```

Contract of `__exit__`: every hook removed, `attn_implementation` restored to the
value observed at construction. Verified in X0 to restore exactly.

### 2.2 `cnn_attribution.py`

```python
def gradcam_plus_plus_map(model, x, target_index, stage: int = 8) -> Tensor   # [B, h, w]
```

Recomputes Grad-CAM++ against a captured stage activation using
`torch.autograd.grad`, so the stage is selectable. Must be validated to
reproduce the existing `fusion_v2_gradcam.gradcam_plus_plus` output at
`stage=8` within 1e-5, otherwise Phase X1 stops (see §10).

### 2.3 `vit_attribution.py`

```python
def attention_rollout(attn: tuple[Tensor, ...], add_identity: bool = True,
                      head_reduce: str = "mean") -> Tensor        # [B, 14, 14]
def cls_attention_last_layer(attn, head_reduce: str = "mean") -> Tensor  # [B, 14, 14]
```

Both take the CLS row over the 196 patch tokens and reshape to 14×14. Rollout
follows the standard formulation (add identity, renormalise, multiply layers).

**Declared assumption, not a result:** attention describes token mixing. Neither
function is evidence that the model used a region as evidence for a class. The
module docstring must say so, and the output schema carries the same statement.

### 2.4 `dual_path_composite.py`

```python
def composite(h_cnn: Tensor, h_vit: Tensor, gate: Tensor,
              grid: int = 14, norm: str = "minmax") -> Tensor     # [B, grid, grid]
```

Implements `H = α_cnn·Ĥ_cnn + α_vit·Ĥ_vit` where `α = gate[:,0], gate[:,1]` and
`Ĥ` is each map independently normalised.

**Mandatory metadata** emitted with every composite, because none of these
choices is determined by the model:

| Field | Value in the proposal | Why it must be recorded |
|---|---|---|
| `common_grid` | 14×14 | CNN stage 8 is 7×7, ViT is 14×14; one must be resampled |
| `resample` | CNN 7→14 bilinear | an arbitrary but stated choice |
| `normalisation` | per-map min–max to [0,1] | the two maps have different units |
| `alpha_source` | `gate[:, 0:2]` | the gate's trained role was feature reweighting |
| `terminology` | "gate-weighted dual-path attribution composite" | required by X0 §composite |

The function must refuse to run (raise) if asked to emit a composite without the
component maps alongside it.

### 2.5 `run_dual_path_analysis.py`

Driver. Loads a verified checkpoint, iterates the authoritative clean-test and
duplicate-disjoint images, writes per-image artifacts and a run manifest.

---

## 3. Hook locations

| Target | Location | Captured shape |
|---|---|---|
| CNN stage 3 | `model.cnn_stages[3]` forward hook | `[B, 40, 28, 28]` |
| CNN stage 5 | `model.cnn_stages[5]` forward hook | `[B, 112, 14, 14]` |
| CNN stage 8 | `model.cnn_stages[8]` forward hook | `[B, 1280, 7, 7]` |
| ViT attention | `model.vit.set_attn_implementation("eager")` + `output_attentions=True` | 12 × `[B, 12, 197, 197]` |
| Branch logits, gate | `model(x, return_aux=True)` — public API, no hook | `[B,4]`, `[B,4]`, `[B,2]` |

All verified available and prediction-neutral in Phase X0.

---

## 4. Class-target behaviour

The target class must be **explicit and recorded**, never implicit.

| Mode | Definition |
|---|---|
| `predicted` (default) | `argmax` of the fused probabilities from the **same** forward pass |
| `true` | the canonical label from `dataset_clean/split_manifest.json` |
| `fixed:<k>` | a caller-specified index |

The chosen mode, the resulting index and the class name are written into every
per-image record. Maps for different modes are never mixed in one figure.

**TTA:** the verified predictions use horizontal-flip TTA, the attribution does
not. Phase X1 must record both the single-pass probability and the TTA
probability per image, and state which one the map corresponds to. It must not
present a map beside a TTA probability without that note.

---

## 5. Normalisation rules

- Each map is normalised independently, min–max to [0,1], over its own spatial
  extent, per image.
- A constant map yields all zeros and sets `degenerate: true` in the record.
- Normalisation happens **after** any resampling, never before.
- No cross-image normalisation: maps from different images are never put on a
  shared scale.

---

## 6. Deterministic settings

| Setting | Value |
|---|---|
| device | CPU for the reference run |
| `torch.manual_seed` | 20260926, recorded |
| dropout | `model.eval()` asserted before any pass |
| batch size | 1 for per-image artifacts |
| image order | sorted by filename from the canonical manifest |
| checkpoint | digest re-verified against the recovery manifest before load |

Re-running must reproduce byte-identical maps. A determinism test asserts this
(§9).

---

## 7. Output schema

Into `publication_results/explainability/dual_path/` (new; must not exist).

```
run_manifest.json
per_image/<seed>/<split>/<filename>.json
per_image/<seed>/<split>/<filename>.npz     # cnn_map, vit_map, composite (float32)
figures/…
DUAL_PATH_ANALYSIS.md
```

Per-image JSON:

```json
{
  "filename": "...", "seed": 42, "split": "clean_test",
  "true_label": "glioma", "target_mode": "predicted",
  "target_index": 0, "target_class": "glioma",
  "fused_probs_single_pass": [ ... ], "fused_probs_tta": [ ... ],
  "map_corresponds_to": "single_pass",
  "aux_cnn_probs": [ ... ], "aux_vit_probs": [ ... ],
  "gate": {"cnn": 0.0, "vit": 0.0, "sums_to_one": true},
  "cnn_map": {"stage": 8, "native_grid": [7,7], "degenerate": false},
  "vit_map": {"method": "attention_rollout", "native_grid": [14,14], "degenerate": false},
  "composite": {"common_grid": [14,14], "resample": "bilinear",
                "normalisation": "per-map minmax",
                "alpha_source": "gate", "terminology":
                "gate-weighted dual-path attribution composite"},
  "prediction_neutrality": {"max_abs_prob_delta_vs_uninstrumented": 0.0,
                            "tolerance": 1e-06, "passed": true},
  "not_established": ["anatomical correctness", "causal branch attribution",
                      "tumour localisation"]
}
```

---

## 8. Proposed analyses

1. **Agreement between paths** — rank correlation between `Ĥ_cnn` and `Ĥ_vit` on
   the common grid, per image and per class. Descriptive.
2. **Gate versus agreement** — whether images with higher CNN gate mass show
   higher CNN/ViT map agreement. Descriptive; **no threshold, no cutoff**, and no
   claim of causation.
3. **Branch/fused decision consistency** — how often `argmax(aux_cnn)`,
   `argmax(aux_vit)` and `argmax(fused)` agree, on clean test and on the
   duplicate-disjoint subset. Reported as counts, with the auxiliary-head caveat.
4. **Shift behaviour** — whether map agreement and branch consistency differ
   between the 848 clean-test images and the 373 duplicate-disjoint images, with
   particular attention to the pituitary class, whose cross-source recall is
   13.81%.

Every analysis is descriptive unless a paired inferential procedure is
prespecified **before** results are seen. No threshold may be introduced after
the effect sizes are known.

---

## 9. Tests

`src/publication/explainability/test_dual_path.py`:

| Test | Assertion |
|---|---|
| restore exactness | after `__exit__`, `attn_implementation` and fused probabilities match the pre-instrumentation values exactly |
| prediction neutrality | instrumented vs uninstrumented fused probs within 1e-6, per seed, on real images |
| gate row validity | shape `[B,2]`, non-negative, sums to 1 within 1e-6 |
| gradcam parity | new `stage=8` map reproduces `fusion_v2_gradcam.gradcam_plus_plus` within 1e-5 |
| attention validity | 12 tensors, `[B,12,197,197]`, rows sum to 1 within 1e-4 |
| CLS reshape | 196 patch tokens reshape to 14×14 and the CLS token is excluded |
| composite refuses alone | raises if components are not emitted alongside |
| composite metadata | every required metadata field present and non-null |
| determinism | two runs at the same seed give byte-identical `.npz` |
| no threshold | no numeric cutoff appears in any verdict path (source scan) |
| protected artifacts | SHA-256 of model code, checkpoints and prediction files unchanged |
| terminology | banned strings absent from all emitted text |

---

## 10. Expected runtime and dependencies

**Dependencies:** all already installed — torch 2.14.0+cpu, torchvision
0.29.0+cpu, transformers 4.57.6, numpy 2.5.3, `pytorch_grad_cam`, PIL. **No new
dependency is required.** No network access is needed.

**Runtime (CPU, measured basis):** a single instrumented forward on a
`[2,3,224,224]` batch completed within the X0 probe in a few seconds. Grad-CAM++
adds a backward pass. An order-of-magnitude estimate for 848 images × 3 seeds is
a few hours on CPU; this is an **estimate, not a measurement**, and Phase X1
should record actual timings rather than inherit this number.

**A blocking caveat for dataset-level work:** `dataset_clean/Testing` holds no
images locally. Phase X0 established this constraint in an earlier phase, and it
means a per-image run over the real clean-test set requires the images to be
restored first. If they remain unavailable, Phase X1 must either operate on the
images that do exist or stop and report `not available`.

---

## 11. Stop conditions for Phase X1

Stop and report rather than improvise if any of these occurs:

1. Instrumented fused probabilities differ from the verified model beyond 1e-6
   on any seed — the instrumented model would not be the verified model.
2. The restore contract fails: attention implementation or probabilities do not
   return exactly to their pre-instrumentation values.
3. The new Grad-CAM++ path does not reproduce the existing implementation at
   `stage=8` within 1e-5.
4. A checkpoint digest does not match the recovery manifest.
5. Gate rows do not sum to one, or the gate returns a shape other than `[B,2]`.
6. The clean-test images required for a per-image run are not present.
7. Any output path already exists.
8. An analysis would require a threshold chosen after seeing the effect sizes.
9. Any protected artifact's SHA-256 changes.
10. A conclusion would require calling the composite a whole-model explanation,
    or treating attention or gate mass as causal evidence.

---

## 12. Terminology the specification binds Phase X1 to

**Permitted:** "gate-weighted dual-path attribution composite"; "CNN-path
Grad-CAM++ sensitivity"; "ViT attention rollout"; "branch utilisation";
"auxiliary branch head output".

**Prohibited:** "whole-model explanation"; "proven explanation"; "tumour
localisation"; "the model looks at the lesion"; "red pixels are tumour";
"anatomically correct"; "attention is the explanation"; "nonzero gate mass
proves the branch is useful"; "Grad-CAM++ validates clinical reasoning"; and any
assertion that a visually plausible map is therefore faithful.
