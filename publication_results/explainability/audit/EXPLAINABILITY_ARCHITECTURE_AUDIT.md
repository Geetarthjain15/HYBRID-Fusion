# Phase X0 — Explainability architecture and evidence audit

**Scope.** Evidence and architecture audit only. No training, no model change, no
checkpoint change, no Phase X1 implementation. Every architectural statement
below is backed either by a `file:line` citation or by a recorded result from
executing the verified model on CPU. The raw probe outputs are
`_probe_raw.json` and `_probe_attention.json` in this directory.

**Contract.** `D:\Research Model\AGENTS.md`, SHA-256 `98ae8e3cb96cba27…`,
byte-identical to `CLAUDE.md` apart from the title line.

**Environment used for the probes.** torch 2.14.0+cpu, torchvision 0.29.0+cpu,
transformers 4.57.6, numpy 2.5.3, `pytorch_grad_cam` installed. CPU only,
seeded, synthetic fixed input — the probes read no dataset file.

---

## Headline findings

Four results change what Phase X1 can assume.

1. **ViT attention is not available as the model currently loads.** The backbone
   loads with `attn_implementation = "sdpa"`, under which `output_attentions=True`
   silently returns `None`. Attention becomes available only after
   `model.vit.set_attn_implementation("eager")`.
2. **That switch is prediction-neutral.** Maximum absolute change in fused
   probabilities is **8.94 × 10⁻⁸**, below the 10⁻⁶ tolerance; predicted classes
   are identical and the original implementation restores exactly. So attention
   can be obtained without invalidating the verified model — but this must be
   re-asserted per seed on the real test set before any explanation is attached
   to a verified number.
3. **The gate emits two scalars per image, not spatial or per-feature weights.**
   Shape `[B, 2]`, softmax over the last dimension, rows sum to 1. They rescale
   the 512-d *feature vectors*, never a spatial map. This constrains the proposed
   composite formula (see the dedicated section below).
4. **Branch logits are already available without retraining.** `aux_cnn` and
   `aux_vit` weights are present in the verified checkpoint and
   `model(x, return_aux=True)` returns them at inference.

---

## Answers to the Phase X0 questions

### 1. Where is the current heatmap implementation?

`src/publication/fusion_v2_gradcam.py`, function `gradcam_plus_plus()`, **lines
157–177**. Consumed by `src/publication/streamlit_fusion_v2.py` (demonstrator)
and `src/publication/audit_gradcam_faithfulness.py` (existing faithfulness
audit).

`src/streamlit_app.py` also contains heatmap code, but it operates on the legacy
v1 `HybridBrainTumorModel`, not FusionV2, and is out of scope.

### 2. Is it genuinely Grad-CAM++, ordinary Grad-CAM, or something else?

**Genuinely Grad-CAM++.** Verified by resolving the class the code imports and
reading its own weight computation:

- resolved class: `pytorch_grad_cam.grad_cam_plusplus.GradCAMPlusPlus`
- it is a distinct class from `GradCAM`: **true**
- `get_cam_weights` computes `grads_power_2 / (2*grads_power_2 + sum_activations*grads_power_3 + eps)`,
  i.e. Eq. 19 of arXiv:1710.11063, with the ReLU masking from Eq. 7
- by contrast `GradCAM.get_cam_weights` is a plain spatial mean of gradients

This was checked by `inspect.getsource` on the installed library, not inferred
from the import name.

### 3. Which exact CNN module is hooked?

`model.cnn_stages[-1]` — `src/publication/fusion_v2_gradcam.py:167`. Resolved at
runtime to a `Conv2dNormActivation`, the final (index 8) EfficientNet-B0 feature
stage, producing activations of shape **[B, 1280, 7, 7]**.

### 4. Is the fused class score differentiated, or a branch-specific score?

**The fused score.** `ClassifierOutputTarget(target_index)` is applied with
`model=model`, the complete FusionV2 instance
(`fusion_v2_gradcam.py:167–168`), so the differentiated scalar is
`model(x)[:, target_index]`.

Because `head()` consumes `fused = cat[c·g₀·2, v·g₁·2]` and `g` is itself a
differentiable function of `c` and `v`, the gradient reaching the CNN activation
flows through **both** the direct CNN path and the gate. The gate is therefore
inside the gradient path — but the resulting map is still only the CNN *spatial*
pathway's contribution, because the ViT branch has no spatial extent in this
computation.

### 5. What exact preprocessing is applied?

`fusion_v2_gradcam.py:38–44`:

```
Resize((224, 224)) → ToTensor() → Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
```

No CLAHE, no centre crop. Aspect ratio is **not** preserved — the resize is
unconditional to a square.

### 6. How is spatial resizing reversed for the displayed overlay?

**It is not reversed.** The CAM is produced on the 7×7 stage grid, upsampled
internally by `pytorch_grad_cam` to the 224×224 model input grid (asserted at
`fusion_v2_gradcam.py:170–171`), then overlaid on a 224×224 bilinear-resized copy
of the source image (`:136`). Nothing maps the map back to the original image
resolution or undoes the aspect-ratio change.

Effective upsample factor: **32×**. The heatmap is normalised min–max to [0,1]
per image, and set to zeros if constant (`:172–176`).

Consequence: every spatial statement made from this overlay is made in distorted
224×224 model space, at an effective resolution of roughly 32 source pixels.

### 7. What are the verified class order and label mapping?

`("glioma", "meningioma", "notumor", "pituitary")`, indices 0–3
(`fusion_v2_gradcam.py:37`). Alphabetical. This matches the probability column
order `p_glioma … p_pituitary` in the authoritative prediction CSVs and the
canonical mapping in `dataset_clean/split_manifest.json`
(fingerprint `3437ccd466d70f6c…`, 848 identities).

### 8. What is the complete FusionV2 forward graph?

From `src/fusion_v2.py:83–107`, with shapes recorded from an executed forward:

| Step | Source | Shape |
|---|---|---|
| input | — | `[B, 3, 224, 224]` |
| CNN stages, taps at 3 and 5 plus final, each `mean((2,3))` | `:85–89` | 40 + 112 + 1280 = 1432 |
| `cnn_proj` → `c` | `:90` | `[B, 512]` |
| `vit.vit(pixel_values=x).last_hidden_state[:, 0]` | `:91` | `[B, 768]` |
| `vit_proj` → `v` | `:91` | `[B, 512]` |
| `g = softmax(gate(cat[c,v]), dim=-1)` | `:97` | `[B, 2]` |
| `fused = cat[c·g₀·2, v·g₁·2]` | `:100` | `[B, 1024]` |
| `logits = head(fused)` | `:104` | `[B, 4]` |
| `return_aux=True` → `aux_cnn(c)`, `aux_vit(v)`, `g` | `:105–106` | `[B,4]`, `[B,4]`, `[B,2]` |

### 9. What CNN feature tensors and spatial stages are available?

Nine stages, shapes recorded on a `[2,3,224,224]` probe:

| Stage | Shape | Role |
|---|---|---|
| 0 | `[2, 32, 112, 112]` | — |
| 1 | `[2, 16, 112, 112]` | — |
| 2 | `[2, 24, 56, 56]` | — |
| **3** | `[2, 40, 28, 28]` | **multi-scale tap** |
| 4 | `[2, 80, 14, 14]` | — |
| **5** | `[2, 112, 14, 14]` | **multi-scale tap** |
| 6 | `[2, 192, 7, 7]` | — |
| 7 | `[2, 320, 7, 7]` | — |
| **8** | `[2, 1280, 7, 7]` | **final; currently hooked** |

All three contributing stages are pooled to vectors inside `branch_features`
(`:88–89`), so the **spatial** maps never leave `forward` and are reachable only
via hooks.

### 10–11. What ViT tensors are available, and does the configuration return attention?

Available by default: `last_hidden_state` of shape `[B, 197, 768]`, of which only
the CLS token at index 0 is consumed (`fusion_v2.py:91`). Hidden states are
obtainable with `output_hidden_states=True`.

**Attention is NOT returned as loaded.** The backbone loads with
`attn_implementation = "sdpa"`; passing `output_attentions=True` under sdpa
returns `None`, and transformers emits a warning to that effect.

After `model.vit.set_attn_implementation("eager")`:

| Property | Value |
|---|---|
| attentions returned | yes |
| number of layers | 12 |
| shape per layer | `[B, 12, 197, 197]` |
| rows sum to 1 | yes |
| patch tokens | 196 |
| patch grid | 14 × 14 |

### 12–14. What does the gate consume and return, and do its outputs sum to one?

**Consumes** `torch.cat([c, v], dim=1)` — the two 512-d *projected feature
vectors*, shape `[B, 1024]`.

**Module** (`fusion_v2.py:72–73`): `Linear(1024, 256) → GELU → Linear(256, 2)`,
followed by `softmax(dim=-1)` at `:97`.

**Returns** shape `[B, 2]` — **two scalar per-branch weights per image**. Not
per-feature, not per-scale, not spatial. Measured example rows on the probe:
`[0.3208, 0.6792]` and `[0.2993, 0.7007]`.

**Sums:** yes, to 1.0, along the **last dimension**, i.e. across the two branches
within one image. Verified numerically (`allclose` to ones, atol 1e-6).

**Applied to** (`:100`): each 512-d vector is scaled by `2·g` before
concatenation. The gate never multiplies a spatial map.

### 15–16. Do inference-time auxiliary branch logits exist, and can branch probabilities be recovered without retraining?

**Yes to both.** The verified checkpoint contains
`aux_cnn.weight [4,512]`, `aux_cnn.bias [4]`, `aux_vit.weight [4,512]`,
`aux_vit.bias [4]`, and loads under `strict=True` (578 state-dict entries).

`model(x, return_aux=True)` returns `(fused_logits, aux_cnn(c), aux_vit(v), g)`
at inference (`fusion_v2.py:105–106`). Softmax of the auxiliary logits gives
per-branch class distributions, verified to sum to 1.

**Caveat that must travel with any use of these.** They are *auxiliary training
heads* on the projected branch features, trained as a regulariser at weight 0.3.
They are branch-readable proxies, not the model's decision, and a disagreement
between an auxiliary head and the fused output is not by itself an error.

### 17. Can one synchronized forward pass expose everything?

**Yes.** Verified: a single `model(x, return_aux=True)` with a forward hook on
`cnn_stages[-1]` and eager attention enabled simultaneously yielded fused logits
`[2,4]`, `aux_cnn [2,4]`, `aux_vit [2,4]`, gate `[2,2]`, CNN final map
`[2,1280,7,7]` and 12 ViT attention tensors of `[2,12,197,197]`, with a
probability delta of **0.0** against the uninstrumented forward.

### 18. Would hooks or output-attention flags change fused probabilities?

**No, beyond float32 round-off.**

| Instrumentation | Max abs. Δ fused probability | Beyond 1e-6 tolerance? |
|---|---:|---|
| read-only forward hook on `cnn_stages[-1]` | 0.0 | no |
| `output_attentions=True` under sdpa (returns nothing) | 0.0 | no |
| `set_attn_implementation("eager")` | 8.94 × 10⁻⁸ | **no** |

Predicted classes are unchanged, hook removal is exact, and restoring sdpa
reproduces the reference probabilities exactly. The 8.94 × 10⁻⁸ figure is
round-off from a different attention kernel, not a behavioural change.

**This does not discharge the obligation.** The equality must be re-asserted per
seed over the real 848-image test set before any explanation is attached to a
verified result. That is a Phase X1 gate, specified in the companion document.

### 19. Which three checkpoints are publication-verified, and what are their SHA-256 values?

All three are `ACCEPTED` in
`publication_results/ablations/checkpoint_recovery/import_manifest.json` with a
matching digest:

| Seed | Path | SHA-256 |
|---|---|---|
| 42 | `publication_runs/recovered_checkpoints/clean/full_fusion_v2/seed_42/checkpoint.pth` | `9b328ba83df5d44b542e19faae68fd1493fee61e0c5b80e493c2f8a387ed592e` |
| 43 | `…/seed_43/checkpoint.pth` | `9f97b6aa0c9ec3f1124e593c8c29e11d31f4aef8bc08e20a40e7ed1f29e80f25` |
| 44 | `…/seed_44/checkpoint.pth` | `4155a2b71df99a6159e1e4a5dbe635f4e9ea7852a6704d804dd1fb5d43aca434` |

`load_verified_model` re-checks the digest against the manifest and loads with
`strict=True` before use (`fusion_v2_gradcam.py:115–131`).

### 20. Which prediction files are authoritative?

| Evaluation | Files |
|---|---|
| duplicate-clean, n = 848 | `runs_v2_clean/fusion_v2_seed{42,43,44}/preds_test.csv` |
| external, n = 499 | `runs_v2_clean/fusion_v2_seed{42,43,44}/preds_dataset_ood.csv` |
| duplicate-disjoint subset, n = 373 | the above, filtered by `results/cross_source.json → manifest[filename].kind == "disjoint"` |
| per-image gate mass | `runs_v2_clean/fusion_v2_seed{42,43,44}/gates_test.csv` |

Canonical labels come from `dataset_clean/split_manifest.json`. Per-file SHA-256
values for all of these are recorded in
`publication_results/ablations/final_analysis/analysis_provenance.json` and
re-recorded in `input_manifest.json` beside this document.

### 21. What code additions could expose explanation tensors without modifying FusionV2?

A wrapper that instruments an already-constructed instance. **No edit to
`src/fusion_v2.py` is required.** Each mechanism below was verified available and
prediction-neutral:

- `register_forward_hook` on `model.cnn_stages[i]` for spatial maps at stages 3,
  5 and 8;
- `model(x, return_aux=True)` for fused logits, both branch logits and the gate —
  already the module's public API;
- `model.vit.set_attn_implementation("eager")` plus `output_attentions=True` for
  the 12 attention tensors, restorable exactly;
- `torch.autograd.grad` against captured activations for CAM-style attribution
  without mutating the module.

Explicitly **not** required: any change to `src/fusion_v2.py`,
`src/train_ablation.py`, any existing Streamlit or diagnostic script, or any
retraining or re-export of a checkpoint.

### 22. What limitations prevent anatomical-localisation claims?

These are structural, not incidental:

1. **No ground truth.** No lesion masks, bounding boxes or expert annotations
   exist for this dataset, so no map can be scored as anatomically correct.
2. **Resolution.** The CNN CAM originates on a 7×7 grid and is upsampled 32×;
   its effective resolution is ~32 pixels, coarser than many lesions.
3. **Grid mismatch.** ViT attention is native 14×14 over 196 patches — a
   different grid from the CNN final stage, so the two paths are not spatially
   commensurate without a resampling choice.
4. **Distorted coordinates.** Preprocessing resizes to 224×224 without preserving
   aspect ratio and the overlay is never mapped back, so coordinates live in
   distorted model space.
5. **TTA mismatch.** Reported predictions average an image and its horizontal
   flip, while the heatmap comes from a single un-flipped pass, so the map does
   not correspond exactly to the reported probability.
6. **Attention is not explanation.** Attention weights describe token mixing;
   treating them as evidence for a class is an assumption, not a result.
7. **Gate mass is utilisation, not attribution.** AGENTS.md prohibits the
   contrary claim.
8. **Plausibility is not faithfulness.** A visually convincing map is not
   evidence; perturbation tests speak to sensitivity, not to anatomy.

---

## The proposed composite formula: verdict before it is used

The brief asks whether the gate really produces compatible CNN and ViT branch
weights before `H_composite = α_CNN·H_CNN + α_ViT·H_ViT` is proposed.

**Verdict: partially, and the gap matters.**

What holds: the gate emits exactly two non-negative per-image scalars that sum to
one. Formally, they can serve as α_CNN and α_ViT.

What does not hold: those scalars were trained to reweight **512-dimensional
feature vectors** immediately before the classifier head. They were never
trained to weight spatial maps, and the maps they would weight are not the
quantities they scale. Applying them to heatmaps is an **analogy**, not an
identity.

A second incompatibility is concrete: `H_CNN` is a 7×7 gradient-weighted
activation map; `H_ViT` would be a 14×14 attention-derived map. They differ in
native resolution, in units and in normalisation. Any sum requires resampling and
rescaling choices that the model does not determine.

Therefore, if the composite is used at all it must be called a **gate-weighted
dual-path attribution composite**, must be reported alongside both components
rather than alone, and must carry its resampling and normalisation rules
explicitly. It must not be called a whole-model explanation, a proven
explanation, tumour localisation, or causal branch attribution.

---

## Acceptance checks

| Check | Result |
|---|---|
| No protected artifact changed | PASS — SHA-256 re-verified after all probes |
| No existing application or model file changed | PASS |
| Every architectural statement cites executable evidence | PASS |
| Grad-CAM++ status verified rather than assumed | PASS — library source inspected |
| Checkpoint identities and hashes recorded | PASS — 3/3 ACCEPTED and digest-matched |
| Class ordering and preprocessing verified | PASS |
| Availability of auxiliary branch logits established | PASS — present in checkpoint, returned at inference |
| ViT attention availability established | PASS — unavailable under sdpa, available under eager, neutrality measured |
| Gate semantics and tensor dimensions established | PASS — `[B,2]`, softmax dim −1, rows sum to 1 |
| Proposed terminology scientifically conservative | PASS |
| Phase X1 not started | PASS |

**Nothing is recorded as `not available`.** Every question was answerable from
executable evidence.
