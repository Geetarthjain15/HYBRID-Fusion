# Phase X3 Amendment Proposal — SDPA-Safe Branch Attribution

**Status:** proposal only. Not adopted, not authorized, not executed.
**Date:** 2026-09-26
**Author:** Claude Code, under `AGENTS.md` (SHA-256 `98ae8e3cb96cba27774d6484eab1ead26a987c33b2a81d777a9da688e65d4f9e`)
**Adopted plan at time of writing:** `CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md`, SHA-256 `43322d4827d6b32197ba461a6fe06701fcaaac993ac53913365fbc56f609c9cd`

This document proposes changes to the **methods** of Phase X3. It does not
modify the adopted plan file, does not implement X3, and does not touch
FusionV2, any checkpoint, X2, X2b, any dataset, any prediction file, or
Streamlit. No code was written for this proposal.

---

## 1. Why X3 as adopted cannot run

The adopted X3 specification lists four ViT methods:

| Adopted X3 ViT method | Requires attention matrices? |
|---|---|
| Attention rollout (14×14) | yes |
| Last-layer CLS attention (14×14) | yes |
| Gradient-weighted ViT attribution (14×14) | yes |
| — | |

All three require `output_attentions=True`, which under this model's `sdpa`
configuration silently returns `None` and therefore requires
`set_attn_implementation("eager")`. That switch was tested exactly this way in
**Phase X2**, which is a **recorded FAIL**:

- 4 of 2,544 image×seed pairs exceeded the prespecified `1e-6` fused-probability
  neutrality gate (worst case `1.609e-06` at batch 16, `1.639e-06` at batch 1).
- Predicted labels and accuracy were unaffected (0.00 pp), but the probability
  gate failed. 17 of 18 X2 checks passed; the neutrality gate did not.
- `x2_instrumentation/x2_neutrality.json` (`cc73e7948f55f880`) and
  `X2_NEUTRALITY_REPORT.md` (`218a948905eb5497`) remain exactly as issued and
  are not altered by this proposal.

The adopted X3 spec also names "the X2 recorder" as an allowed input. That
recorder's attention path is the one that failed. X3 cannot be executed as
written without re-running an eager-attention route that has already failed its
own neutrality gate.

**Phase X2b**, by contrast, is a separate SDPA-preserving method (Gradient ×
Input on the ViT patch-embedding tokens) and is a **recorded PASS**:

- All 2,544 image×seed pairs: max abs fused-probability difference
  **0.000000e+00**, exactly zero, on both the single pass and TTA.
- 0 predicted-label changes; accuracy identical to the recorded values on every
  seed (98.2311%, 98.2311%, 98.3491%; mean 98.27 ± 0.07%, matching the verified
  duplicate-clean baseline).
- All 848 × 3 = 2,544 maps finite, non-constant, correct 14×14 shape.
- `x2b_token_attribution/x2b_neutrality.json` (`d1449ba2b0206ac3`) records the
  full result.

This proposal replaces X3's ViT methods with the X2b method. It changes no
other part of X3.

---

## 2. What this amendment changes

| | Adopted X3 | This amendment |
|---|---|---|
| CNN method | Grad-CAM++, native grid | **unchanged** — Grad-CAM++, native grid |
| ViT method 1 | Attention rollout (14×14) | **removed** |
| ViT method 2 | Last-layer CLS attention (14×14) | **removed** |
| ViT method 3 | Gradient-weighted attention (14×14, feasibility-gated) | **removed** |
| ViT method (new) | — | **ViT patch-embedding Gradient × Input (14×14)**, the verified X2b method |
| Attention matrices | required for 3 of 4 methods | **never read** |
| Attention implementation | switches to eager for the ViT methods | **stays `sdpa` throughout** |
| Composite / fused map | none formed | **unchanged — none formed** |
| Faithfulness, stability, CNN–ViT agreement, external/cross-source results | not in X3 | **unchanged — not in X3** (deferred to X5/X6) |
| Target-class rule | `predicted` / `true` / `fixed:<k>`, default `predicted` | **unchanged** |
| Synchronization | same image, same forward | **unchanged**, extended: same fused target class, same checkpoint |

Everything not listed above is retained from the adopted X3 specification as
written.

---

## 3. A gap in reusing the existing Grad-CAM++ helper

`src/publication/fusion_v2_gradcam.py:157` (`gradcam_plus_plus`) is the
verified, existing CNN attribution helper used by the Streamlit demonstrator.
It cannot be reused unmodified for X3's "native resolution" requirement:

- It calls `pytorch_grad_cam.GradCAMPlusPlus`, whose `BaseCAM.forward` →
  `compute_cam_per_layer` (`base_cam.py:148-179`) **always** calls
  `scale_cam_image(cam, target_size)` before returning, where `target_size` is
  the 224×224 input resolution. The library's public API does not expose the
  pre-upsample grid.
- The helper's returned array is therefore already `(224, 224)` and
  min–max normalised to `[0, 1]` — post-upsample, post-normalisation. It is not
  the native 7×7 stage-8 grid X3's table calls for.

**Proposed resolution.** X3's CNN module computes Grad-CAM++ weights and the
weighted activation sum directly at native resolution, using the same
mathematical definition `pytorch_grad_cam.GradCAMPlusPlus.get_cam_weights`
implements (second/third-order gradient terms, ReLU on the target-logit
gradient), applied to the same target layer (`model.cnn_stages[-1]`, EfficientNet
stage 8, native `7×7`) and the same target (the complete fused-model logit for
the selected class, gradient flowing through the gate and both projections,
exactly as the existing helper does). It does not call `pytorch_grad_cam`.

**Required comparison check.** Because a new implementation is proposed rather
than reuse, X3 must verify equivalence: apply the existing verified helper to
get its normalised 224×224 map, downsample the new native 7×7 map by the same
`scale_cam_image` bilinear upsample used inside the library, min–max normalise
it the same way, and assert the two agree within **1e-5** (per §7, S/A checks).
If they do not agree, X3 stops and reports rather than adjusting the tolerance.

This is the one point at which X3 introduces new numerical code rather than
calling verified code unmodified, and it is flagged here for approval rather
than implemented silently.

---

## 4. Method table (amended)

| Method | Native grid | Basis | Attention required? |
|---|---|---|---|
| CNN Grad-CAM++ | 7×7 (stage 8) | second/third-order-weighted gradient of the fused target logit w.r.t. the stage-8 activation | no |
| ViT patch-embedding Gradient × Input | 14×14 | `gradient * input` at `model.vit.vit.embeddings` output, summed over 768 channels, 196 patch tokens (X2b method, unchanged) | no |

No fused or gate-weighted composite is formed. CNN and ViT maps are stored and
reported **separately**. No attention rollout, no CLS-attention map, no
gradient-weighted-attention method — all three are dropped, not deferred.

**Target-class rule** (unchanged from the adopted plan): `target_mode` ∈
{`predicted`, `true`, `fixed:<k>`}, default `predicted` = argmax of the fused
probabilities from that same synchronized forward pass. The mode, resolved
index and class name are written into every record. Both methods use the
**same** target index for a given image, seed and mode. Maps for different
modes are never combined.

---

## 5. Files this amendment would allow X3 to create

Code:

- `src/publication/explainability/cnn_attribution.py`
- `src/publication/explainability/vit_attribution.py` (thin wrapper around the
  verified `sdpa_token_attribution.TokenAttributor` — no new attribution math)
- `src/publication/explainability/run_attribution.py`
- `src/publication/explainability/test_attribution.py`

Outputs:

- `publication_runs/explainability/attribution/<seed>/<filename>.npz`
  (one file per image per seed: `cnn_map` [7,7], `vit_map` [14,14],
  `target_index`, `target_mode`, `target_class`)
- `publication_results/explainability/x3_attribution/X3_ATTRIBUTION_REPORT.md`
- `publication_results/explainability/x3_attribution/x3_attribution.json`
- `publication_results/explainability/x3_attribution/gradcam_equivalence.csv`
  (the §3 comparison, one row per checked image)
- `publication_results/explainability/x3_attribution/sha256_manifest.json`

No other path may be created and no existing file may be modified. If
implementation appears to require editing any existing file, X3 stops and
requests approval per `AGENTS.md`.

---

## 6. Output schema

`x3_attribution.json`:

```
{
  "phase": "X3",
  "created_utc": "<ISO-8601>",
  "amendment": "X3_BRANCH_ATTRIBUTION_AMENDMENT_PROPOSAL.md",
  "attn_implementation": "sdpa",
  "cnn_method": "gradcam_plus_plus_native",
  "vit_method": "gradient_x_input",
  "vit_attribution_site": "vit.vit.embeddings output, patch tokens 1:197",
  "target_mode": "predicted",
  "seeds": [42, 43, 44],
  "n_images": 848,
  "class_order": ["glioma", "meningioma", "notumor", "pituitary"],
  "checkpoint_sha256": {"42": "...", "43": "...", "44": "..."},
  "environment": {"python": "...", "torch": "...", "transformers": "..."},
  "per_seed": [{"seed": 42, "n_images": 848,
                "cnn_maps_valid": 848, "vit_maps_valid": 848,
                "gradcam_equivalence_max_abs_diff": 0.0,
                "runtime_s": 0.0}],
  "acceptance": {"S1": true, "...": true},
  "overall": "PASS" | "FAIL"
}
```

Per-image `.npz`: `cnn_map` float32 `(7,7)`, unnormalised;
`vit_map` float32 `(14,14)`, unnormalised (matches X2b's storage convention —
see AGENTS.md evidence rule 6, no invented normalisation); `target_index` int;
`target_mode` str; `target_class` str; `filename` str; `true_label` str.

---

## 7. Smoke test

8 images (2 per class), seed 42, `target_mode="predicted"`:

| Check | Requirement |
|---|---|
| S1 | `config._attn_implementation == "sdpa"` before, during, after |
| S2 | no `set_attn_implementation` call and no `output_attentions` request anywhere in `cnn_attribution.py`, `vit_attribution.py`, `run_attribution.py` (AST check, not phrase search — see X2b's `module_source_clean`) |
| S3 | CNN map: shape `(7,7)`, finite, non-constant |
| S4 | ViT map: shape `(14,14)`, finite, non-constant, patch-gradient sum `> 0` |
| S5 | CNN map (native, rescaled per §3) agrees with the existing verified `fusion_v2_gradcam.gradcam_plus_plus` output within **1e-5** |
| S6 | max abs fused-probability difference vs ordinary SDPA inference `== 0.0` for the ViT path, `<= 1e-6` for the CNN path (Grad-CAM++'s own backward pass does not touch inference probabilities either, but is checked identically) |
| S7 | 0 predicted-label changes |
| S8 | CNN and ViT target indices identical for a given image/seed/mode |
| S9 | every scored image present in the X1 manifest and in the seed-42 reference predictions |

If any smoke check fails, X3 stops and reports. The full run is not started.

---

## 8. Full extraction

848 images × 3 seeds × 2 methods = 5,088 maps. Dry run first, printing the full
matrix (images × seeds × methods × target mode) and projected cost, for review
before execution — as the adopted plan already requires.

Per image: one synchronized forward pass producing both the CNN activation/
gradient capture and the ViT patch-embedding gradient in the same inference
context (same checkpoint, same input tensor, same target index), so both maps
are guaranteed to correspond to the same fused decision.

**Comparison against existing CNN Grad-CAM++** (§3): run on all 848 seed-42
images, record `gradcam_equivalence.csv` with per-image max abs difference
between the new native-then-rescaled map and the existing helper's output.

---

## 9. Acceptance checks — X3 PASSES only if all hold

| # | Check | Requirement |
|---|---|---|
| A1 | SDPA active throughout | `sdpa` at entry, after every image, at exit, all 3 seeds |
| A2 | No eager path reachable | AST-verified absence of `set_attn_implementation` / `output_attentions` in all X3 modules |
| A3 | Predicted-label changes | 0 across 848×3, vs ordinary SDPA inference and vs recorded predictions |
| A4 | Accuracy difference | 0.00 pp on every seed |
| A5 | Neutrality | ViT path: max abs fused-probability difference `== 0.0` (per X2b precedent); CNN path (Grad-CAM++ backward touches no forward output): `<= 1e-6` |
| A6 | Map validity | all 5,088 maps finite, correct native shape, non-constant |
| A7 | Grad-CAM++ equivalence | native-then-rescaled CNN map agrees with the existing verified helper within 1e-5 on all 848 seed-42 images |
| A8 | Synchronization | CNN and ViT target index identical per image/seed/mode; both derived from the same forward pass's fused probabilities |
| A9 | No retained state | 0 retained hooks; all parameter `.grad is None`; model in `eval()` after every seed |
| A10 | Preservation | protected artifacts unchanged vs the Phase 0 baseline manifest |
| A11 | Identity | 0 filename/true-label mismatches vs the X1 manifest |
| A12 | No composite formed | no file, field, or code path in X3 combines CNN and ViT maps into one map |

---

## 10. Stop conditions

1. Any acceptance check in §9 fails.
2. The §3 native/library equivalence check exceeds 1e-5 on any image.
3. `config._attn_implementation` is anything other than `sdpa` at any point.
4. Any output path already exists.
5. A checkpoint SHA-256 does not match the three verified digests.
6. Image count is not 848, or class counts are not 243/228/116/261.
7. A filename or class-order misalignment is detected.
8. Any existing file would need modification.
9. Runtime exceeds twice the §11 estimate without explanation.
10. Any attempt is made to compute faithfulness, stability, CNN–ViT agreement,
    or an external/cross-source result inside X3 — those belong to X5/X6 and
    are out of scope here by design.

---

## 11. Runtime and storage estimate

X2b measured 0.891 s/image (batch 1, CPU) for one grad-enabled forward + one
backward. X3 additionally computes a CNN Grad-CAM++ backward per image. Both
are single backward passes through the same forward, so the combined cost is
estimated near double the ViT-only cost, not additive with a second full
forward.

| Item | Estimate |
|---|---|
| Smoke test (8 images, 1 seed) | ~15 s |
| Full grid, 848 × 3 seeds, both methods | **~70–90 min on this CPU-only machine** |
| Storage: 5,088 maps × (7×7 + 14×14) float32 | **~7.5 MB** |
| `.npz` per-image overhead (filenames, metadata) | ~2–3 MB |
| Total new storage | **< 15 MB** |

Per the user's standing instruction, the full grid is proposed to run on
**Kaggle GPU**, not locally. This changes floating-point reduction order
relative to the CPU runs reported for X2b, so the neutrality gates in §9 (A5)
must be re-measured on Kaggle and are not carried over from the CPU result.
The smoke test (§7) may run locally first as a cheap pre-flight check.

---

## 12. What this amendment does not change

- FusionV2 architecture, weights, gate, training, preprocessing, TTA policy.
- Any published metric: 99.14 ± 0.10% (original), 98.27 ± 0.07% (duplicate-clean),
  69.53 ± 1.98% (duplicate-disjoint cross-source, 373 images).
- Phase X2 (still FAILED, still recorded as issued).
- Phase X2b (still PASSED, still recorded as issued).
- The Streamlit application.
- The adopted implementation plan file.

---

## 13. Permitted claims if this amended X3 passes

- Separate CNN and ViT attribution maps were extracted for the same fused
  target class, from the same synchronized inference, with SDPA attention
  active throughout.
- The CNN map is Grad-CAM++ at native 7×7 resolution, verified equivalent to
  the existing Streamlit-demonstrator implementation within 1e-5.
- The ViT map is the same Gradient × Input sensitivity measure verified neutral
  in Phase X2b, extended to the full attribution grid.
- Both branches can be inspected independently for what the fused decision was
  sensitive to in that branch.

## 14. Prohibited claims

- That any map region is tumour, lesion, oedema, or any anatomical structure.
- That either map is segmentation, detection, or anatomical localisation.
- Any attention-based claim — no attention matrix is read anywhere in X3.
- Any claim of faithfulness, stability, or CNN–ViT agreement — not computed in
  X3; reserved for X5 (faithfulness/stability) and X6 (cross-source).
- Any claim comparing or combining the two maps into a single explanation.
- Any external or cross-source explainability claim — reserved for X6.
- Any of the standing prohibited claims in `AGENTS.md`.

---

## 15. Authorization required before execution

1. Approval of removing the three attention-based ViT methods and replacing
   them with the X2b Gradient × Input method.
2. Approval of the native-resolution Grad-CAM++ reimplementation described in
   §3, including its required 1e-5 equivalence check against the existing
   verified helper.
3. Approval to create the files in §5.
4. Confirmation that the full grid runs on Kaggle, with neutrality gates
   re-measured there rather than assumed from the CPU-measured X2b result.

No work beyond this proposal has been performed.
