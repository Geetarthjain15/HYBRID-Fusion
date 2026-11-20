# Phase X2b Proposal — SDPA-Preserving ViT Token Attribution

**Status:** proposal only. Not adopted, not authorized, not executed.
**Date:** 2026-09-26
**Author:** Claude Code, under `AGENTS.md` (SHA-256 `98ae8e3cb96cba27774d6484eab1ead26a987c33b2a81d777a9da688e65d4f9e`)
**Adopted plan at time of writing:** `CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md`, SHA-256 `43322d4827d6b32197ba461a6fe06701fcaaac993ac53913365fbc56f609c9cd`

This document proposes a new phase. It does not modify the adopted implementation
plan, and it does not implement anything. No code was written for X2b. Three
read-only probes were executed in a session scratchpad to test the premises stated
below; they produced no repository artifact.

---

## 1. Relationship to Phase X2

Phase X2 (read-only dual-path instrumentation) is a **recorded FAIL** and remains
so. It is not altered, hidden, rerun, relabelled, or superseded by this proposal.
Its artifacts stay exactly as issued:

| Artifact | SHA-256 (first 16) |
|---|---|
| `x2_instrumentation/x2_neutrality.json` | `cc73e7948f55f880` |
| `x2_instrumentation/X2_NEUTRALITY_REPORT.md` | `218a948905eb5497` |

The X2 failure, restated for the record: switching the ViT attention
implementation to eager in order to obtain `output_attentions` changed the
float32 reduction order, and 4 of 2,544 image×seed pairs moved the fused
probability by more than the prespecified `1e-6` gate (worst case `1.609e-06` at
batch 16, `1.639e-06` at batch 1). Predicted labels never changed and accuracy
was identical to 0.00 pp, but the probability-neutrality gate failed. 17 of 18
X2 checks passed.

X2b is a **separate alternative extraction method**, proposed because it avoids
the eager switch entirely rather than because the X2 result is in doubt. If X2b
passes, the honest description of the outcome is that attention-matrix export was
abandoned and replaced with gradient-based token attribution — not that X2 was
fixed.

---

## 2. Verified premises

Every premise below was checked by execution before being written here, per the
`AGENTS.md` evidence rules. Probes ran on seed 42, CPU, `transformers 4.57.6`,
`torch 2.14.0+cpu`, on `glioma_00000.jpg` from the X1 reconstructed clean test set.

### 2.1 A grad-enabled forward pass is bit-identical to `no_grad`

    max | p(grad-enabled) - p(no_grad) | = 0.000000e+00
    predicted class identical

This is the technical basis for expecting X2b to pass a `1e-6` neutrality gate
where X2 could not. Enabling autograd records a graph; it does not alter the
forward kernels or their reduction order. Combined with the X2 finding that
read-only forward hooks perturb fused probabilities by exactly `0.0`, the
expected X2b neutrality residual is **exact zero**, not merely sub-tolerance.

This expectation must still be measured across all 848 images × 3 seeds. It is
not assumed.

### 2.2 SDPA remains active and is differentiable

    model.vit.config._attn_implementation == "sdpa"   (before and throughout)
    backward through the SDPA attention succeeded

`set_attn_implementation` is never called in X2b, and `output_attentions` is
never requested.

### 2.3 CORRECTION — final-layer patch tokens carry exactly zero gradient

The specification for this phase asked for relevance with respect to *the final
ViT hidden-state patch tokens*. **That is degenerate and must not be
implemented as written.** Measured:

    last_hidden_state shape                (1, 197, 768)
    |grad| CLS token      sum = 7.017653e-01   max = 4.422898e-03
    |grad| patch tokens   sum = 0.000000e+00   max = 0.000000e+00
                          nonzero = 0 / 150528

The cause is structural, not numerical. `src/fusion_v2.py:91` is:

```python
vit = self.vit_proj(self.vit.vit(pixel_values=x).last_hidden_state[:, 0])
```

Only index 0 — the CLS token — is consumed. The 196 final-layer patch tokens are
not read by the projection, the gate, the fused head, or either auxiliary head.
They are leaves with respect to every downstream output, so
`d(fused_logit)/d(last_hidden_state[:, 1:])` is identically zero. Gradient ×
Input at that site would produce an all-zero 14×14 map for every image, and
Integrated Gradients would too.

**Proposed correction:** take the attribution site to be the **patch-embedding
output** (`model.vit.vit.embeddings`), whose patch tokens do influence the final
CLS token through all 12 encoder layers. Measured at that site:

    embeddings output shape                (1, 197, 768)
    |grad| patch tokens   sum = 4.343228e+01   max = 7.053134e-03
                          nonzero = 150528 / 150528
    197 - 1 == 196 -> reshapes to 14 x 14

This is the one substantive deviation from the specification, it is required for
the method to be non-degenerate, and it is flagged here rather than absorbed
silently. **X2b must not begin until this deviation is explicitly approved or
an alternative site is directed.** An intermediate encoder layer is an equally
valid site and can be substituted on instruction; the embedding output is
proposed because it is the unique site at which the attribution covers the whole
ViT branch.

### 2.4 Attention-implementation probes by module class are invalid here

Under `transformers 4.57.6` the self-attention module class is `ViTSelfAttention`
under **both** `sdpa` and `eager`; dispatch is functional, not by subclass. A
class-name check would therefore silently pass even if eager were active. The
acceptance check in §7 reads `config._attn_implementation` and additionally
asserts that `output_attentions` is never requested.

---

## 3. Method

Unchanged from the verified pipeline: same checkpoints, same 848-image
reconstructed clean test set, same class order, same `eval_transform`, same
`tta_logits` TTA policy where TTA is used, same seeds 42/43/44.

For each selected image and seed:

1. Load the verified checkpoint; `model.eval()`; attention implementation left at
   its loaded value, `sdpa`.
2. Register one read-only forward hook on `model.vit.vit.embeddings` that calls
   `retain_grad()` on the output and stores a reference. The hook returns the
   output unmodified.
3. Run the ordinary forward pass with autograd enabled. No `no_grad`, no eager
   switch, no `output_attentions`.
4. Select the fused target class by the **same rule as X2** — predicted class
   from the recorded inference — and take the scalar fused logit for that class.
5. Obtain the gradient with `torch.autograd.grad(logit, hidden)` rather than
   `logit.backward()`, so no gradient is ever accumulated into model parameters.
6. Relevance = elementwise `gradient * input`, summed over the 768 embedding
   channels, for the 196 patch tokens (CLS at index 0 excluded).
7. Reshape the 196 values to 14×14 in row-major order, matching the ViT
   patch raster.
8. Remove the hook. Assert no retained hooks, no parameter `.grad`, and
   `config._attn_implementation == "sdpa"`.

Attribution is computed on the **un-flipped single forward pass**, because
averaging attribution across a TTA flip would mix two different spatial frames.
Neutrality is nevertheless measured on **both** the single-pass probabilities and
the TTA probabilities, since the recorded predictions use TTA.

**Primary method: Gradient × Input.** Integrated Gradients is deferred: at the
measured 0.891 s/image it costs roughly 20 hours of CPU for the full grid at 32
steps (§9), against 38 minutes for Gradient × Input. If IG is wanted it should be
authorized separately on a named subset, with the baseline, step count, and
completeness-error tolerance prespecified before any map is inspected.

CNN Grad-CAM++ stays in its own phase. **X2b produces no fused composite, no
gate-weighted combination, and no overlay of the two branches.**

---

## 4. Files X2b would be allowed to create

Code:

- `src/publication/explainability/sdpa_token_attribution.py`
- `src/publication/explainability/test_sdpa_token_attribution.py`

Outputs (all new paths; nothing overwritten):

- `publication_results/explainability/x2b_token_attribution/X2B_NEUTRALITY_REPORT.md`
- `publication_results/explainability/x2b_token_attribution/x2b_neutrality.json`
- `publication_results/explainability/x2b_token_attribution/per_seed_neutrality.csv`
- `publication_results/explainability/x2b_token_attribution/per_image_neutrality.csv`
- `publication_results/explainability/x2b_token_attribution/token_maps_seed{42,43,44}.npz`
- `publication_results/explainability/x2b_token_attribution/sha256_manifest.json`

No other path may be created, and no existing file may be modified. If the
implementation appears to require editing any existing file, X2b stops and
requests approval per `AGENTS.md`.

---

## 5. Smoke test

Before the full grid, on **16 images (4 per class), seed 42 only**:

| Check | Requirement |
|---|---|
| S1 | `config._attn_implementation == "sdpa"` before, during, and after |
| S2 | `output_attentions` never passed; no `set_attn_implementation` call in the module |
| S3 | max abs single-pass probability difference vs ordinary SDPA inference `== 0.0` |
| S4 | max abs TTA probability difference vs ordinary SDPA inference `<= 1e-6` |
| S5 | 0 predicted-label changes |
| S6 | every map is finite, shape `(14, 14)`, and not constant |
| S7 | patch-token gradient sum `> 0` (guards against the §2.3 degenerate site) |
| S8 | 0 retained hooks and all parameter `.grad is None` after the context exits |
| S9 | hooks removed even when the body raises |
| S10 | every scored image present in the reference predictions |

If any smoke check fails, X2b stops and reports. The full run is not started.

---

## 6. Full neutrality test

848 images × 3 seeds = 2,544 pairs. For each pair, record: filename, seed, true
label, recorded predicted label, instrumented predicted label, the four
single-pass probabilities, the four TTA probabilities, the max abs difference
against ordinary SDPA inference for each, target class index, patch-gradient sum,
and map min/max/mean.

Reference for comparison is ordinary SDPA inference run in the same process with
the same batch configuration, plus the recorded predictions in
`runs_v2_clean/fusion_v2_seed{42,43,44}/preds_test.csv` (read-only). Batch size
is fixed and recorded, because X2 established that which images sit near a
tolerance boundary is batch-dependent.

---

## 7. Acceptance checks — X2b PASSES only if all hold

| # | Check | Requirement |
|---|---|---|
| A1 | SDPA active throughout | `config._attn_implementation == "sdpa"` at entry, after every image, and at exit, all 3 seeds |
| A2 | No eager path reachable | no `set_attn_implementation` and no `output_attentions` anywhere in the X2b module |
| A3 | Predicted-label changes | exactly **0** across 2,544 pairs, against ordinary SDPA inference **and** against the recorded predictions |
| A4 | Accuracy difference | **0.00 pp** on every seed |
| A5 | Max fused-probability difference | **<= 1e-6** across all 848 images × 3 seeds |
| A6 | Map validity | all 2,544 maps finite, shape `(14, 14)`, non-constant, patch-gradient sum `> 0` |
| A7 | No retained state | 0 retained hooks; all parameter `.grad is None`; model still in `eval()` |
| A8 | Exact restoration | attention implementation, `vit.forward`, and module state byte-identical to pre-run, verified by re-hashing the model `state_dict` |
| A9 | Preservation | every protected artifact unchanged, verified against the Phase 0 baseline manifest |
| A10 | Identity | 0 filename mismatches, 0 true-label mismatches vs the X1 manifest |

A5 is stated as `<= 1e-6` because that is the gate this phase inherits. The
*expected* result given §2.1 is exact `0.0`; if the measured residual is nonzero
at all, that is a finding to report explicitly, not to round away. The tolerance
is not to be widened to accommodate a result.

---

## 8. Stop conditions

X2b stops and reports rather than improvising if any of these occurs:

1. The §2.3 deviation has not been approved.
2. `config._attn_implementation` is anything other than `sdpa` at any point.
3. Patch-token gradients are zero or non-finite for any image.
4. Any acceptance check in §7 fails.
5. Any output path already exists.
6. A checkpoint SHA-256 does not match the three verified digests.
7. Image count is not 848, or class counts are not 243/228/116/261.
8. A filename or class-order misalignment is detected.
9. Any existing file would need modification.
10. Runtime exceeds twice the §9 estimate without explanation.

---

## 9. Runtime and storage

Measured: 0.891 s/image, batch 1, CPU, one backward, seed 42.

| Item | Estimate |
|---|---|
| Smoke test (16 images, 1 seed) | ~15 s |
| Full grid, Gradient × Input (2,544 passes) | **~38 min** |
| Full grid, Integrated Gradients at 32 steps | ~20 h — **not proposed** |
| Token maps, 2,544 × 14×14 float32 | **~2.0 MB** |
| Per-image CSV (2,544 rows) | ~1 MB |
| Total new storage | **< 5 MB** |

No GPU is available in this environment (`torch.cuda.is_available() == False`), so
these are CPU figures.

---

## 10. Output schema

`x2b_neutrality.json`:

```
{
  "phase": "X2b",
  "created_utc": "<ISO-8601>",
  "attn_implementation": "sdpa",
  "attribution_method": "gradient_x_input",
  "attribution_site": "vit.vit.embeddings output, patch tokens 1:197",
  "target_rule": "predicted class from recorded inference",
  "batch_size": <int>,
  "seeds": [42, 43, 44],
  "n_images": 848,
  "class_order": ["glioma", "meningioma", "notumor", "pituitary"],
  "checkpoint_sha256": {"42": "...", "43": "...", "44": "..."},
  "environment": {"python": "...", "torch": "...", "transformers": "..."},
  "inputs": {"manifest": "...", "reference_predictions": ["..."]},
  "per_seed": [{"seed": 42, "max_abs_prob_diff_single": 0.0,
                "max_abs_prob_diff_tta": 0.0, "n_over_tolerance": 0,
                "pred_label_changes": 0, "accuracy_diff_pp": 0.0,
                "maps_valid": 848}],
  "acceptance": {"A1": true, ..., "A10": true},
  "overall": "PASS" | "FAIL"
}
```

`token_maps_seed{S}.npz`: arrays `filenames` (848 strings), `true_labels`,
`pred_labels`, `target_index`, `maps` (848, 14, 14) float32. Maps are stored
**unnormalised**, so any later normalisation is explicit and auditable.

---

## 11. What X2b does not change

X2b changes **only the explainability extraction method**. It does not modify:

- FusionV2 architecture (`src/fusion_v2.py` stays at `dd1a1180459b9eb1`)
- model weights or any checkpoint
- the gate, its parameterisation, or its outputs
- training, any hyperparameter, or any seed
- preprocessing or `eval_transform`
- the TTA policy
- any published metric, table, figure, or prediction CSV
- the Streamlit application
- Phase X1 or Phase X2 artifacts
- the adopted implementation plan

The published FusionV2 results are untouched and are not re-derived here:
99.14 ± 0.10% original-split accuracy, 98.27 ± 0.07% duplicate-clean accuracy,
69.53 ± 1.98% duplicate-disjoint cross-source accuracy over 373 external images.

---

## 12. Permitted claims if X2b passes

- A gradient-based token attribution map was extracted from the ViT branch while
  the verified SDPA attention implementation remained active throughout.
- The instrumentation was numerically neutral to the stated tolerance: predicted
  labels and accuracy were unchanged, and the maximum fused-probability
  difference was the measured value.
- The 14×14 map shows **model sensitivity**: how the fused logit for the selected
  class responds to perturbation of the patch-embedding representation.
- Gradient × Input is a local, first-order sensitivity measure evaluated at one
  input.
- The eager-attention route attempted in X2 failed its neutrality gate and was
  abandoned in favour of this method.

## 13. Prohibited claims

- That any map region is tumour, lesion, oedema, or any anatomical structure. **No
  red or high-relevance region may be described as tumour.**
- That the map is anatomical localisation, segmentation, or detection.
- Any attention-based claim. X2b measures no attention matrix, so it says nothing
  about attention.
- That the map is a causal, complete, or exact explanation of the prediction.
- That the map demonstrates clinically correct reasoning.
- That X2b resolves, repairs, or invalidates the X2 failure.
- That agreement between the ViT map and the CNN Grad-CAM++ map validates either.
- Any of the standing prohibited claims in `AGENTS.md`: clinical readiness,
  elimination of leakage, patient independence, universal cross-site
  generalisation, or statistical superiority over BRAIN-META.

Faithfulness of these maps is **not** established by X2b. It is measured only by
the prespecified deletion/insertion and stability protocol in Phase X5, and no
faithfulness language may be used before that phase completes.

---

## 14. Authorization required before execution

X2b must not begin until the following are explicitly granted:

1. Approval of the §2.3 attribution-site correction, or direction to a different site.
2. Approval to create the files in §4.
3. Approval of Gradient × Input as the primary method, with IG deferred.
4. Confirmation that A5 stays at `<= 1e-6` and is not to be widened.

No work beyond this proposal has been performed.
