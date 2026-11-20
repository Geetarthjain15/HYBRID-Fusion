# Proposed amendment: explainability phases X0–X7

**Status: PROPOSAL ONLY. NOT APPROVED. NOT IMPLEMENTED.**

`CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md` (SHA-256 `521b0e0f30444d45…`) was
searched for `Phase X0` … `Phase X7`. **None is present.** The approved plan
covers Phases 0–10 of the classification study and contains no explainability
phase numbering.

This document defines X0–X7 so the sequence can be reviewed as a whole before
any of it is executed. **The implementation-plan file has not been modified.**
Incorporating these phases into the approved plan requires explicit user
approval; until that is given, X0–X7 have no standing as approved phases and
nothing beyond X0 may begin.

The proposal is deliberately conservative: two of the eight phases can end in a
documented negative result, and one is a stop-gate whose only job is to refuse
to continue if a prerequisite fails.

---

## Why a separate numbering is needed

The classification plan's Phase 4 is "Gate utility and branch-collapse
diagnostics" and Phase 6 is "Quantitative explainability evaluation". Neither
covers dual-path attribution: Phase 4 examines gate mass as branch utilisation,
and Phase 6 was scoped to the CNN-path Grad-CAM++ demonstrator that already
exists. The proposed work — extracting and comparing attribution from *both*
pathways — is a distinct study, and folding it into an existing phase number
would obscure what was approved when.

---

## Proposed phases

### X0 — Evidence and architecture audit ✅ complete

Establish, by execution rather than code reading, what the verified model
exposes: forward graph, gate semantics, auxiliary-head availability, ViT
attention availability, instrumentation neutrality, checkpoint identity,
authoritative prediction files.

*Status: complete. Outputs in `publication_results/explainability/audit/`;
independently re-verified in `../audit_addendum/`, which corrected three X0
claims.*

### X0a — Verification addendum ✅ complete

Independent re-verification of X0, correction of its procedural weaknesses,
comprehensive preservation audit against the Phase 0 baseline, and evidence-based
dataset availability determination.

*Status: complete. This directory.*

### X1 — Read-only instrumentation wrapper

Build `DualPathRecorder` per the X0 specification: hooks on CNN stages 3/5/8,
eager attention, `return_aux=True`, exact restore on exit. **No attribution
method yet** — X1 delivers only the plumbing plus its tests.

**Gate to exit X1:** instrumented fused probabilities must equal uninstrumented
within 1e-6 **per seed on the real evaluation images**, not on a synthetic probe.
If that fails, X1 stops and the sequence ends.

**Outputs:** `src/publication/explainability/dual_path_hooks.py`; test suite;
per-seed neutrality report.

### X2 — Clean-test image reconstruction and content verification

Reconstruct the 848 clean-test images from `dataset/` by filename against
`split_manifest.json` into a scratch location, then **prove content identity by
reproducing the recorded reference predictions** (target: 0 filename mismatches,
0 predicted-label mismatches, accuracy difference 0.00 pp), following the
checkpoint-recovery precedent.

**Stop condition:** if predictions do not reproduce exactly, the images are not
the authoritative ones and every later phase is void.

`dataset/` and `dataset_clean/` remain read-only; the reconstruction is written
only to a scratch path.

### X3 — Per-path attribution extraction

Compute, per image and per seed, the CNN-path Grad-CAM++ map (selectable stage)
and the ViT attention-derived map (rollout and last-layer CLS variants). Validate
the new CNN path against the existing `fusion_v2_gradcam.gradcam_plus_plus`
within 1e-5 at stage 8.

**No composite is formed in X3.** The two paths are produced and stored
separately so that any later combination can be audited against its components.

### X4 — Dual-path agreement analysis

Descriptive comparison of the two maps: rank correlation on a common grid,
per class and per evaluation regime. Report the resampling rule and per-map
normalisation explicitly.

**Constraints.** No effect-size threshold may be introduced after the values are
seen. Any inferential procedure must be prespecified before results are
inspected, or the analysis stays descriptive.

**Permitted conclusion:** the two pathways agree/disagree to a measured degree.
**Not permitted:** that agreement indicates correctness, or that either map
localises a lesion.

### X5 — Gate-weighted dual-path attribution composite (conditional)

Only if X4 establishes that a composite is interpretable. Forms
`H = α_cnn·Ĥ_cnn + α_vit·Ĥ_vit` with α taken from the gate row.

**Carried caveat, mandatory in every output:** the gate's two scalars were
trained to reweight 512-d feature vectors before the classifier head, never to
weight spatial maps; the maps differ in native resolution (7×7 versus 14×14),
units and normalisation. The composite is a **visualisation choice with a stated
rationale**, not a derived quantity of the model.

Always reported alongside both components, never alone. Terminology fixed as
**"gate-weighted dual-path attribution composite"**.

**X5 may legitimately conclude that the composite should not be used.**

### X6 — Explanation-consistency under source shift

Whether dual-path agreement and branch/fused decision consistency differ between
the 848 duplicate-clean images and the 373 duplicate-disjoint images, with
attention to the pituitary class (cross-source recall 13.81%, with 260 of 333
predictions assigned to no-tumour).

This is the phase with genuine scientific interest: whether explanation
behaviour changes where accuracy collapses. It may return a null result, which
would be reported as such.

### X7 — Reporting and limitation register

Assemble figures, tables and a limitation register. Every anatomical,
diagnostic or causal claim is excluded by construction; the register enumerates
what the study cannot establish, including the eight structural limitations
recorded in the X0 audit.

---

## Cross-cutting constraints

Binding on every phase X1–X7:

1. No modification to `src/fusion_v2.py`, `src/train_ablation.py`, any existing
   Streamlit or diagnostic script, any checkpoint, dataset or prediction file.
2. New code under `src/publication/explainability/`; outputs under
   `publication_results/explainability/`.
3. Each phase prints the AGENTS.md pre-change and post-change reports.
4. One phase at a time; no automatic continuation.
5. Instrumentation neutrality re-asserted per seed on real images before any
   explanation is attached to a verified result.
6. No effect-size threshold chosen after seeing results.
7. Prohibited throughout: "red pixels are tumour"; "anatomically correct";
   "attention is the explanation"; "nonzero gate mass proves branch usefulness";
   "a visually plausible map is faithful"; "Grad-CAM++ validates clinical
   reasoning"; any claim of clinical readiness.

---

## Phases that may end in a negative result

X4, X5 and X6 can each legitimately conclude that the analysis does not support
a positive finding. Under the AGENTS.md completion standard a documented
negative result is preferable to an unsupported positive claim, and these phases
are specified so that such an outcome is a valid completion rather than a
failure.

---

## Approval required

To adopt this amendment, the phases above would be appended to
`CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md`. That file has not been touched.
Until approval is given explicitly, X1 must not begin.
