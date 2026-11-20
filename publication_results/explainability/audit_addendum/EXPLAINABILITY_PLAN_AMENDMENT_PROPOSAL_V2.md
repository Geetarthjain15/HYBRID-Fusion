# Proposed amendment (V2): explainability phases X0 – X7

**Status: PROPOSAL ONLY. NOT APPROVED. NOT IMPLEMENTED.**

This supersedes `EXPLAINABILITY_PLAN_AMENDMENT_PROPOSAL.md` (SHA-256
`ee3b74196584fd32…`). That version is retained unchanged; nothing in it has been
edited. The two differ substantively, and the differences are listed in §0.3.

`CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md` (SHA-256 `521b0e0f30444d45…`) contains
no Phase X0–X7. **It has not been modified by this document.** Adopting this
amendment requires explicit user approval. Until that approval is given, X1 must
not begin.

---

## 0.1 Evidence this proposal rests on

Every architectural figure below was established by execution during Phase X0 and
independently re-verified during X0a. Nothing is taken from a docstring, a
filename or a variable name.

| Fact | Value | Where established |
|---|---|---|
| Gate output | `[B, 2]`, softmax on dim −1, rows sum to 1 | X0 probe, X0a recheck |
| Gate parameters | 262,914 | checkpoint recovery manifest |
| Gate operand | the two 512-d projected feature vectors, not spatial maps | `src/fusion_v2.py:97,100` |
| CNN stages available | 3 → `[B,40,28,28]`, 5 → `[B,112,14,14]`, 8 → `[B,1280,7,7]` | X0 probe |
| Currently hooked layer | `cnn_stages[-1]`, `Conv2dNormActivation` | `fusion_v2_gradcam.py:167` |
| ViT attention, as loaded | **unavailable** — `attn_implementation="sdpa"` returns `None` | X0 probe, X0a recheck |
| ViT attention, eager | 12 × `[B,12,197,197]`, 196 patches, 14×14 grid | X0 probe, X0a recheck |
| Eager-switch neutrality | max Δ fused probability 8.94e-08, argmax unchanged, restore exact | X0 probe, X0a recheck |
| Auxiliary heads | `aux_cnn`/`aux_vit` `Linear(512,4)` present in checkpoint; returned by `forward(return_aux=True)` | X0 probe, X0a recheck |
| Grad-CAM++ identity | genuine (2nd- and 3rd-order gradient terms, Eq. 19 of arXiv:1710.11063) | X0 probe, X0a recheck |
| Verified checkpoints | seeds 42/43/44, digests match the recovery manifest | X0, X0a |
| Authoritative predictions | `runs_v2_clean/fusion_v2_seed{42,43,44}/preds_test.csv` (848), `preds_dataset_ood.csv` (499; 373 disjoint) | X0 |
| Canonical labels | `dataset_clean/split_manifest.json`, fingerprint `3437ccd466d70f6c…`, 848 identities | X0 |
| Clean-test images | `dataset_clean/Testing` **does not exist**; all 848 filenames present under `dataset/` (7,023 JPEG by signature) | X0a |
| Reconstruction precedent | checkpoint recovery rebuilt the 848 from `dataset/` and reproduced all predictions, 0 mismatches, 0.00 pp | recovery manifest |

## 0.2 Two scope limits that constrain every later phase

**The neutrality result is synthetic-probe scope.** The 8.94e-08 figure was
measured on a two-image synthetic batch. It is *not* evidence about the 848
clean-test images. X2 must re-establish neutrality on the real images across all
three checkpoints before any explanation may be attached to a verified result.

**Reported predictions use hflip TTA; attribution does not.** The published
probabilities average an image and its horizontal flip. A Grad-CAM++ map is
computed from a single un-flipped pass. Map and reported probability therefore do
not correspond exactly. Every phase that pairs a map with a probability must
record which probability it means.

## 0.3 What changed from V1

| Area | V1 | V2 |
|---|---|---|
| Phase order | X1 instrumentation, X2 reconstruction | **X1 reconstruction, X2 instrumentation** — nothing can be validated against images that have not been proven to be the right images |
| Neutrality scope | asserted per seed | explicitly **all 848 images × all 3 checkpoints**, after reconstruction |
| ViT attribution | rollout and CLS only | adds a **gradient-weighted transformer attribution**, subject to a feasibility gate |
| Sanity checks | folded into other phases | **X4 is a dedicated phase**, including parameter-randomization and two null baselines |
| Faithfulness | one descriptive analysis | **X5 is quantitative**: deletion/insertion AUC, top-k masking, stability, bootstrap CIs, with the protocol locked before results are seen |
| Cross-source | descriptive comparison | **X6 applies the locked X5 protocol unchanged** to the 373 disjoint images |
| Streamlit | not addressed | **X7, gated** on X2–X5 passing, new wrapper only |
| Compute | order-of-magnitude guess | per-phase runtime and storage estimates with an explicit **infeasibility finding** for X5 at full scale (§X5) |

---

# Phase X0 — Evidence and architecture audit

**Status: COMPLETE.** Outputs in `publication_results/explainability/audit/`.

**Objective.** Establish by execution what the verified model exposes: forward
graph, gate semantics, auxiliary-head availability, ViT attention availability,
instrumentation neutrality, checkpoint identity, authoritative prediction files.

**Outcome.** All 22 audit questions answered from executable evidence. Four
findings changed what later phases may assume; see §0.1.

**Known defects, corrected in X0a.** `sha256_manifest.json` carries one stale
digest; the preservation claim was over-broad; the dataset-availability
conclusion was wrong.

---

# Phase X0a — Verification addendum

**Status: COMPLETE.** Outputs in `publication_results/explainability/audit_addendum/`.

**Objective.** Independently re-verify X0 and correct its procedural weaknesses
without overwriting it.

**Outcome.** Three X0 claims corrected: the stale digest was identified and
reported (not replaced); preservation was re-audited against a 150-file Phase 0
baseline and the wording narrowed to *no changes detected within the verifiable
scope*; the dataset blocker was withdrawn after recursive, signature-based
enumeration located all 848 filenames under `dataset/`. A later self-correction
established that the repository **is** under git (HEAD `e507c65`, 11 tracked
files, no protected root tracked), so version control provides no corroboration
for protected artifacts.

---

# Phase X1 — Clean-test reconstruction and identity verification

**Objective.** Produce the authoritative 848-image clean test set as files on
disk, and *prove by outcome* that they are the images behind the published
results. No attribution work occurs in X1. If X1 does not pass, every later phase
is void.

**Allowed inputs (all read-only).**
`dataset/` (source images); `dataset_clean/split_manifest.json` (canonical
filename → class → split); `publication_runs/recovered_checkpoints/clean/full_fusion_v2/seed_{42,43,44}/checkpoint.pth`;
`runs_v2_clean/fusion_v2_seed{42,43,44}/preds_test.csv`;
`publication_results/ablations/checkpoint_recovery/import_manifest.json`.

**Files and directories that may be created.**

```
publication_runs/explainability/reconstructed_clean_test/<class>/<filename>.jpg
publication_runs/explainability/reconstructed_clean_test/RECONSTRUCTION_MANIFEST.json
publication_results/explainability/x1_reconstruction/X1_RECONSTRUCTION_REPORT.md
publication_results/explainability/x1_reconstruction/x1_reconstruction.json
publication_results/explainability/x1_reconstruction/per_seed_reproduction.csv
src/publication/explainability/reconstruct_clean_test.py
src/publication/explainability/test_reconstruction.py
```

Nothing may be written under `dataset/`, `dataset_clean/`, `runs*`, `results*`
or the recovered-checkpoint tree.

**Commands expected.**

```
env_win/Scripts/python.exe src/publication/explainability/test_reconstruction.py --project-root .
env_win/Scripts/python.exe src/publication/explainability/reconstruct_clean_test.py --project-root . --smoke
env_win/Scripts/python.exe src/publication/explainability/reconstruct_clean_test.py --project-root . --seed 42
env_win/Scripts/python.exe src/publication/explainability/reconstruct_clean_test.py --project-root . --all-seeds
```

**Smoke test.** Reconstruct 16 images (4 per class, filename-sorted), score them
under seed 42, and compare against the corresponding rows of `preds_test.csv`.
Expect 0 filename mismatches, 0 label mismatches, 0 predicted-label mismatches.
Runtime target under 3 minutes. If the smoke test fails, the full run must not
start.

**Full-run procedure.**

1. Read the split manifest; assert exactly 848 Testing entries and 848 unique
   filenames; record the manifest SHA-256.
2. Locate each filename under `dataset/` by exact name. Assert exactly one match
   per filename — a name resolving to two paths is a stop condition, not a
   choice to be made silently.
3. Copy into `reconstructed_clean_test/<class>/`, where `<class>` is the
   manifest class, not the source directory. Record per-file SHA-256, byte size
   and source path.
4. Assert the reconstructed tree contains 848 files, zero duplicates by content
   hash within a class, and class counts glioma 243, meningioma 228, notumor
   116, pituitary 261.
5. For each of seeds 42, 43, 44: load the checkpoint (digest re-verified against
   the recovery manifest), score all 848 images under the **published evaluation
   protocol including hflip TTA**, and compare against `preds_test.csv` joined on
   filename.
6. Record per-seed: filename mismatches, true-label mismatches, predicted-label
   mismatches, maximum absolute per-class probability difference, accuracy
   difference in percentage points.

**Acceptance checks.**

- 848 unique manifest filenames resolved; 0 missing, 0 ambiguous, 0 duplicates.
- Class and label of every reconstructed file agree with the split manifest.
- Per seed: **0 predicted-label mismatches** and **0.00 pp accuracy difference**.
- Maximum absolute probability difference recorded per seed and below a
  **prespecified tolerance of 1e-4** (the checkpoint-recovery phase observed
  ≤ 4.8e-6 on this exact task; 1e-4 is a deliberately loose ceiling that still
  excludes a wrong-image scenario).
- Per-file SHA-256 recorded for all 848 reconstructed images.
- Source trees byte-identical before and after.

**Stop conditions.** Any filename absent from `dataset/`, or resolving to more
than one path. Any class/label disagreement. Any predicted-label mismatch. Any
accuracy difference above 0.00 pp. Probability difference above 1e-4. A
checkpoint digest not matching the recovery manifest. Any write attempted
outside the permitted paths.

**Estimated runtime.** Copy and hash 848 files: under 2 minutes. Scoring
848 × 3 seeds with hflip TTA on CPU: **approximately 1.5–3 hours**, extrapolated
from a two-image forward pass measured during X0. This is an estimate, not a
measurement; X1 must record actual wall-clock.

**Estimated storage.** Reconstructed images ≈ **25–40 MB** (848 JPEG at the
observed size distribution). Manifests and reports < 2 MB.

**Permitted claims.** That the 848 reconstructed files reproduce the published
per-image predictions exactly under all three verified checkpoints, and are
therefore the image inputs behind the published clean-test results.

**Prohibited claims.** That the images are patient-independent. That filename
matching alone establishes content identity — the reproduction does, and the
claim must rest on it. That reconstruction validates anything about the external
set.

---

# Phase X2 — Read-only dual-path instrumentation

**Objective.** Expose, in one synchronized inference operation, every tensor the
later phases need, and prove that instrumenting the model does not change what it
predicts — on the real images, not on a probe.

**Allowed inputs.** The X1-verified reconstructed image set; the three verified
checkpoints; `runs_v2_clean/.../preds_test.csv` for the neutrality comparison.

**Files that may be created.**

```
src/publication/explainability/dual_path_hooks.py
src/publication/explainability/test_dual_path_hooks.py
publication_results/explainability/x2_instrumentation/X2_NEUTRALITY_REPORT.md
publication_results/explainability/x2_instrumentation/x2_neutrality.json
publication_results/explainability/x2_instrumentation/per_seed_neutrality.csv
```

**Interface (proposed).**

```python
class DualPathRecorder:
    def __init__(self, model, cnn_stages=(3, 5, 8), enable_vit_attention=True): ...
    def __enter__(self) -> "DualPathRecorder": ...
    def __exit__(self, *exc) -> None: ...        # remove hooks, restore attn impl
    def run(self, x, target_mode="predicted") -> DualPathOutputs: ...
```

`DualPathOutputs` carries, from **one** forward: `fused_logits [B,4]`,
`fused_probs [B,4]`, `aux_cnn_logits [B,4]`, `aux_cnn_probs [B,4]`,
`aux_vit_logits [B,4]`, `aux_vit_probs [B,4]`, `gate [B,2]`,
`cnn_maps {3:[B,40,28,28], 5:[B,112,14,14], 8:[B,1280,7,7]}`,
`vit_attention` (12 × `[B,12,197,197]`), `target_index [B]`, `target_mode`,
`attn_impl_used`.

**Commands expected.**

```
env_win/Scripts/python.exe src/publication/explainability/test_dual_path_hooks.py --project-root .
env_win/Scripts/python.exe -m src.publication.explainability.dual_path_hooks --project-root . --smoke
env_win/Scripts/python.exe -m src.publication.explainability.dual_path_hooks --project-root . --neutrality --all-seeds
```

**Smoke test.** 8 reconstructed images, seed 42: assert every tensor above is
present with the stated shape; assert instrumented fused probabilities equal
uninstrumented to 1e-6; assert `__exit__` restores `attn_implementation` and
reproduces the uninstrumented probabilities exactly. Under 5 minutes.

**Full-run procedure.** For each seed in {42, 43, 44}, over all 848 images:
compute uninstrumented fused probabilities, then instrumented ones, and record
the maximum absolute difference, the number of images exceeding 1e-6, and whether
any predicted class changes. Verify hook count returns to zero and the attention
implementation is restored after every batch.

**Acceptance checks.**

- All 848 images × 3 checkpoints: max |Δ fused probability| ≤ **1e-6**.
- **Zero** predicted-class changes across all 2,544 image-seed pairs.
- Hook registry empty and `attn_implementation` restored after each run.
- Gate rows sum to 1 within 1e-6 on every image.
- Attention rows sum to 1 within 1e-4 on every layer sampled.
- All tensor shapes match §0.1.

**Stop conditions.** Neutrality exceeded on any image or seed — the instrumented
model would not be the verified model and the sequence ends. Restore failure.
Any tensor absent or mis-shaped. Gate rows not summing to 1.

**Estimated runtime.** 2,544 instrumented plus 2,544 uninstrumented forwards on
CPU: **approximately 2–4 hours**. Estimate, to be replaced by measurement.

**Estimated storage.** Reports and CSVs < 10 MB. Raw tensors are **not**
persisted at this phase.

**Permitted claims.** That instrumentation is prediction-neutral on the full
verified clean-test set for all three checkpoints, within the stated tolerance.

**Prohibited claims.** That neutrality on clean test implies neutrality on the
external set — X6 must re-check. That exposing attention makes attention an
explanation.

---

# Phase X3 — Branch-specific attribution

**Objective.** Produce, per image and per seed, attribution maps for each pathway
against the **same fused target class from the same synchronized inference
operation**. No composite is formed in X3.

**Allowed inputs.** X1-verified images; X2 recorder; verified checkpoints.

**Files that may be created.**

```
src/publication/explainability/cnn_attribution.py
src/publication/explainability/vit_attribution.py
src/publication/explainability/test_attribution.py
publication_runs/explainability/attribution/<seed>/<filename>.npz
publication_results/explainability/x3_attribution/X3_ATTRIBUTION_REPORT.md
publication_results/explainability/x3_attribution/x3_attribution.json
```

**Methods.**

| Method | Native grid | Basis |
|---|---|---|
| CNN Grad-CAM++ | 7×7 (stage 8; stages 3/5 optional) | gradient of the fused target logit w.r.t. the captured stage activation |
| ViT attention rollout | 14×14 | layer-wise attention product with identity added and renormalised; CLS row over the 196 patch tokens |
| ViT last-layer CLS attention | 14×14 | final-layer CLS row, head-mean |
| Gradient-weighted ViT attribution | 14×14 | attention weighted by the gradient of the fused target logit, head-wise, with negatives clipped — **subject to the feasibility gate below** |

**Feasibility gate for the gradient-weighted variant.** Gradients must flow to
the attention tensors under eager attention. This is *not* established by X0 and
must be tested first on 8 images. If gradients are unavailable or the resulting
map is degenerate, the method is **documented as infeasible and dropped**, and
X3 proceeds with the other three. Dropping it is a valid outcome, not a failure.

**Target-class rule.** `target_mode` ∈ {`predicted`, `true`, `fixed:<k>`},
default `predicted` = argmax of the fused probabilities from that same forward.
The mode, resolved index and class name are written into every record. Maps for
different modes are never combined.

**Commands expected.**

```
env_win/Scripts/python.exe src/publication/explainability/test_attribution.py --project-root .
env_win/Scripts/python.exe -m src.publication.explainability.cnn_attribution --project-root . --smoke
env_win/Scripts/python.exe -m src.publication.explainability.vit_attribution --project-root . --feasibility-gate
env_win/Scripts/python.exe -m src.publication.explainability.run_attribution --project-root . --dry-run
env_win/Scripts/python.exe -m src.publication.explainability.run_attribution --project-root . --all-seeds
```

**Smoke test.** 8 images, seed 42, all methods: assert each map is finite,
non-constant, correct native shape, and that the CNN map reproduces the existing
`fusion_v2_gradcam.gradcam_plus_plus` output at stage 8 within **1e-5**.

**Full-run procedure.** Dry run printing the full matrix (images × seeds ×
methods × target mode) and projected cost, reviewed before execution. Then
per-image: one synchronized forward, capture, compute each map, store at
**native resolution** as float32 in a compressed `.npz` with a JSON sidecar.

**Acceptance checks.**

- Every map finite, non-constant, and of the stated native shape.
- CNN parity with the existing implementation within 1e-5 at stage 8.
- All maps for an image derive from one forward with one recorded target index.
- Attention rows sum to 1 within 1e-4.
- Determinism: two runs at a fixed seed produce byte-identical `.npz`.

**Stop conditions.** CNN parity failure. A constant or non-finite map outside the
documented degenerate case. Target index disagreeing with the fused argmax under
`predicted` mode. Any write outside permitted paths.

**Estimated runtime.** Grad-CAM++ requires a backward pass per image per method.
848 × 3 seeds × 3–4 methods ≈ **7,600–10,200 forward+backward passes**. On CPU
this is **approximately 8–16 hours**. A GPU run, or a documented subset, should be
considered at the dry-run review.

**Estimated storage.** Native-resolution maps: 7×7 and 14×14 float32 ≈ 200–800 B
each; 848 × 3 × 4 ≈ **10–20 MB** compressed. **Upsampled 224×224 maps are not
persisted** — storing them would cost roughly 2 GB and they are recomputable by
interpolation.

**Permitted claims.** That each map is the stated quantity computed against the
stated target from the stated forward pass.

**Prohibited claims.** That any map localises a lesion. That attention is an
explanation. That a map is faithful — X5 tests that and X3 does not.

---

# Phase X4 — Explanation sanity checks

**Objective.** Establish that the X3 maps behave the way an attribution map must
behave before any faithfulness number is computed. A method that fails here is
excluded from X5.

**Allowed inputs.** X3 maps and code; X1 images; verified checkpoints.

**Files that may be created.**

```
src/publication/explainability/sanity_checks.py
src/publication/explainability/test_sanity_checks.py
publication_results/explainability/x4_sanity/X4_SANITY_REPORT.md
publication_results/explainability/x4_sanity/x4_sanity.json
publication_results/explainability/x4_sanity/per_method_results.csv
```

**Checks.**

| # | Check | Pass criterion |
|---|---|---|
| 1 | Determinism | two runs, fixed seed → byte-identical maps |
| 2 | Batch vs single-image | batch-of-8 maps equal per-image maps within 1e-5 |
| 3 | Target-class sensitivity | maps for different target classes differ; median pairwise correlation < 0.99 |
| 4 | Finite and non-constant | no NaN/Inf; per-map std > 0 except in documented degenerate cases |
| 5 | Native-shape verification | CNN 7×7 (or 28×28 / 14×14 per stage), ViT 14×14 |
| 6 | Orientation and resizing | a horizontally flipped input yields a horizontally flipped map within tolerance; upsampling preserves argmax location |
| 7 | Hook cleanup | hook registry empty and attention implementation restored after every call |
| 8 | Parameter randomization | progressively randomising model parameters degrades map similarity to the original; a map insensitive to randomisation is **failing**, not robust |
| 9 | Random-map baseline | uniform-random maps generated for X5 comparison |
| 10 | Centre-prior baseline | fixed centred Gaussian generated for X5 comparison |
| 11 | Parity with existing implementation | CNN Grad-CAM++ matches `fusion_v2_gradcam.gradcam_plus_plus` within 1e-5 at stage 8 |

**Commands expected.**

```
env_win/Scripts/python.exe src/publication/explainability/test_sanity_checks.py --project-root .
env_win/Scripts/python.exe -m src.publication.explainability.sanity_checks --project-root . --smoke
env_win/Scripts/python.exe -m src.publication.explainability.sanity_checks --project-root . --all-methods
```

**Smoke test.** Checks 1–7 and 11 on 16 images, seed 42, all methods. Under
15 minutes.

**Full-run procedure.** All 11 checks on a prespecified stratified sample of
**120 images (30 per class, filename-ordered)** across all three seeds. The
parameter-randomization test uses a cascading schedule from the classifier head
backwards, recorded step by step. The sample is fixed before results are seen and
is not reselected afterwards.

**Acceptance checks.** Checks 1, 2, 4, 5, 7 and 11 must pass for a method to
enter X5. Check 8 must show measurable degradation. Checks 3 and 6 are reported;
failure there is a documented limitation of the method, not necessarily an
exclusion, and the decision is recorded with its reasoning.

**Stop conditions.** Determinism failure. Hook leakage. Parity failure. A method
insensitive to parameter randomization proceeding to X5 as if valid.

**Estimated runtime.** 120 images × 3 seeds × 4 methods × ~11 checks, several
involving re-computation: **approximately 4–8 hours** on CPU.

**Estimated storage.** < 50 MB including the randomization-schedule maps.

**Permitted claims.** That a method is or is not internally consistent,
deterministic, target-sensitive and parameter-dependent under this protocol.

**Prohibited claims.** That passing sanity checks makes a map faithful, correct
or clinically meaningful. Sanity checks are necessary, not sufficient.

---

# Phase X5 — Quantitative faithfulness, stability and composite evaluation

**Objective.** Measure how much each attribution actually predicts the model's
behaviour, under a protocol fixed in writing before any result is inspected.

**Allowed inputs.** X3 maps for methods that passed X4; X1 images; verified
checkpoints; X4 null baselines.

**Files that may be created.**

```
src/publication/explainability/faithfulness.py
src/publication/explainability/composite.py
src/publication/explainability/test_faithfulness.py
publication_results/explainability/x5_faithfulness/X5_PROTOCOL_LOCK.md      (written FIRST)
publication_results/explainability/x5_faithfulness/X5_FAITHFULNESS_REPORT.md
publication_results/explainability/x5_faithfulness/x5_faithfulness.json
publication_results/explainability/x5_faithfulness/per_image_curves.csv
publication_results/explainability/x5_faithfulness/bootstrap_effects.csv
```

**Methods evaluated.** CNN Grad-CAM++; ViT rollout; gradient-weighted ViT
attribution (if X3 retained it); equal-weight composite; gate-weighted composite;
random baseline; centre-prior baseline.

**Protocol, locked before execution.** `X5_PROTOCOL_LOCK.md` is written and
hashed **before** any faithfulness number is computed, and fixes:

| Parameter | Value |
|---|---|
| Common grid | 14×14; CNN 7→14 bilinear upsample |
| Normalisation | per-map min–max to [0,1] after resampling |
| Masking value | dataset channel mean (0.5 after normalisation) |
| Deletion/insertion steps | 20 equal fractions, 0–100 % of pixels |
| Top-k values | k ∈ {1, 5, 10, 20, 50} % of the common grid |
| Transformations | horizontal flip; ±10° rotation; ±5 % scale; Gaussian noise σ=0.01 |
| Interpolation | bilinear, `align_corners=False`, recorded |
| Agreement metric | Spearman rank correlation on the common grid |
| Bootstrap | image-level, stratified by true class, 10,000 iterations, fixed recorded seed |
| Composite α | `gate[:,0]`, `gate[:,1]` from the same forward |

**Metrics.** Deletion curve and AUC; insertion curve and AUC; confidence drop
after top-k masking; prediction-change rate; stability under each
label-preserving transformation; CNN–ViT map agreement; correct-vs-incorrect
split; class-wise breakdown; image-level bootstrap CIs on every headline
quantity.

**Commands expected.**

```
env_win/Scripts/python.exe src/publication/explainability/test_faithfulness.py --project-root .
env_win/Scripts/python.exe -m src.publication.explainability.faithfulness --project-root . --write-protocol-lock
env_win/Scripts/python.exe -m src.publication.explainability.faithfulness --project-root . --smoke
env_win/Scripts/python.exe -m src.publication.explainability.faithfulness --project-root . --dry-run
env_win/Scripts/python.exe -m src.publication.explainability.faithfulness --project-root . --all
```

**Smoke test.** 8 images, seed 42, 2 methods, 5 deletion steps. Asserts curve
monotonic bookends (0 % and 100 % masking give the expected endpoints), finite
AUC, and reproducibility at a fixed seed. Under 20 minutes.

**Full-run procedure.**

1. Write `X5_PROTOCOL_LOCK.md`, record its SHA-256, and print it. **No
   faithfulness number may be computed before this file exists.**
2. Run the dry run: print the full matrix (images × seeds × methods × metrics ×
   steps) with projected forward-pass count and wall-clock, and halt for review.
   The scale decision (§compute feasibility) is taken here, before any result is
   seen, and is recorded in the lock file as an amendment with its own hash.
3. For each seed in {42, 43, 44}, for each image in the agreed set: load the X3
   maps for every retained method plus the two null baselines; resample each to
   the locked common grid; normalise per the locked rule.
4. Deletion: mask the top-ranked fraction of the grid in 20 steps, scoring the
   fused probability of the locked target class at each step; record the curve
   and its AUC. Insertion: the mirror procedure from a fully masked image.
5. Top-k masking at each locked k: record confidence drop and whether the
   predicted class changes.
6. Stability: re-run attribution under each locked transformation and record
   agreement with the untransformed map.
7. Agreement: Spearman rank correlation between the CNN and ViT maps on the
   common grid, per image.
8. Aggregate: per method, per class, and split by whether the fused prediction
   was correct. Attach an image-level stratified bootstrap CI (10,000 iterations,
   fixed seed) to every headline quantity.
9. Write results; **no locked parameter may be revisited after step 3 begins.**

**COMPUTE FEASIBILITY — this phase does not fit on CPU at full scale.**

A deletion curve at 20 steps costs 20 forward passes per image per method. At
848 images × 3 seeds × 7 methods × (20 deletion + 20 insertion) that is
**approximately 712,000 forward passes**, plus stability re-runs across four
transformations. On the CPU throughput observed in X0 this is on the order of
**weeks**, which is not a viable plan.

The dry run must therefore present, and the user must choose between:

1. a GPU run;
2. a prespecified stratified image subset (for example 200 images, 50 per class,
   fixed before results are seen), with the reduction and its power implications
   stated in the report;
3. a reduced step count, stated and justified;
4. a reduced method set, dropping methods that X4 showed to be weakest.

**No subset may be chosen after seeing partial results.** The full grid requires
explicit approval after dry-run review.

**Acceptance checks.** `X5_PROTOCOL_LOCK.md` exists and is hashed before the
first result. Every headline number carries a bootstrap CI. Null baselines are
evaluated identically to real methods. Curves are finite and reproducible. The
gate-weighted composite is reported **alongside both components**, never alone.
No parameter is changed after results are inspected.

**Stop conditions.** Any protocol parameter altered after the lock. A method
evaluated without its null-baseline comparison. A composite reported without its
components. A subset chosen post hoc. Any threshold introduced after effect sizes
are known.

**Estimated runtime.** Full grid on CPU: **infeasible (weeks)**. GPU full grid:
**approximately 10–20 hours**, estimated. 200-image CPU subset:
**approximately 30–60 hours**. All estimates, to be replaced by dry-run
measurement.

**Estimated storage.** Per-image curves as CSV/JSON: **approximately 100–300 MB**
at full scale; under 80 MB for a 200-image subset.

**Permitted claims.** That a method's deletion/insertion AUC does or does not
exceed the random and centre-prior baselines by a stated margin with a stated CI.
That agreement between pathways is of a measured magnitude. That the
**gate-weighted dual-path attribution composite** does or does not outperform its
components on these metrics.

**Prohibited claims.** That the gate-weighted composite is a causal, exact or
complete explanation of the fused decision. That high deletion AUC means the map
is anatomically correct. That any composite is a whole-model explanation. That
failing to beat a baseline proves a method worthless — it constrains what can be
claimed, no more.

**Mandatory terminology.** The gate-weighted result is called, in every table,
figure caption and sentence, the **"gate-weighted dual-path attribution
composite"**. Its α values came from a gate trained to reweight 512-d feature
vectors before the classifier head, never to weight spatial maps; the two maps
combined differ in native resolution, units and normalisation. It is a
visualisation choice with a stated rationale, not a derived quantity of the
model.

---

# Phase X6 — Locked cross-source explanation analysis

**Objective.** Apply the X5 protocol, unchanged, to the 373 duplicate-disjoint
external images, and report how explanation behaviour shifts where accuracy
collapses.

**Allowed inputs.** The locked X5 protocol (hash-verified identical); the
external images resolved through `results/cross_source.json`
(`manifest[filename].kind == "disjoint"`); `preds_dataset_ood.csv`; verified
checkpoints.

**Files that may be created.**

```
publication_runs/explainability/reconstructed_cross_source/...
publication_results/explainability/x6_cross_source/X6_CROSS_SOURCE_REPORT.md
publication_results/explainability/x6_cross_source/x6_cross_source.json
publication_results/explainability/x6_cross_source/per_image_curves.csv
publication_results/explainability/x6_cross_source/shift_effects.csv
```

**Commands expected.**

```
env_win/Scripts/python.exe -m src.publication.explainability.faithfulness --project-root . --verify-protocol-lock
env_win/Scripts/python.exe -m src.publication.explainability.cross_source --project-root . --smoke
env_win/Scripts/python.exe -m src.publication.explainability.cross_source --project-root . --all
```

**Smoke test.** 8 disjoint images, seed 42: assert the protocol-lock hash matches
X5 exactly; assert the 373-image selector reproduces exactly 373 filenames;
assert instrumentation neutrality holds on these images too (it was proven on
clean test, not here).

**Full-run procedure.** Reconstruct and identity-verify the 373 disjoint images
by the X1 procedure. Re-verify neutrality on them. Run the locked X5 pipeline.
Additionally record: branch prediction disagreement (`aux_cnn` vs `aux_vit` vs
fused); attribution agreement; gate-mass distribution versus clean test;
confidence, margin and entropy; class-wise errors; and the
**pituitary-to-no-tumour** failures specifically — the clean/cross-source gap is
recall 99.23 % → 13.81 %, with 260 of 333 pituitary predictions assigned to
no-tumour.

**Acceptance checks.** Protocol-lock hash identical to X5. Exactly 373 images.
Neutrality re-established on this set. No external label used for any selection,
threshold or tuning decision. Every comparison to clean test computed on
separate image sets and never averaged together.

**Stop conditions.** Protocol-lock hash mismatch. Image count ≠ 373. Neutrality
failure on the external set. Any use of external labels beyond scoring. Any
alteration of the X5 protocol "to make the comparison work".

**Estimated runtime.** 373/848 ≈ 44 % of the X5 cost under the same
configuration: **approximately 4–9 hours GPU**, or proportionally on a subset.

**Estimated storage.** Reconstructed external images ≈ 12–20 MB; results
≈ 50–150 MB depending on the chosen scale.

**Permitted claims.** That faithfulness, stability, branch agreement and gate
mass do or do not shift under source change, by measured amounts with CIs. That
the pituitary failure is or is not accompanied by a measurable change in
explanation behaviour.

**Prohibited claims.** That an explanation shift explains the accuracy drop —
association is not mechanism. Any anatomical or clinical statement. Any claim of
robustness across scanners, hospitals or populations. Tuning on external labels
in any form.

---

# Phase X7 — Publication reporting and optional Streamlit wrapper

**Objective.** Assemble reportable outputs and, only if earlier phases pass, a
demonstrator.

**Allowed inputs.** Outputs of X1–X6 only.

**Files that may be created.**

```
publication_results/explainability/x7_reporting/figures/*.png  and  *.pdf
publication_results/explainability/x7_reporting/tables/*.csv
publication_results/explainability/x7_reporting/X7_EXPLAINABILITY_REPORT.md
publication_results/explainability/x7_reporting/EVIDENCE_TRACEABILITY.csv
publication_results/explainability/x7_reporting/LIMITATION_REGISTER.md
publication_results/explainability/x7_reporting/supplement/...
src/publication/explainability/streamlit_dual_path.py          (NEW wrapper only)
```

**Streamlit condition.** Integration is permitted **only after X2, X3, X4 and X5
have all passed**, and only as a **new** wrapper. `src/streamlit_app.py` and
`src/publication/streamlit_fusion_v2.py` may not be modified. If a change to an
existing app later appears necessary, work stops and explicit permission is
requested.

**Commands expected.**

```
env_win/Scripts/python.exe -m src.publication.explainability.make_x7_figures --project-root .
env_win/Scripts/python.exe -m src.publication.explainability.make_x7_tables --project-root .
env_win/Scripts/python.exe -m streamlit run src/publication/explainability/streamlit_dual_path.py --server.port 8503
```

**Smoke test.** Generate one figure and one table from stored results, confirming
every plotted value traces to a machine-readable source row.

**Full-run procedure.** Tables generated programmatically from JSON/CSV — never
typed. Figures exported as PNG **and** PDF. A traceability CSV mapping each
reported number to its source file, field and verification status. A limitation
register enumerating what the study cannot establish.

**Acceptance checks.** No number typed by hand. Every figure has a PNG and a PDF.
Traceability covers every headline number. The limitation register includes the
eight structural limitations from the X0 audit. Terminology audit passes. Streamlit
gate respected.

**Stop conditions.** A reported number without a traceable source. A figure
without its data. Streamlit work begun before X2–X5 pass. Any modification of an
existing application.

**Estimated runtime.** Figures and tables: under 1 hour. Streamlit wrapper:
development time, not compute.

**Estimated storage.** Figures and supplement: **approximately 50–150 MB**.

**Permitted claims.** Restatement of what X1–X6 measured, with their stated
uncertainties and scopes.

**Prohibited claims.** Anything not established in an earlier phase. Clinical
readiness, diagnostic utility, anatomical correctness, patient-level inference.

---

# Cross-cutting rules

Binding on X1–X7 without exception:

1. **No model retraining.** No optimiser step on any FusionV2 parameter.
2. **No FusionV2 modification.** `src/fusion_v2.py` is read-only; all
   instrumentation is external.
3. **No checkpoint, dataset or prediction modification.** Existing artifacts are
   immutable. New outputs go only to `publication_results/explainability/` and
   `publication_runs/explainability/`.
4. **No test-label-based selection.** No threshold, hyperparameter, method or
   subset may be chosen using final test or external labels.
5. **One phase per explicit authorization.** No automatic continuation.
6. **No anatomical-localisation claim without genuine masks.** None exist for
   this dataset.
7. **No patient-level claim.** No patient, study or series key exists.
8. **Every number traces to machine-readable evidence**, recorded with its
   source file and field.
9. **Negative results are retained.** A method that fails X4, or fails to beat a
   null baseline in X5, is reported, not dropped.
10. **Full grids require a dry run and explicit approval.**
11. **Every phase prints the AGENTS.md pre-change and post-change reports.**
12. **Prohibited language throughout:** "red pixels are tumour"; "anatomically
    correct"; "the model looks at the lesion"; "attention is the explanation";
    "nonzero gate mass proves the branch is useful"; "a visually plausible map is
    faithful"; "Grad-CAM++ validates clinical reasoning"; "clinically ready".

# Phases that may legitimately end negative

X3 (the gradient-weighted ViT variant may prove infeasible), X4 (a method may
fail sanity checks and be excluded), X5 (a method, including either composite,
may fail to beat the random or centre-prior baseline) and X6 (no measurable
explanation shift may be found). Under the AGENTS.md completion standard a
documented negative result is preferable to an unsupported positive claim, and
each of these is specified so that such an outcome is a valid completion.

# Approval

To adopt this amendment, phases X0–X7 as defined above would be appended to
`CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md`. **That file has not been touched.**
Until approval is given explicitly, X1 must not begin.

**NEXT PHASE: not started.**
