# Phase X0 verification addendum

**Scope.** Independent re-verification of the Phase X0 outputs and correction of
its procedural weaknesses. No Phase X1. No attribution method implemented. No
change to FusionV2, checkpoints, datasets, predictions, Streamlit code, or the
existing `publication_results/explainability/audit/` directory.

**Method.** Nothing below is copied from the X0 outputs. Each value was
recomputed by `src/publication/explainability/verification/verify_x0.py`, and
where the recomputation disagrees with X0 the disagreement is reported rather
than reconciled.

---

## Summary of findings

Three X0 claims did not survive independent checking. Two are procedural, one is
substantive.

| # | X0 claim | Addendum finding |
|---|---|---|
| 1 | `sha256_manifest.json` records the deliverables | **Stale.** One digest is wrong because the file it describes was edited ten seconds after the manifest was written. |
| 2 | "all protected artifacts unchanged" | **Overreaching.** X0 checked 9 files. A 150-file pre-X0 baseline existed and was not used. |
| 3 | Phase X1 is blocked because clean-test images are unavailable | **Incorrect.** All 848 authoritative filenames are present under `dataset/`, and a documented reconstruction precedent exists. |

---

## 1–4. Digest verification of the X0 deliverables

All seven files in `publication_results/explainability/audit/` were re-hashed
and compared against the digests recorded in X0's own
`sha256_manifest.json` (generated `2026-09-26T10:37:36Z`).

| File | Modified (UTC) | Digest match |
|---|---|---|
| `_probe_raw.json` | 10:31:46 | ✅ match |
| `_probe_attention.json` | 10:32:48 | ✅ match |
| `input_manifest.json` | 10:35:00 | ✅ match |
| `EXPLAINABILITY_ARCHITECTURE_AUDIT.md` | 10:36:16 | ✅ match |
| `PHASE_X1_IMPLEMENTATION_SPEC.md` | 10:37:20 | ✅ match |
| **`explainability_architecture_audit.json`** | **10:37:50** | ❌ **MISMATCH** |
| `sha256_manifest.json` | 10:37:40 | not listed in itself |

### The mismatch, and why it happened

**Recorded in X0 manifest:** `6ccb79a7c7218b7e7729…`
**Actual digest now:** `8c72d7cfe0d7d25d73da…`

The cause is a write-ordering error in Phase X0, and the timestamps establish it
directly:

```
10:37:40   sha256_manifest.json written  (hashes the 6 files then present)
10:37:50   explainability_architecture_audit.json EDITED
```

Phase X0 wrote the manifest, then patched the audit JSON to fill two `null`
fields in `acceptance_checks` (`no_protected_artifact_changed`,
`no_existing_file_modified`), add a `protected_artifact_recheck` block and set
`acceptance_status`. The manifest was never regenerated. It therefore describes
a version of the file that no longer exists on disk.

**Answer to the mandated question:** **Yes** — `explainability_architecture_audit.json`
was modified after `sha256_manifest.json` was generated. This is confirmed by two
independent signals: the modification time is 10 seconds later, and the digest
does not match.

**Remedy applied.** The stale digest is **reported, not corrected**. The X0
manifest is left exactly as issued, in accordance with the instruction not to
modify the existing audit directory and not to replace stale hashes silently.
The current digest of every X0 output is recorded in `final_manifest.json` under
`x0_outputs_as_found_now`.

`sha256_manifest.json` is absent from its own listing. That is correct rather
than an error — a file cannot contain its own digest — but it does mean the X0
manifest self-describes only 6 of the 7 files now in that directory.

### Consequence for the X0 acceptance status

X0 reported `ACCEPTANCE STATUS: PASS`. That verdict was computed from the
acceptance-check block *after* it was patched, and the patch is precisely what
invalidated the manifest. The individual findings in the X0 audit are unaffected
— all of them are re-verified as correct in section 5 below — but the integrity
claim attached to them was not self-consistent at the moment it was made.

---

## Preservation audit

### What X0 claimed, and why it did not follow

X0 stated that all protected artifacts were unchanged, on the basis of nine
files: six source files and three checkpoints. The protected scope defined by
`AGENTS.md` spans `runs*`, `results*`, `dataset*`, checkpoints and predictions —
tens of thousands of files. Nine matching digests do not license a statement
about all of them.

### A comprehensive pre-X0 baseline did exist

`publication_results/audit/baseline_manifest.json`, the Phase 0 immutable
baseline audit, generated `2026-09-23T20:11:24Z` — three days before Phase X0 —
carries per-file SHA-256 for:

- 61 prediction CSVs
- 89 metric files

**150 protected files with genuine pre-X0 digests.** Phase X0 did not use it.

### Result of the comparison

| Quantity | Value |
|---|---:|
| Files covered by the Phase 0 baseline | 150 |
| Re-hashed and compared | 150 |
| Digest changes detected | **0** |
| Files missing since baseline | **0** |
| Unreadable | 0 |

### What the baseline does *not* cover

Source code files, model checkpoints and dataset images are outside its scope.
For those categories the only available evidence is modification time, which is
weaker: mtime survives a copy and can be set arbitrarily, so a clean mtime is
consistent with an unchanged file but does not prove one.

### Version control — corrected finding

**An earlier revision of this addendum stated that there is no `.git` directory.
That was wrong, and the error is recorded here rather than quietly replaced.**

A shell `git rev-parse` had failed with `fatal: detected dubious ownership in
repository … is on a file system that does not record ownership`. That is git
refusing to operate, not git reporting an absence. The refusal was misread as
"not a git repository".

Re-inspected directly, and then with the ownership guard relaxed:

| Property | Value |
|---|---|
| `.git` directory present | **yes** |
| Is a git work tree | **yes** |
| HEAD | `e507c65` — 2026-09-14 18:25:23 +0530 — "Initial project setup" |
| Tracked files, whole repository | **11** |
| Protected roots containing a tracked file | **0** |
| `src/fusion_v2.py` | untracked |
| `src/train_ablation.py` | untracked |
| `src/streamlit_app.py` | tracked |

**The corrected conclusion is the same as the original one, but for a sounder
reason.** Version control provides **no corroboration for the protected
artifacts**: only 11 files are tracked in the entire repository, and no tracked
file lives under `runs*`, `results*`, `dataset*` or `publication_runs`. Git can
therefore neither confirm nor refute any change to them. `runs/`, `results/`,
`dataset_clean/`, `publication_runs/` and the rest all report as untracked
(`??`).

One observation, recorded rather than attributed: `src/streamlit_app.py` is
tracked and shows as modified (` M`) relative to the single commit. That
modification predates this work — its SHA-256 has been constant at
`bfae3567ba77d62b…` across every phase in this session, and its modification
time (2026-09-22) precedes Phase X0 (2026-09-26). It is not attributed to any
phase audited here.

### Wording

> **No changes detected within the verifiable scope.**

The verified scope is: 150 prediction and metric files digest-compared against
the Phase 0 baseline; plus modification-time inspection of 13 protected
directory trees against the Phase X0 start reference. The phrase "all protected
artifacts unchanged" is **not** used, because it is not supported.

---

## Dataset availability

X0 recorded that `dataset_clean/Testing` holds no images, repeating a finding
from an earlier phase. Prose is a claim, so the directory was enumerated
recursively and every file classified by magic-byte signature rather than by
extension.

| Path inspected | Exists | Files (recursive) | Image signatures |
|---|---|---:|---|
| `dataset_clean/Testing` | **No — the directory does not exist** | — | — |
| `dataset_clean` | yes | 2 | none (both are JSON manifests) |
| `dataset/Testing` | yes | 1,052 | 1,052 JPEG |
| `dataset` | yes | **7,023** | **7,023 JPEG** |

**First correction.** X0 said the directory "holds no images". It does not exist
at all. The distinction matters: an empty directory suggests images were removed
from a live path; an absent directory is consistent with `dataset_clean/` never
having held images, which is what the checkpoint-recovery record states.

### Matching against the authoritative split manifest

`dataset_clean/split_manifest.json` (SHA-256 `a02bcb62847ce1b9…`) lists **848**
Testing entries with 848 unique filenames.

| Location | Filenames matched | Fraction of 848 |
|---|---:|---:|
| `dataset/Testing` | 103 | 12.2% |
| **`dataset`** | **848** | **100.0%** |

The 103/848 figure for `dataset/Testing` is expected rather than anomalous: the
duplicate-clean repartition reassigned images across the original split, so
clean-test images are distributed across both `dataset/Training` and
`dataset/Testing`. Searching `dataset/` as a whole finds all 848.

### Is filename matching sufficient evidence?

**No, on its own.** `split_manifest.json` carries `file`, `class`, `new_split`,
`original_split`, `cluster_size` and `cluster_members` — **no content hash**. A
filename match proves a name exists, not that the bytes are the authoritative
image.

**But a content-level verification already exists.** The checkpoint-recovery
phase reconstructed all 848 clean-test images from `dataset/` by exactly this
route and recorded:

```
note                 : "dataset_clean/ was read only; images sourced from dataset/"
n_testing_in_manifest: 848
n_reconstructed      : 848
full_test            : n_compared 848, n_filename_mismatch 0,
                       n_pred_label_mismatch 0, accuracy_diff_pp 0.0
```

Reproducing all 848 recorded reference predictions with zero mismatches is an
outcome-level content check: had the wrong images been sourced, the predictions
could not have matched. The reconstruction procedure is documented and
repeatable.

### Verdict

**The blocking statement does not apply and is therefore not recorded.** The
exact 848 clean-test images *are* obtainable, through a documented reconstruction
from `dataset/` that has already been validated by prediction reproduction.

This **overturns** the X0 warning that Phase X1 is blocked by image availability.
X0 reached that conclusion by repeating a prior phase's wording about
`dataset_clean/Testing` without checking whether the images existed elsewhere.

Two conditions attach, and Phase X1 must honour both:

1. Images must be reconstructed from `dataset/` by filename against
   `split_manifest.json`, into a scratch location. `dataset_clean/` and
   `dataset/` remain read-only.
2. Content identity must be re-established by reproducing the reference
   predictions, not assumed from the filename match.

**No copy, substitution or reconstruction was performed by this addendum.** No
alternative location is treated as authoritative on its own.

---

## 5. Scientific recheck

Every finding below was independently recomputed. None was read from the X0
outputs.

| Finding | Independent result | Agrees with X0 |
|---|---|---|
| Grad-CAM++ identity | `pytorch_grad_cam.grad_cam_plusplus.GradCAMPlusPlus`; distinct from `GradCAM`; `grads_power_2` and `grads_power_3` present in `get_cam_weights` | yes |
| Hooked CNN layer | `model.cnn_stages[-1]`, type `Conv2dNormActivation`, activation `[2, 1280, 7, 7]` | yes |
| Gate tensor | input `[2, 1024]`, output `[2, 2]`, normalised on dim −1, rows sum to 1 (atol 1e-6) | yes |
| Auxiliary heads | `aux_cnn.weight [4,512]`, `aux_cnn.bias [4]`, `aux_vit.weight [4,512]`, `aux_vit.bias [4]` in checkpoint; aux logits `[2,4]` returned at inference | yes |
| SDPA attention | `attn_implementation = "sdpa"`; `output_attentions=True` returns `None` | yes |
| Eager attention | 12 tensors of `[2, 12, 197, 197]`, rows sum to 1 | yes |
| Prediction neutrality | max abs Δ fused probability `8.94e-08`, below 1e-6; argmax unchanged; sdpa restore exact | yes |

All seven X0 scientific findings are confirmed.

### Scope limit on the neutrality result

The `8.94e-08` figure was measured on a **two-image synthetic probe batch**. It
is not evidence about the 848 clean-test images and is **not extended to them**.
Phase X1 must re-establish neutrality per seed on the real evaluation set before
any explanation is attached to a verified result.

---

## Implementation-plan status

`CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md` (SHA-256 `521b0e0f30444d45…`) was
searched for `Phase X0` … `Phase X7`.

**Result: no explainability phases are present.** The plan defines Phases 0–10
for the classification study; the dual-path attribution work has no approved
phase numbering.

Accordingly `EXPLAINABILITY_PLAN_AMENDMENT_PROPOSAL.md` is written to this
directory, defining X0–X7. **The implementation-plan file itself is not
modified.** Adding these phases to the approved plan requires explicit user
approval.

---

## Acceptance conditions

| Condition | Result |
|---|---|
| Existing X0 outputs remain unchanged | PASS — all 7 re-hashed; none written by this phase |
| No protected artifact modified | PASS within verifiable scope — 150/150 baseline digests match, 0 changes |
| Stale hashes identified rather than silently replaced | PASS — one mismatch reported with its cause and timeline; X0 manifest untouched |
| Preservation claims match available evidence | PASS — "no changes detected within the verifiable scope"; scope and exclusions stated |
| Dataset availability checked recursively and by signature | PASS — 4 paths, 7,023 files signature-classified |
| Implementation-plan status explicit | PASS — X0–X7 absent; proposal written, plan unmodified |
| No Phase X1 code implemented | PASS |
| No model inference presented as full-dataset verification | PASS — synthetic probe scope stated explicitly |
| No anatomical-localisation claim | PASS — none made |

---

## Corrections the X0 record should carry

1. `sha256_manifest.json` contains one stale digest. It is left as issued; the
   correct current digests are in `final_manifest.json`.
2. The X0 preservation claim should read "no changes detected within the
   verifiable scope", not "all protected artifacts unchanged".
3. The X0 warning that Phase X1 is blocked by clean-test image availability is
   **withdrawn**. The images are obtainable by documented reconstruction from
   `dataset/`, subject to content re-verification.
4. `dataset_clean/Testing` does not exist; it is not an empty directory.
