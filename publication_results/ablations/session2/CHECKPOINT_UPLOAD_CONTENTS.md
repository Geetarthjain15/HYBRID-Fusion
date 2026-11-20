# Session 2 - Recovered Checkpoint Archive

Separate from the code archive: 1.02 GB of weights does not belong in a 56 KB
code package, and Kaggle versions the two independently.

| | |
|---|---|
| file | `kaggle/upload/fusionv2_recovered_checkpoints_v1.zip` |
| SHA-256 | `321affcdcef48f6ff257ee79e4eeb81d19f0313aaea5028bbce05b29f4e397c3` |
| size | 1.02 GB |
| checkpoints | 3 |

Checksum file: `kaggle/upload/fusionv2_recovered_checkpoints_v1.zip.sha256`
Per-checkpoint manifest: `publication_results/ablations/session2/checkpoint_manifest.json`

## Contents

| Seed | Relative path | Bytes | SHA-256 (first 24) | Expected clean-test acc | Status |
|---:|---|---:|---|---:|---|
| 42 | `clean/full_fusion_v2/seed_42/checkpoint.pth` | 365236755 | `9b328ba83df5d44b542e19fa` | 0.9823 | ACCEPTED |
| 43 | `clean/full_fusion_v2/seed_43/checkpoint.pth` | 365236755 | `9f97b6aa0c9ec3f1124e593c` | 0.9823 | ACCEPTED |
| 44 | `clean/full_fusion_v2/seed_44/checkpoint.pth` | 365236755 | `4155a2b71df99a6159e1e4a5` | 0.9835 | ACCEPTED |

Architecture for all three: **FusionV2 (gated, multi-scale, proj_dim 512)**.

## Provenance

Trained on Kaggle during the FusionV2 clean-split run, recovered from staging,
and verified in the Phase 3 addendum. Each checkpoint:

- loaded strictly against the reference architecture - 578/578 keys, 0 missing,
  0 unexpected, 0 shape mismatches;
- retained a **learned, input-dependent** gate (262,914 parameters, output
  varying across inputs) - not a fixed weight;
- reproduced its stored clean-test predictions **exactly**: 0.0000 pp accuracy
  difference, 0 predicted-label mismatches on all 848 images, maximum absolute
  probability difference ~4e-06 attributable to float non-determinism between
  CUDA training and CPU re-scoring.

Full evidence: `publication_results/ablations/checkpoint_recovery/CHECKPOINT_RECOVERY.md`

## How they are used

Only by the three `no_tta` evaluation cells, which load the weights, disable
horizontal-flip TTA, and score. They **train nothing, update nothing, and write
no checkpoint** - the validator fails an evaluation cell that contains
`checkpoint.pth`.

Each digest is verified **twice**: once in the notebook before the session
starts, and again by the trainer at load time via
`--expect-checkpoint-sha256`. A mismatch stops the run rather than scoring
unverified weights.

## Not to be modified

These are the verified reference weights. Retraining `full_fusion_v2` would
replace a verified reference with an unverified one, which is why that variant
is refused by name.
