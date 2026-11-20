# Session 2 - Code Upload Contents

Built from a strict **allowlist**: every file named individually, so the archive
cannot silently acquire a dataset, a checkpoint, a virtualenv or an application
file. A denylist would eventually let something through.

| | |
|---|---|
| file | `kaggle/upload/fusionv2_code_session2_v1.zip` |
| SHA-256 | `33bae5e3b79bb84c25cea27bde6b1a7a3dda850798b86560b3cefa9bc1fb26a0` |
| size | 56.2 KB |
| files | 15 |

Checksum file: `kaggle/upload/fusionv2_code_session2_v1.zip.sha256`

## Contents

| Path | Bytes | SHA-256 (first 16) |
|---|---:|---|
| `src/fusion_v2.py` | 7056 | `dd1a1180459b9eb1` |
| `src/train_ablation.py` | 19571 | `ed8bf84c4953a3c4` |
| `src/make_clean_split.py` | 6298 | `07557298a6aa127b` |
| `src/seeding.py` | 2580 | `a34679fc1070ab15` |
| `src/hybrid_model.py` | 1525 | `6ae47d39bf3de158` |
| `src/publication/__init__.py` | 149 | `fed671cc2354b756` |
| `src/publication/session_scope.py` | 1855 | `51cdb4b7d7b31a41` |
| `src/publication/session1_config.py` | 6583 | `1d3c82c64fca325f` |
| `src/publication/session2_config.py` | 8298 | `9344e3ff191e9df5` |
| `src/publication/session2_driver.py` | 21294 | `e92825820052302c` |
| `src/publication/train_ablation_variant.py` | 25880 | `f2351e9093aeaa19` |
| `src/publication/validate_run_outputs.py` | 8282 | `22e148f0a5a40085` |
| `src/publication/ablation_models.py` | 6440 | `47e277c903dc2150` |
| `src/publication/run_ablations.py` | 33629 | `7ef4ca1f13b41872` |
| `src/publication/test_ablations.py` | 19707 | `5c2899b5f7485820` |

## Why each file is present

| File | Role |
|---|---|
| `src/fusion_v2.py` | the model under ablation. **Frozen** - imported, never edited |
| `src/train_ablation.py` | transforms, split logic, evaluation, dump helpers. **Frozen** |
| `src/make_clean_split.py` | rebuilds the duplicate-clean split on Kaggle |
| `src/seeding.py` | seeding helpers |
| `src/hybrid_model.py` | imported by `train_ablation.py`; required for it to load |
| `src/publication/session_scope.py` | session registry - refuses a variant another session owns |
| `src/publication/session1_config.py` | needed by the registry to answer "which session owns this?" |
| `src/publication/session2_config.py` | scope, preflight fingerprint, checkpoint manifest, artifact contract |
| `src/publication/session2_driver.py` | preflight, checkpoint verification, smoke, grid, evaluation, packaging |
| `src/publication/train_ablation_variant.py` | variant trainer **and** the evaluation-only path |
| `src/publication/validate_run_outputs.py` | completeness validator, overfitting report |
| `src/publication/ablation_models.py` | the branch-only controls - used by two Session 2 variants |
| `src/publication/run_ablations.py` | variant definitions the trainer resolves against |
| `src/publication/test_ablations.py` | the 17-test suite run before training |

## Deliberately excluded

| Excluded | Reason |
|---|---|
| recovered `.pth` checkpoints | shipped as a **separate 1.02 GB archive**; 1.1 GB does not belong in a code package |
| `dataset/`, `dataset_ood/`, `dataset_clean/` | uploaded as the data input; the split is rebuilt and fingerprinted on Kaggle |
| Session 1 output archives | prior results; uploading them risks their being mistaken for new ones |
| `env/`, `env_win/` | virtualenvs; Kaggle supplies its own |
| `runs*/`, `results*/`, `publication_runs/`, `publication_results/` | existing outputs |
| `src/streamlit_app.py`, `*.bat` | application and launcher files |
| notebooks | imported through the Kaggle UI, not as a dataset |
| secrets | none exist here; the allowlist would exclude them regardless |
