# Session 1 - Upload Contents

Two Kaggle inputs. The code archive is built from a strict **allowlist**: every
file is named individually, so the upload cannot silently acquire a dataset, a
checkpoint, a virtualenv or an application file. A denylist would eventually let
something through.

## Archive

| | |
|---|---|
| file | `kaggle/upload/fusionv2_code_session1_v1.zip` |
| SHA-256 | `90ceaa12e28fdfe94e8f38d4352e6a55994422199bade333b46272aa195c708c` |
| size | 48.5 KB |
| files | 13 |

Checksum file: `kaggle/upload/fusionv2_code_session1_v1.zip.sha256`

## Contents

| Path | Bytes | SHA-256 (first 16) |
|---|---:|---|
| `src/fusion_v2.py` | 7056 | `dd1a1180459b9eb1` |
| `src/train_ablation.py` | 19571 | `ed8bf84c4953a3c4` |
| `src/make_clean_split.py` | 6298 | `07557298a6aa127b` |
| `src/seeding.py` | 2580 | `a34679fc1070ab15` |
| `src/hybrid_model.py` | 1525 | `6ae47d39bf3de158` |
| `src/publication/__init__.py` | 149 | `fed671cc2354b756` |
| `src/publication/session1_config.py` | 5284 | `82fd466847de6cad` |
| `src/publication/session1_driver.py` | 15154 | `4f865290d63fc593` |
| `src/publication/train_ablation_variant.py` | 19398 | `062ab9be8787a0c8` |
| `src/publication/validate_run_outputs.py` | 8282 | `22e148f0a5a40085` |
| `src/publication/ablation_models.py` | 6440 | `47e277c903dc2150` |
| `src/publication/run_ablations.py` | 33629 | `7ef4ca1f13b41872` |
| `src/publication/test_ablations.py` | 19204 | `6deb4d1ef3e5e31c` |

## Why each file is present

| File | Role |
|---|---|
| `src/fusion_v2.py` | the model under ablation. **Frozen** - imported, never edited |
| `src/train_ablation.py` | transforms, split logic, evaluation, dump helpers. **Frozen** |
| `src/make_clean_split.py` | rebuilds the duplicate-clean split on Kaggle |
| `src/seeding.py` | seeding helpers used by the trainer |
| `src/hybrid_model.py` | imported by `train_ablation.py`; required for it to load |
| `src/publication/session1_config.py` | scope, preflight fingerprint, artifact contract |
| `src/publication/session1_driver.py` | preflight, smoke test, grid, packaging |
| `src/publication/train_ablation_variant.py` | the variant trainer |
| `src/publication/validate_run_outputs.py` | completeness validator, overfitting report |
| `src/publication/ablation_models.py` | branch-only controls (imported; unused in Session 1) |
| `src/publication/run_ablations.py` | variant definitions the trainer resolves against |
| `src/publication/test_ablations.py` | the 17-test suite run before training |

## Deliberately excluded

| Excluded | Reason |
|---|---|
| `dataset/`, `dataset_ood/`, `dataset_clean/` | uploaded separately as the data input; the split is **rebuilt and fingerprinted** on Kaggle |
| `*.pth` checkpoints | not needed; every Session 1 cell trains from pretrained backbones |
| `env/`, `env_win/` | virtualenvs; Kaggle supplies its own |
| `runs*/`, `results*/`, `publication_runs/`, `publication_results/` | existing outputs; uploading them would risk them being mistaken for new results |
| `src/streamlit_app.py`, `*.bat` | application and launcher files, irrelevant to training |
| notebooks | imported through the Kaggle UI, not as a dataset |
| secrets | none exist in this repository; the allowlist would exclude them regardless |

## Data input

A separate Kaggle dataset providing:

```
dataset/Training/<class>/*.jpg
dataset/Testing/<class>/*.jpg
dataset_ood/Testing/<class>/*.jpg
```

`dataset_clean/` is **not** uploaded. The driver rebuilds it with
`make_clean_split.py` at seed 42, Hamming 8, and refuses to train unless the
result reproduces the verified fingerprint exactly (7023 pooled, 5648 clusters,
4800 train, 848 test, manifest SHA-256 `a02bcb62...`).
