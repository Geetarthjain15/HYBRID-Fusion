# HYBRID-Fusion

FusionV2 is a four-class brain MRI research classifier using gated, multi-scale
CNN–ViT feature fusion. This repository includes the Streamlit prototype,
evaluation code, prediction artifacts, and publication results.

> Research prototype only. It is not a clinical diagnostic system.

## Run the Streamlit prototype on Windows

Requirements:

- Python 3.11 or 3.12 (64-bit recommended)
- Git
- Approximately 4 GB of free disk space
- Internet access during first-time setup

Clone the repository and run:

```powershell
git clone https://github.com/Geetarthjain15/HYBRID-Fusion.git
cd HYBRID-Fusion
.\RUN_FUSIONV2_DEMO.bat
```

The first run creates `env_win`, installs the Python dependencies, and downloads
the three verified FusionV2 checkpoints from the repository's GitHub Release.
Each checkpoint is checked against its recorded SHA-256 hash before the app is
started. The browser interface is available at <http://localhost:8503>.

## Manual setup (Windows, macOS, or Linux)

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_checkpoints.py
python -m streamlit run src/publication/streamlit_fusion_v2.py --server.port 8503
```

## Model checkpoints

The verified checkpoints are about 348 MB each and cannot be stored as normal
Git objects because GitHub limits individual repository files to 100 MB. They
are distributed as release assets under `v1.0.0-model-checkpoints`. The download
script writes them to the exact paths expected by the prototype and refuses to
accept a file whose SHA-256 hash differs from the verification manifest.

## Included and excluded artifacts

Source code, lightweight prediction files, metrics, figures, reports, and test
artifacts are versioned. Datasets, virtual environments, transient logs,
multi-gigabyte Kaggle bundles, and duplicate training-run packages are excluded.
These are not needed to run the Streamlit inference prototype.

## Verified FusionV2 results

- Original split accuracy: 99.14 ± 0.10%
- Duplicate-clean accuracy: 98.27 ± 0.07%
- Duplicate-disjoint cross-source accuracy: 69.53 ± 1.98%

The cross-source result demonstrates substantial source-shift degradation; it
should not be interpreted as clinical readiness or universal generalization.
