"""Build the self-contained, hash-pinned Kaggle package for Phase X6.

This script performs no model inference.  It refuses to overwrite an existing
bundle and copies only the inputs required by the reviewed X6 full grid.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
OUT = ROOT / "publication_runs/explainability/kaggle_x6_v2"
UPLOAD = OUT / "upload"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_file(relative: str) -> None:
    source = ROOT / relative
    if not source.is_file():
        raise RuntimeError("required file absent: %s" % source)
    target = UPLOAD / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(relative: str) -> None:
    source, target = ROOT / relative, UPLOAD / relative
    if not source.is_dir():
        raise RuntimeError("required directory absent: %s" % source)
    if target.exists():
        raise RuntimeError("refusing to overwrite: %s" % target)
    shutil.copytree(source, target)


def notebook() -> dict:
    def code(text):
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": [line + "\n" for line in text.splitlines()]}
    def markdown(text):
        return {"cell_type": "markdown", "metadata": {},
                "source": [line + "\n" for line in text.splitlines()]}
    cells = [
        markdown("# Phase X6 — locked cross-source explanation analysis\n"
                 "Run with a Kaggle GPU. This notebook trains nothing and does not alter X5."),
        code("from pathlib import Path\nimport hashlib, json, os, shutil, subprocess, sys, zipfile\n"
             "IN=Path('/kaggle/input'); ROOT=Path('/kaggle/working/x6_project')\n"
             "assert not ROOT.exists(), f'refusing to overwrite {ROOT}'\n"
             "manifests=[]\n"
             "for p in IN.rglob('UPLOAD_MANIFEST.json'):\n"
             " try:\n  j=json.loads(p.read_text())\n except Exception:\n  continue\n"
             " if j.get('phase')=='X6 Kaggle GPU bundle': manifests.append(p)\n"
             "if len(manifests)==1:\n"
             " shutil.copytree(manifests[0].parent,ROOT)\n"
             " print('using extracted Kaggle dataset',manifests[0].parent)\n"
             "else:\n"
             " zips=list(IN.rglob('fusion_v2_x6_upload_v2.zip'))\n"
             " assert len(zips)==1, f'expected one X6 manifest or zip; manifests={manifests}, zips={zips}'\n"
             " with zipfile.ZipFile(zips[0]) as z: z.extractall(ROOT)\n"
             " print('using unextracted zip',zips[0])\n"
             "print('ROOT',ROOT)"),
        code("manifest=json.loads((ROOT/'UPLOAD_MANIFEST.json').read_text())\n"
             "bad=[]\n"
             "for rel,want in manifest['files'].items():\n"
             " p=ROOT/rel\n"
             " got=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'MISSING'\n"
             " if got!=want: bad.append((rel,want,got))\n"
             "assert not bad, bad[:5]\n"
             "print(f\"verified {len(manifest['files'])}/{len(manifest['files'])} input hashes\")\n"
             "assert manifest['protocol_lock_sha256']=='a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5'"),
        code("import torch\nassert torch.cuda.is_available(), 'GPU is not enabled'\n"
             "try:\n import pytorch_grad_cam\nexcept ImportError:\n subprocess.run([sys.executable,'-m','pip','install','-q','grad-cam'],check=True)\n import pytorch_grad_cam\n"
             "print(torch.__version__, torch.cuda.get_device_name(0))\n"
             "os.chdir(ROOT)"),
        code("subprocess.run([sys.executable,'src/publication/explainability/test_cross_source.py'],check=True)"),
        code("subprocess.run([sys.executable,'-m','src.publication.explainability.cross_source',"
             "'--project-root','.', '--smoke','--device','cuda','--skip-preservation'],check=True)"),
        code("subprocess.run([sys.executable,'-m','src.publication.explainability.cross_source',"
             "'--project-root','.', '--all','--device','cuda','--skip-preservation'],check=True)"),
        code("result=ROOT/'publication_results/explainability/x6_cross_source'\n"
             "full=json.loads((result/'x6_cross_source.json').read_text())\n"
             "assert full['acceptance_status']=='PASS', full\n"
             "assert full['n_selector_images']==373 and full['n_evaluated_images']==373\n"
             "assert full['seeds']==[42,43,44]\n"
             "assert full['protocol_lock_sha256']=='a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5'\n"
             "print(json.dumps(full['neutrality'],indent=2))"),
        code("OUT=Path('/kaggle/working/x6_output')\n"
             "assert not OUT.exists(); OUT.mkdir()\n"
             "shutil.copytree(result,OUT/'x6_cross_source')\n"
             "dig={}\n"
             "for p in sorted(OUT.rglob('*')):\n"
             " if p.is_file(): dig[p.relative_to(OUT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()\n"
             "(OUT/'KAGGLE_OUTPUT_SHA256.json').write_text(json.dumps(dig,indent=2))\n"
             "archive=Path('/kaggle/working/x6_results.zip')\n"
             "with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:\n"
             " for p in sorted(OUT.rglob('*')):\n"
             "  if p.is_file(): z.write(p,p.relative_to(OUT))\n"
             "print(archive,archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest())"),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    if OUT.exists():
        raise RuntimeError("refusing to overwrite existing package: %s" % OUT)
    UPLOAD.mkdir(parents=True)
    code = [
        "src/fusion_v2.py", "src/train_ablation.py", "src/hybrid_model.py", "src/seeding.py",
        "src/publication/fusion_v2_gradcam.py",
        "src/publication/explainability/cnn_attribution.py",
        "src/publication/explainability/vit_attribution.py",
        "src/publication/explainability/run_attribution.py",
        "src/publication/explainability/composite.py",
        "src/publication/explainability/faithfulness.py",
        "src/publication/explainability/cross_source.py",
        "src/publication/explainability/test_cross_source.py",
    ]
    for item in code:
        copy_file(item)
    for seed in (42, 43, 44):
        copy_file("publication_runs/recovered_checkpoints/clean/full_fusion_v2/seed_%d/checkpoint.pth" % seed)
        copy_file("runs_v2_clean/fusion_v2_seed%d/preds_dataset_ood.csv" % seed)
        copy_file("runs_v2_clean/fusion_v2_seed%d/gates_dataset_ood.csv" % seed)
    for item in [
        "publication_results/ablations/checkpoint_recovery/import_manifest.json",
        "publication_results/explainability/x5_protocol/x5_protocol_lock.json",
        "publication_results/explainability/x5_protocol/x5_scale_decision.json",
        "publication_results/explainability/x5_faithfulness/per_image_curves.csv",
        "results/cross_source.json",
    ]:
        copy_file(item)
    copy_tree("publication_runs/explainability/reconstructed_cross_source")

    files = {p.relative_to(UPLOAD).as_posix(): sha(p)
             for p in sorted(UPLOAD.rglob("*")) if p.is_file()}
    manifest = {"phase": "X6 Kaggle GPU bundle",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_lock_sha256":
                    "a553c93b3eeb654f7adaed319a499404d7a83efbe9dac576f482a0b10cf2a1a5",
                "scope": "373 duplicate-disjoint images x 3 seeds x 6 locked methods",
                "training": False, "external_label_tuning": False,
                "estimated_t4_hours_from_measured_x5": 1.4270969336,
                "files": files}
    (UPLOAD / "UPLOAD_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    nb = notebook()
    (OUT / "kaggle_x6_notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    guide = ("# X6 Kaggle steps\n\n"
             "1. Upload `fusion_v2_x6_upload_v2.zip` as one private Kaggle dataset.\n"
             "2. Create a notebook, enable a GPU, add that dataset as input, and import `kaggle_x6_notebook.ipynb`.\n"
             "3. Run all cells. Tests and one smoke run execute before the full grid.\n"
             "4. Download `x6_results.zip` from Output. Do not rename or edit its contents.\n"
             "5. Extract outside the repository and perform project-host hash/preservation checks before acceptance.\n\n"
             "Expected full runtime: approximately 1.4 hours on one Tesla T4, based on the measured X5 runtime.\n")
    (OUT / "KAGGLE_X6_STEPS.md").write_text(guide, encoding="utf-8")
    archive = OUT / "fusion_v2_x6_upload_v2.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in sorted(UPLOAD.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(UPLOAD))
    print(json.dumps({"upload_files": len(files), "archive": str(archive),
                      "bytes": archive.stat().st_size, "sha256": sha(archive),
                      "notebook": str(OUT / 'kaggle_x6_notebook.ipynb')}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
