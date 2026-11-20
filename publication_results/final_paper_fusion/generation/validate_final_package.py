from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "publication_results" / "final_paper_fusion"
FAILURES: list[str] = []
PASSES: list[str] = []


def check(condition: bool, message: str) -> None:
    (PASSES if condition else FAILURES).append(message)


def close(actual: float, expected: float, tolerance: float = 0.0001) -> bool:
    return abs(actual - expected) <= tolerance


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


metrics = json.loads((ROOT / "publication_results/metrics/fusion_v2_metrics.json").read_text(encoding="utf-8"))
expected = {
    "original": (1052, 0.991445, 0.991237),
    "clean": (848, 0.982704, 0.982761),
    "cross_source_disjoint": (373, 0.695264, 0.686866),
}
for setting, (n, acc, f1) in expected.items():
    row = metrics["settings"][setting]
    check(row["n_images"] == n, f"{setting}: sample count {n}")
    check(close(row["across_seeds"]["accuracy"]["mean"], acc), f"{setting}: accuracy")
    check(close(row["across_seeds"]["macro_f1"]["mean"], f1), f"{setting}: macro-F1")

rows = list(csv.DictReader((ROOT / "publication_results/ablations/cross_source/metrics_by_seed.csv").open(encoding="utf-8")))
for variant, expected_acc in (("full_fusion_v2", 0.695264), ("cnn_branch_only", 0.445040)):
    values = [float(r["accuracy"]) for r in rows if r["variant"] == variant]
    check(len(values) == 3 and close(statistics.mean(values), expected_acc), f"cross-source {variant}: three-seed mean")

manuscript = (OUT / "FUSION_NETWORK_MANUSCRIPT.md").read_text(encoding="utf-8")
for required in ("99.14 ± 0.10%", "98.27 ± 0.07%", "69.53 ± 1.98%", "contextual only"):
    check(required in manuscript, f"manuscript contains required statement: {required}")
for prohibited in ("AGMF-Net", "clinically ready", "statistically outperforms BRAIN-META", "splits are patient-independent"):
    check(prohibited not in manuscript, f"manuscript excludes prohibited wording: {prohibited}")

for path in (OUT / "FUSION_NETWORK_MANUSCRIPT.docx", OUT / "supplement/FUSION_NETWORK_SUPPLEMENT.docx"):
    check(path.exists() and zipfile.is_zipfile(path), f"valid DOCX container: {path.name}")

for path, minimum_pages in ((OUT / "FUSION_NETWORK_MANUSCRIPT.pdf", 10),
                            (OUT / "supplement/FUSION_NETWORK_SUPPLEMENT.pdf", 3)):
    pdf = PdfReader(str(path))
    extracted = "".join(page.extract_text() or "" for page in pdf.pages)
    check(len(pdf.pages) >= minimum_pages, f"PDF page count: {path.name}")
    check(len(extracted) > 5000, f"searchable PDF text: {path.name}")

main_pdf = PdfReader(str(OUT / "FUSION_NETWORK_MANUSCRIPT.pdf"))
link_count = sum(len(page.get("/Annots") or []) for page in main_pdf.pages)
check(link_count >= 10, "main PDF contains clickable reference links")

for number in range(1, 12):
    stem = next(OUT.glob(f"figures/figure_{number:02d}_*.png"), None)
    check(stem is not None and stem.with_suffix(".pdf").exists(), f"Figure {number} has PNG and PDF")

baseline = json.loads((ROOT / "publication_results/audit/baseline_manifest.json").read_text(encoding="utf-8"))
protected = baseline["inventory"]["prediction_csvs"] + baseline["inventory"]["metric_files"]
changed = []
for item in protected:
    path = ROOT / item["path"]
    if not path.exists() or sha256(path) != item["sha256"]:
        changed.append(item["path"])
check(not changed, f"baseline-manifest protected artifacts unchanged ({len(protected)} checked)")

report = {
    "status": "PASS" if not FAILURES else "FAIL",
    "n_pass": len(PASSES),
    "n_fail": len(FAILURES),
    "passes": PASSES,
    "failures": FAILURES,
    "pdf": {
        "manuscript_pages": len(main_pdf.pages),
        "clickable_links": link_count,
    },
    "protected_artifacts_checked": len(protected),
}
(OUT / "qa/PACKAGE_VALIDATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(1 if FAILURES else 0)
