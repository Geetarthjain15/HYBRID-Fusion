from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "publication_results" / "final_paper_fusion"
DOCX = OUT / "RESULTS_AND_EVIDENCE_PRESENTATION_GUIDE.docx"
PDF = OUT / "qa/results_evidence_guide/RESULTS_AND_EVIDENCE_PRESENTATION_GUIDE.pdf"
checks: dict[str, bool] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


checks["docx_exists"] = DOCX.exists()
checks["docx_container_valid"] = zipfile.is_zipfile(DOCX)
with zipfile.ZipFile(DOCX) as archive:
    document_xml = archive.read("word/document.xml").decode("utf-8")
for phrase in (
    "99.14 ± 0.10%",
    "98.27 ± 0.07%",
    "69.53 ± 1.98%",
    "97.10%",
    "contextual reference only",
    "Patient-level independence cannot be verified",
    "no detectable accuracy benefit under this protocol",
):
    checks[f"contains_{phrase}"] = phrase in document_xml

checks["prohibited_claims_are_framed_as_warnings"] = (
    "Do not state" in document_xml and "Avoid this wording" in document_xml
)

reader = PdfReader(str(PDF))
text = "".join(page.extract_text() or "" for page in reader.pages)
checks["pdf_nine_pages"] = len(reader.pages) == 9
checks["pdf_searchable"] = len(text) > 10000
checks["rendered_pages_present"] = len(list((OUT / "qa/results_evidence_guide").glob("page-*.png"))) == 9

baseline = json.loads((ROOT / "publication_results/audit/baseline_manifest.json").read_text(encoding="utf-8"))
protected = baseline["inventory"]["prediction_csvs"] + baseline["inventory"]["metric_files"]
changed = []
for item in protected:
    path = ROOT / item["path"]
    if not path.exists() or sha256(path) != item["sha256"]:
        changed.append(item["path"])
checks["protected_artifacts_unchanged"] = not changed

result = {
    "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks,
    "page_count": len(reader.pages),
    "protected_artifacts_checked": len(protected),
    "protected_artifacts_changed": changed,
}
path = OUT / "qa/results_evidence_guide/VALIDATION.json"
path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["status"] == "PASS" else 1)
