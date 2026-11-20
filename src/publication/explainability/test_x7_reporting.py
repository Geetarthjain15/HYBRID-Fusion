"""Preflight and post-generation acceptance checks for Phase X7."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FORBIDDEN = [
    "clinically ready", "anatomically correct", "red pixels are tumour",
    "the model looks at the lesion", "attention is the explanation",
    "grad-cam++ validates clinical reasoning",
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root", default=".")
    ap.add_argument("--post", action="store_true"); args = ap.parse_args()
    root = Path(args.project_root).resolve()
    results = root / "publication_results/explainability"
    required = [
        results / "audit/explainability_architecture_audit.json",
        results / "x1_reconstruction/x1_reconstruction.json",
        results / "x2_instrumentation/x2_neutrality.json",
        results / "x2b_token_attribution/x2b_neutrality.json",
        results / "x3_attribution/x3_attribution.json",
        results / "x4_sanity/x4_sanity.json",
        results / "x5_faithfulness/x5_faithfulness.json",
        results / "x6_cross_source/x6_cross_source.json",
    ]
    checks = []
    checks.append(("inputs_exist", all(p.exists() for p in required)))
    if checks[-1][1]:
        x2, x2b, x6 = load(required[2]), load(required[3]), load(required[7])
        checks += [
            ("original_x2_remains_fail", x2.get("acceptance_status") == "FAIL"),
            ("x2b_replacement_pass", str(x2b.get("overall", "")).upper() == "PASS"),
            ("x6_pass", x6.get("acceptance_status") == "PASS"),
            ("x5_x6_protocol_lock_match", load(required[6])["protocol_lock_sha256"] == x6["protocol_lock_sha256"]),
            ("x6_scope", x6["n_evaluated_images"] == 373 and x6["seeds"] == [42, 43, 44]),
        ]
    if args.post:
        out = results / "x7_reporting"
        checks += [
            ("report_exists", (out / "X7_EXPLAINABILITY_REPORT.md").exists()),
            ("traceability_exists", (out / "EVIDENCE_TRACEABILITY.csv").exists()),
            ("limitation_register_exists", (out / "LIMITATION_REGISTER.md").exists()),
        ]
        png = {p.stem for p in (out / "figures").glob("*.png") if not p.stem.startswith("smoke_")}
        pdf = {p.stem for p in (out / "figures").glob("*.pdf") if not p.stem.startswith("smoke_")}
        checks.append(("figure_pairs", bool(png) and png == pdf))
        trace = list(csv.DictReader((out / "EVIDENCE_TRACEABILITY.csv").open(encoding="utf-8")))
        checks.append(("traceability_nonempty", len(trace) >= 100 and all(r["source_file"] and r["source_locator"] for r in trace)))
        limits = (out / "LIMITATION_REGISTER.md").read_text(encoding="utf-8")
        checks.append(("eight_x0_limitations", sum(line[:2].rstrip(".").isdigit() for line in limits.splitlines() if line and line[0].isdigit()) >= 8))
        text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in out.rglob("*.md")).lower()
        checks.append(("prohibited_language_absent", not any(term in text for term in FORBIDDEN)))
        phase_rows = list(csv.DictReader((out / "tables/phase_status.csv").open(encoding="utf-8")))
        status = {r["phase"]: r["status"] for r in phase_rows}
        checks.append(("x2_and_x2b_distinct", status.get("X2") == "FAIL" and status.get("X2b") == "PASS"))
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    failed = [n for n, ok in checks if not ok]
    print(f"{len(checks)} checks, {len(checks)-len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
