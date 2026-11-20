"""Fast acceptance tests for Phase 3C."""
import csv, json, sys
from pathlib import Path

root=Path(sys.argv[sys.argv.index("--project-root")+1]).resolve() if "--project-root" in sys.argv else Path.cwd()
out=root/"publication_results/ablations/publication_artifacts"
src=json.loads((root/"publication_results/ablations/final_analysis/final_ablation_analysis.json").read_text(encoding="utf-8"))
tests=[]
def check(name,ok): tests.append((name,bool(ok))); print(("PASS" if ok else "FAIL"),name)
check("Phase 3B accepted",src.get("acceptance_status")=="PASS")
rows=list(csv.DictReader((out/"tables/final_ablation_table.csv").open(encoding="utf-8")))
check("eight models in ablation table",len(rows)==8)
ref=next(r for r in rows if r["model"]=="full_fusion_v2")
check("reference accuracy preserved",abs(float(ref["accuracy_mean_pct"])-98.2704402516)<1e-8)
check("reference macro-F1 preserved",abs(float(ref["macro_f1_mean_pct"])-98.2761297247)<1e-8)
check("verdicts agree",all(r["verdict"]==next((x["verdict"] for x in src["component_conclusions"] if x["variant"]==r["model"]),"reference") for r in rows))
for stem in ("ablation_accuracy","reference_confusion","ablation_confusion_comparison","paired_disagreements","reference_reliability","gate_mass_by_class"):
    check(stem+" PNG",(out/f"figures/{stem}.png").stat().st_size>10000)
    check(stem+" PDF",(out/f"figures/{stem}.pdf").stat().st_size>5000)
md=(out/"PHASE3C_ARTIFACTS.md").read_text(encoding="utf-8").lower()
check("no equivalence claim","statistically equivalent" not in md)
check("gate caveat","not causal contribution" in md)
check("uncertainty separated","not a confidence interval" in md)
man=json.loads((out/"artifact_manifest.json").read_text(encoding="utf-8"))
check("no training",man["training_performed"] is False)
check("model unmodified flag",man["model_modified"] is False)
failed=[n for n,ok in tests if not ok]
print(f"{len(tests)-len(failed)}/{len(tests)} passed")
(out/"TEST_REPORT.md").write_text("# Phase 3C test report\n\n"+"\n".join(
    f"- {'PASS' if ok else 'FAIL'} — {name}" for name,ok in tests)+
    f"\n\n**Result: {len(tests)-len(failed)}/{len(tests)} passed.**\n",encoding="utf-8")
raise SystemExit(1 if failed else 0)
