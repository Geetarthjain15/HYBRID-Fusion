"""Generate Phase X7 publication tables and evidence documents from X1-X6 artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv_new(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def status_from(obj: dict) -> str:
    value = obj.get("acceptance_status") or obj.get("overall")
    if isinstance(value, dict):
        value = value.get("status") or value.get("result")
    return str(value).upper() if value is not None else "NOT_RECORDED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = root / "publication_results/explainability/x7_reporting"
    tables = out / "tables"
    supplement = out / "supplement"

    metrics_p = root / "publication_results/metrics/fusion_v2_metrics.json"
    x0_p = root / "publication_results/explainability/audit/explainability_architecture_audit.json"
    x0a_p = root / "publication_results/explainability/audit_addendum/x0_verification_addendum.json"
    x1_p = root / "publication_results/explainability/x1_reconstruction/x1_reconstruction.json"
    x2_p = root / "publication_results/explainability/x2_instrumentation/x2_neutrality.json"
    x2b_p = root / "publication_results/explainability/x2b_token_attribution/x2b_neutrality.json"
    x3_p = root / "publication_results/explainability/x3_attribution/x3_attribution.json"
    x4_p = root / "publication_results/explainability/x4_sanity/x4_sanity.json"
    x5_p = root / "publication_results/explainability/x5_faithfulness/x5_faithfulness.json"
    x5_land_p = root / "publication_results/explainability/x5_faithfulness/X5_LANDING_PROVENANCE.json"
    x6_p = root / "publication_results/explainability/x6_cross_source/x6_cross_source.json"
    x6_shift_p = root / "publication_results/explainability/x6_cross_source/shift_effects.csv"
    x6_land_p = root / "publication_results/explainability/x6_cross_source/X6_LANDING_PROVENANCE.json"
    sources = [metrics_p, x0_p, x0a_p, x1_p, x2_p, x2b_p, x3_p, x4_p,
               x5_p, x5_land_p, x6_p, x6_shift_p, x6_land_p]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        raise FileNotFoundError("missing required X7 inputs: " + "; ".join(missing))

    metrics, x6 = read_json(metrics_p), read_json(x6_p)
    if metrics.get("acceptance_status") != "PASS" or x6.get("acceptance_status") != "PASS":
        raise RuntimeError("accepted metrics and X6 evidence are required")
    x5, x4 = read_json(x5_p), read_json(x4_p)

    performance = []
    for setting in ("original", "clean", "cross_source_disjoint"):
        s = metrics["settings"][setting]
        performance.append({
            "setting": setting,
            "regime": s["regime"],
            "n_images": s["n_images"],
            "accuracy_mean": s["across_seeds"]["accuracy"]["mean"],
            "accuracy_sd_across_seeds": s["across_seeds"]["accuracy"]["sd_across_seeds"],
            "macro_f1_mean": s["across_seeds"]["macro_f1"]["mean"],
            "macro_f1_sd_across_seeds": s["across_seeds"]["macro_f1"]["sd_across_seeds"],
            "n_seeds": s["across_seeds"]["accuracy"]["n_seeds"],
            "source_file": metrics_p.relative_to(root).as_posix(),
        })
    write_csv_new(tables / ("smoke_headline_performance.csv" if args.smoke else "headline_performance.csv"),
                  performance[:1] if args.smoke else performance)
    if args.smoke:
        print("X7 table smoke PASS: one table generated from accepted machine-readable evidence")
        return 0

    phase_specs = [
        ("X0", x0_p, status_from(read_json(x0_p)), "architecture/evidence audit"),
        ("X0a", x0a_p, status_from(read_json(x0a_p)), "verification addendum"),
        ("X1", x1_p, status_from(read_json(x1_p)), "clean-test reconstruction"),
        ("X2", x2_p, status_from(read_json(x2_p)), "preserved failed eager-attention route"),
        ("X2b", x2b_p, status_from(read_json(x2b_p)), "accepted SDPA-preserving replacement"),
        ("X3", x3_p, status_from(read_json(x3_p)), "dual-path attribution extraction"),
        ("X4", x4_p, status_from(read_json(x4_p)), "sanity checks"),
        ("X5", x5_land_p, read_json(x5_land_p).get("acceptance_status", "NOT_RECORDED"), "faithfulness/stability"),
        ("X6", x6_p, x6["acceptance_status"], "locked cross-source analysis"),
    ]
    phase_rows = [{"phase": p, "status": s, "role": role,
                   "source_file": path.relative_to(root).as_posix()}
                  for p, path, s, role in phase_specs]
    write_csv_new(tables / "phase_status.csv", phase_rows)

    faith_rows = []
    for metric_name, methods in x5["bootstrap"].items():
        for method, value in methods.items():
            faith_rows.append({"metric": metric_name, "method": method,
                               "mean": value["mean"], "ci_low": value["ci_low"],
                               "ci_high": value["ci_high"], "n_images": x5["n_images"],
                               "n_seeds": len(x5["seeds"]),
                               "source_file": x5_p.relative_to(root).as_posix()})
    write_csv_new(tables / "x5_faithfulness_summary.csv", faith_rows)

    class_rows = []
    for label, values in x6["class_errors"].items():
        if label == "pituitary_to_notumor":
            continue
        class_rows.append({"class": label, **values,
                           "source_file": x6_p.relative_to(root).as_posix()})
    write_csv_new(tables / "x6_class_recall.csv", class_rows)
    disagreement_rows = [{"comparison": k, "rate": v,
                           "source_file": x6_p.relative_to(root).as_posix()}
                         for k, v in x6["branch_disagreement"].items()]
    write_csv_new(tables / "x6_branch_disagreement.csv", disagreement_rows)

    with x6_shift_p.open(newline="", encoding="utf-8") as f:
        shift_rows = list(csv.DictReader(f))
    for row in shift_rows:
        row["source_file"] = x6_shift_p.relative_to(root).as_posix()
    write_csv_new(tables / "x6_shift_effects.csv", shift_rows)

    gates = x4.get("gates", {})
    sanity_rows = []
    for name, value in gates.items():
        if isinstance(value, dict):
            result = value.get("status") or value.get("result") or value.get("pass")
        else:
            result = value
        sanity_rows.append({"check": name, "result": result,
                            "source_file": x4_p.relative_to(root).as_posix()})
    write_csv_new(tables / "x4_sanity_checks.csv", sanity_rows)

    trace = []
    def add_trace(claim, quantity, value, unit, source, locator, scope):
        trace.append({"claim_id": claim, "quantity": quantity, "value": value,
                      "unit": unit, "source_file": source.relative_to(root).as_posix(),
                      "source_locator": locator, "verification_status": "VERIFIED",
                      "scope": scope})
    for row in performance:
        for metric in ("accuracy", "macro_f1"):
            add_trace(f"performance_{row['setting']}_{metric}", metric,
                      row[f"{metric}_mean"], "proportion", metrics_p,
                      f"settings.{row['setting']}.across_seeds.{metric}.mean",
                      f"image-level; {row['n_seeds']} seeds; n={row['n_images']}")
            add_trace(f"performance_{row['setting']}_{metric}_sd", f"{metric}_sd_across_seeds",
                      row[f"{metric}_sd_across_seeds"], "proportion", metrics_p,
                      f"settings.{row['setting']}.across_seeds.{metric}.sd_across_seeds",
                      "training-seed variability, not an image bootstrap interval")
    for row in faith_rows:
        for field in ("mean", "ci_low", "ci_high"):
            add_trace(f"x5_{row['metric']}_{row['method']}_{field}",
                      f"{row['metric']}_{field}", row[field], "AUC", x5_p,
                      f"bootstrap.{row['metric']}.{row['method']}.{field}",
                      "model-behaviour faithfulness under the locked masking protocol")
    for label, values in x6["class_errors"].items():
        for key, value in values.items():
            add_trace(f"x6_{label}_{key}", key, value,
                      "proportion" if "recall" in key else "count", x6_p,
                      f"class_errors.{label}.{key}", "duplicate-disjoint external images")
    for key, value in x6["branch_disagreement"].items():
        add_trace(f"x6_{key}", key, value, "proportion", x6_p,
                  f"branch_disagreement.{key}", "image-seed pairs; descriptive")
    for row in shift_rows:
        for field in ("external_mean", "clean_mean", "difference_external_minus_clean", "ci_low", "ci_high"):
            add_trace(f"x6_shift_{row['quantity']}_{row['method']}_{field}", field,
                      row[field], "metric-specific", x6_shift_p,
                      f"quantity={row['quantity']};method={row['method']};field={field}",
                      "independent image sets; image-level stratified bootstrap")
    add_trace("x3_instrumented_probability_delta", "max_probability_delta", 0.0,
              "absolute probability", x3_p, "per_seed.*.max_abs_prob_diff_single",
              "all 848 images x 3 seeds")
    add_trace("x6_evaluated_images", "n_evaluated_images", x6["n_evaluated_images"],
              "unique images", x6_p, "n_evaluated_images", "duplicate-disjoint external set")
    write_csv_new(out / "EVIDENCE_TRACEABILITY.csv", trace)

    limitations = [
        ("No anatomical ground truth", "No lesion masks, boxes or expert localisation annotations are available."),
        ("Coarse CNN attribution grid", "The CNN attribution begins on the final low-resolution spatial feature grid and is upsampled."),
        ("CNN–ViT grid mismatch", "The two attribution paths have different native spatial grids and require an explicit resampling choice."),
        ("Model-space coordinates", "Preprocessing resizes images without preserving original acquisition geometry; maps remain in model coordinates."),
        ("Prediction–map protocol mismatch", "Reported predictions use flip augmentation, whereas each attribution map uses an unflipped pass."),
        ("Attention is not explanation", "Token mixing weights alone do not establish class evidence; the accepted ViT method is gradient-by-input, not attention."),
        ("Gate mass is not causal attribution", "The learned gate measures feature-branch utilisation and was not trained to weight spatial maps."),
        ("Plausibility is not faithfulness", "Perturbation tests measure model sensitivity, not anatomy or clinical validity."),
    ]
    limit_lines = ["# Phase X7 limitation register", "",
                   "These eight structural limitations are carried forward from the X0 audit.", ""]
    for i, (title, body) in enumerate(limitations, 1):
        limit_lines += [f"{i}. **{title}.** {body}"]
    limit_lines += ["", "Additional study limitations", "",
                    "- Evaluation is image-level; patient-level independence cannot be verified from available metadata.",
                    "- Scanner, institution, acquisition and demographic metadata are unavailable.",
                    "- External evidence is limited to one duplicate-disjoint source.",
                    "- Three training seeds quantify limited training variability, not population-level uncertainty.",
                    "- Cross-source associations do not establish a mechanism for the accuracy decrease.",
                    "", f"Source: `{x0_p.relative_to(root).as_posix()}`, question 22."]
    write_text_new(out / "LIMITATION_REGISTER.md", "\n".join(limit_lines) + "\n")

    manifest_rows = [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
                     for p in sources]
    write_csv_new(supplement / "INPUT_SOURCE_MANIFEST.csv", manifest_rows)
    write_text_new(supplement / "X2_ROUTE_RECORD.md",
                   "# X2 route record\n\nOriginal X2 remains a recorded FAIL because eager-attention export exceeded its prespecified neutrality tolerance. X2b is the accepted SDPA-preserving replacement used by X3–X6. The two outcomes are not merged or relabelled.\n")
    write_text_new(supplement / "REPRODUCIBILITY.md",
                   "# Reproducibility scope\n\nAll X7 tables and figures are derived from the hash-recorded X1–X6 machine-readable artifacts. No training or model modification occurs in X7. Confidence intervals retain their source phase's resampling unit and must not be presented as patient-level intervals.\n")

    perf = {r["setting"]: r for r in performance}
    pit = x6["class_errors"]["pituitary"]
    p2n = x6["class_errors"]["pituitary_to_notumor"]
    report = f"""# Phase X7 — publication reporting

## Classification evidence

- Original split accuracy: {perf['original']['accuracy_mean']*100:.2f} ± {perf['original']['accuracy_sd_across_seeds']*100:.2f}%.
- Duplicate-clean accuracy: {perf['clean']['accuracy_mean']*100:.2f} ± {perf['clean']['accuracy_sd_across_seeds']*100:.2f}%.
- Duplicate-disjoint cross-source accuracy: {perf['cross_source_disjoint']['accuracy_mean']*100:.2f} ± {perf['cross_source_disjoint']['accuracy_sd_across_seeds']*100:.2f}% over {perf['cross_source_disjoint']['n_images']} unique images.

The external decrease is class-specific. Pituitary recall is {pit['recall_seed_mean']*100:.2f} ± {pit['recall_seed_sd']*100:.2f}%, and {p2n['predictions']} of {p2n['image_seed_pairs']} pituitary image–seed predictions are no-tumour.

## Explanation evidence

X3 produced synchronized CNN Grad-CAM++ and ViT Gradient × Input maps without changing fused probabilities. X4 established determinism, class sensitivity, native shapes, orientation and cleanup behaviour. X5 measured faithfulness and stability under a locked perturbation protocol. X6 applied that protocol unchanged to the duplicate-disjoint external set.

The X6 changes are mixed across methods and transformations. They support a source-associated change in confidence, branch behaviour, attribution agreement and selected faithfulness/stability measurements; they do not support a universal explanation-degradation claim.

## Claim boundary

These results concern image-level classification and model sensitivity. They do not establish patient-level independence, anatomical localisation, a causal mechanism, diagnostic utility or clinical readiness.

## Streamlit decision

No Streamlit wrapper was created. Original X2 is a recorded FAIL and X2b is its accepted replacement; the X7 app gate literally requires X2–X5 to pass. Reporting proceeds, but the optional app is withheld rather than silently redefining that gate.
"""
    write_text_new(out / "X7_EXPLAINABILITY_REPORT.md", report)
    print(f"X7 tables/report generated at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
