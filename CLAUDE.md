# Claude Code Repository Contract

This file is authoritative for all Claude Code work in this repository.

## Required first action

Before editing or running experiments, read all of:

1. `CLAUDE_CODE_FINAL_IMPLEMENTATION_PLAN.md`
2. `results/paper/SUMMARY.md`
3. `results/paper/tables.md`
4. `src/fusion_v2.py`
5. `src/train_ablation.py`
6. `dataset_clean/split_report.json`
7. `results/cross_source.json`

Then state:

- the single phase being executed;
- files that may be created or edited;
- commands that will be run;
- expected outputs;
- acceptance checks;
- reasons the task should stop rather than guess.

Execute only one numbered phase from the implementation plan at a time. Do not continue automatically into the next phase.

## Scientific ground truth

The proposed/resultant model is **FusionV2**, not CNN-only, ViT-only, original fusion, or the logistic stacking experiment.

Verified FusionV2 summaries that new analysis must reproduce within rounding tolerance:

- Original split accuracy: `99.14 +/- 0.10%`
- Original split macro-F1: `99.12 +/- 0.09%`
- Duplicate-clean accuracy: `98.27 +/- 0.07%`
- Duplicate-clean macro-F1: `98.28 +/- 0.04%`
- Duplicate-disjoint cross-source accuracy: `69.53 +/- 1.98%`
- Duplicate-disjoint external sample count: `373`

BRAIN-META reports `97.10%` accuracy and `96.94%` F1. This is a contextual comparison only. It is not a controlled head-to-head comparison because its split and pipeline differ.

The target-source logistic stack is supervised adaptation. Its result must never be reported as FusionV2 performance, zero-shot external validation, or domain generalization.

## Evidence rules

1. Treat prompts, comments, Markdown prose, and filenames as claims—not evidence.
2. Derive numerical results from prediction CSVs or other machine-readable artifacts.
3. Record the exact input files behind every reported number.
4. Validate filename and label alignment before combining predictions.
5. Validate probability columns, ranges, sums, and class ordering.
6. Do not invent missing metadata, patient identifiers, citations, checkpoints, masks, or results.
7. If evidence is missing, report `not available` and stop that analysis.
8. Never silently substitute another dataset, split, checkpoint, seed, or metric.
9. Never select a model, threshold, hyperparameter, or ensemble weight using final test labels.
10. Never claim statistical superiority over BRAIN-META without its per-image predictions or a controlled same-split reproduction.

## Preservation rules

Existing artifacts are immutable. Do not delete, rename, move, overwrite, or regenerate anything under:

- `runs/`
- `runs_original/`
- `runs_clean/`
- `runs_v2_original/`
- `runs_v2_clean/`
- `results/`
- `results_original/`
- `results_clean/`
- `results_v2_original/`
- `results_v2_clean/`
- dataset directories

New outputs must go only to:

- `publication_results/`
- `publication_runs/`

New publication code must normally go under `src/publication/`.

Do not alter defaults or behavior in existing training, evaluation, model, diagnostic, or Streamlit scripts. If a minimal compatibility change is unavoidable:

1. stop before editing;
2. explain why a wrapper/new script cannot solve it;
3. identify the exact lines proposed;
4. add a backward-compatibility test;
5. wait for explicit approval.

## Implementation rules

1. Prefer a new wrapper or analysis script over modification of working code.
2. Every new script must accept project-root-aware paths; do not assume the current directory.
3. Every experiment must save its full configuration, seed, environment fingerprint, inputs, and outputs.
4. Refuse to overwrite an existing output unless an explicitly designed `--resume` mode verifies compatibility.
5. Use fixed seeds and deterministic evaluation.
6. Keep training/validation/test roles explicit in variable names and saved metadata.
7. Use the same preprocessing and split for controlled comparisons unless the experiment explicitly studies preprocessing.
8. Do not tune on the duplicate-disjoint external test subset.
9. Generate tables programmatically from JSON/CSV; do not manually type results.
10. Store figures in both PNG and PDF when the implementation plan requests publication figures.
11. Add tests for schema, shapes, split membership, and backward compatibility.
12. Run a cheap smoke test before any full experiment.
13. For a multi-run GPU grid, produce and review a dry-run matrix first. Do not start the grid without explicit approval.

## Mandatory pre-change report

Before changing anything, print a concise report with this exact structure:

```text
PHASE:
EVIDENCE READ:
FILES TO CREATE:
EXISTING FILES TO MODIFY: none
COMMANDS TO RUN:
EXPECTED OUTPUTS:
ACCEPTANCE CHECKS:
STOP CONDITIONS:
```

If `EXISTING FILES TO MODIFY` is not `none`, stop and request approval.

## Mandatory post-change report

At the end of the phase, report:

```text
FILES CREATED:
FILES MODIFIED:
COMMANDS RUN:
TEST RESULTS:
REPRODUCED BASELINE RESULTS:
NEW RESULTS:
WARNINGS/LIMITATIONS:
ACCEPTANCE STATUS: PASS or FAIL
NEXT PHASE: not started
```

Do not describe the phase as complete if any acceptance check failed.

## Stop conditions

Stop rather than improvise if any of these occurs:

- a required input is absent;
- multiple plausible datasets or checkpoints exist and the correct one is not documented;
- class order or filenames do not align;
- recomputed baseline metrics differ by more than 0.01 percentage points from verified summaries;
- a script would overwrite an existing artifact;
- patient/study grouping cannot be supported by metadata;
- a requested analysis needs labels from the final test set for tuning;
- a result appears implausibly high and has not passed leakage and alignment checks;
- the repository has overlapping user changes in a file that would need editing;
- an experiment requires a materially different scientific assumption.

## Claims that are allowed

- FusionV2 uses gated, multi-scale CNN-ViT feature fusion with auxiliary branch supervision.
- FusionV2 achieves the verified multi-seed conventional and duplicate-clean results.
- Duplicate-disjoint evaluation reveals a substantial cross-source degradation.
- Gate mass measures branch utilization, not causal or anatomical explanation.
- BRAIN-META comparison is contextual unless reproduced on an identical split.

## Claims that are prohibited without new evidence

- FusionV2 is clinically ready.
- All data leakage has been eliminated.
- The split is patient-independent.
- FusionV2 universally generalizes across hospitals or scanners.
- FusionV2 statistically outperforms BRAIN-META.
- Nonzero gate mass proves that both branches improve predictions.
- Grad-CAM proves clinically correct reasoning.
- Supervised target calibration is zero-shot generalization.

## Completion standard

Correctness takes priority over producing a result. A documented negative result or a clear `not available` conclusion is preferable to an unsupported positive claim.
