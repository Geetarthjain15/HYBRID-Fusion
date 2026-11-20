# Visual QA report

## Summary

**The page-level visual QA required by the brief was NOT performed, and the PDF deliverable was NOT produced.** Both depend on rendering, and this environment has no renderer. Details and the exact remaining steps are below. Figure-level QA *was* performed and is reported in section 3.

I am stating this plainly rather than producing a PDF by some other route, because the brief specifies that the PDF be exported from a visually approved Word version and that Word and PDF must not diverge. A PDF generated independently from Markdown would satisfy the file list while violating the requirement it exists to serve.

---

## 1. Why no PDF and no page rendering

| Capability needed | Tool | Status in `env_win` |
|---|---|---|
| DOCX → PDF conversion | `docx2pdf` | not installed |
| DOCX → PDF conversion | Microsoft Word automation | Word is installed on this machine but is not reachable from this shell |
| DOCX/PDF → PDF conversion | LibreOffice `soffice` | not on PATH |
| PDF → page images | `PyMuPDF` / `fitz` | not installed |
| PDF → page images | `pdftoppm` (poppler) | not on PATH |
| Direct PDF authoring | `reportlab` | not installed |
| HTML → PDF | `wkhtmltopdf`, `weasyprint` | not installed |

There is no network access to install any of these.

Corroborating evidence that this step has always been a human one in this project: the pre-existing `publication_results/manuscript_draft/AGMF_Net_Manuscript.pdf` carries `/Producer: Microsoft® Word 2019` in its metadata. That earlier PDF was exported by a person opening the DOCX in Word — exactly the workflow the brief prescribes.

---

## 2. Steps for a human to complete the deliverable

1. Open `publication_results/final_paper_claude/AGMF_NET_MANUSCRIPT.docx` in Word.
2. Inspect every page for: table overflow past the right margin; figures clipped or pushed to a page of their own; captions separated from their figure; headings orphaned at a page bottom; blank pages; reference lines wrapping badly.
3. Correct any layout defect **in Word**, not in the Markdown, or correct it in the Markdown and regenerate so that the two stay in agreement.
4. Export to PDF from the approved Word document (File → Save as → PDF).
5. Re-inspect the exported PDF.

Section 6 of `TEMPLATE_FIDELITY_REPORT.md` lists the five specific risks most worth looking at first.

---

## 3. Figure-level QA that *was* performed

All eight figures were generated at 400 dpi, exported as both PNG and PDF, and inspected as images. Three defects were found and fixed; they are listed here rather than omitted.

| Figure | Inspected | Defect found | Resolution |
|---|---|---|---|
| 01 architecture | yes | none | — |
| 02 evaluation map | yes | none | — |
| 03 regime collapse | yes | the "drop under source shift" annotation arrow crossed the plot and collided with the 76.49 value label | annotation moved to a bracket in the left margin, clear of all value labels; x-limit tightened |
| 04 confusion matrices | yes | none | — |
| 05 ablation forest | yes | the CNN-only bootstrap intervals were clipped at the left axis limit; a dead gap sat between the plot and the value column | x-limit widened to −4.7; value and Holm columns moved inward; gridlines masked behind the text column |
| 06 calibration | yes | none | — |
| 07 gate behaviour | yes | none | — |
| 08 class failure | yes | (a) a double-negative label, "−-2.0 pp", where no-tumour recall *rose* under shift; (b) segment labels clipped inside narrow bars; (c) large vertical dead space | (a) signed formatting, so an improvement now reads "+2.01 pp"; (b) narrow segments label outside with a leader line; (c) panel height reduced and bar given full height |

### Figure design checks applied

- **Colour by job.** Magnitude (confusion cells, calibration density) uses one hue, light to dark. Identity (evaluation regime, ablation verdict) uses a fixed categorical order that is never cycled. No rainbow scale is used anywhere.
- **No dual axes.** No panel carries two y-scales.
- **Identity is never colour-alone.** Every coloured mark is accompanied by a text label, a legend entry, or both.
- **Uncertainty is drawn, not implied.** Ablation effects are points with paired intervals rather than bars, because a bar reads as a magnitude from zero and hides the interval that decides whether an effect is resolved.
- **The two uncertainty kinds are kept visually distinct.** Seed spread and bootstrap intervals never share a mark, and both are named in the caption or legend.
- **Sample counts are on the figure.** Every panel that reports a metric states the number of images behind it.
- **Reliability bins carry their populations.** Figure 6 plots bin counts beneath the curve, so a bin holding a handful of images cannot be mistaken for a trend.

### Not checked

- Figure legibility at final print scale in a typeset two-column layout. Figures were sized for a single-column body at A4 width. If the venue typesets in two columns, font sizes inside the figures should be re-checked.
- Colour-vision-deficiency simulation was **not** run; no simulation tool is installed. The palette was chosen with separated lightness steps and every series is redundantly labelled, which limits the exposure, but this is a mitigation rather than a verification.
- CMYK conversion for print was not performed.

---

## 4. Markdown / DOCX agreement

The DOCX is generated from `AGMF_NET_MANUSCRIPT.md` by a script, so the two cannot drift in substance. Differences are confined to presentation:

| Difference | Nature |
|---|---|
| Figures appear in the DOCX and are referenced by number in the Markdown | the Markdown is the editable source; figures live as separate files |
| LaTeX-style equations are rendered as italic plain text in the DOCX | the writer emits no Word equation objects; the expressions remain readable but are not editable as equations |
| Horizontal rules become bottom-bordered empty paragraphs | presentational only |
| Block quotes become indented grey paragraphs | presentational only |

No substantive text, number, table row, caveat or placeholder differs between the two.

---

## 5. Status

| Deliverable | Status |
|---|---|
| `AGMF_NET_MANUSCRIPT.md` | complete |
| `AGMF_NET_MANUSCRIPT.docx` | complete, structurally validated, **not visually inspected page by page** |
| `AGMF_NET_MANUSCRIPT.pdf` | **NOT PRODUCED** — requires Word, see section 2 |
| `supplement/AGMF_NET_SUPPLEMENT.md` | complete |
| `supplement/AGMF_NET_SUPPLEMENT.docx` | **NOT PRODUCED** — see note below |
| Figures (PNG + PDF) | complete, inspected, three defects fixed |
| Tables, ledger, reports, manifest | complete |

The supplement DOCX was not produced because the same Word-export step would be needed to verify it, and shipping an unverified second Word file adds risk without adding information. The supplement Markdown is complete and can be converted by the same script (`make_paper_docx.py`) with its input path changed, once a human is in a position to check the output.
