# Visual QA report

## Files inspected

- `FUSION_NETWORK_MANUSCRIPT.docx`, rendered through Microsoft Word to `FUSION_NETWORK_MANUSCRIPT.pdf`
- `supplement/FUSION_NETWORK_SUPPLEMENT.docx`, rendered through Microsoft Word to `supplement/FUSION_NETWORK_SUPPLEMENT.pdf`
- Main manuscript: 16 pages
- Supplement: 5 pages

## Inspection method

Every final PDF page was rasterised at 110 dpi. Contact sheets were inspected for pagination, clipping, overflow, blank pages, figure placement, table splitting and obvious font substitution. Full-resolution checks were then made on the title/abstract page, architecture page, internal and cross-source ablation pages, Grad-CAM++ page, comparison table and reference page. The architecture figure was revised once because its explanatory text and parameter note crowded the plotted boxes; the corrected render was re-inspected.

## Result

- No blank, truncated or visibly overlapping body pages.
- Eleven figures are legible and placed beside the corresponding discussion.
- Wide ablation and comparison tables fit within the text area.
- The four-class Grad-CAM++ audit is visible as a two-by-two panel and its non-localisation limitation appears in the caption and prose.
- References fit on one page and the final PDF contains 11 clickable link annotations.
- Both PDFs contain searchable text.
- Author-input placeholders remain intentionally visible.

## Known formatting limits

The supplied article is a publisher-rendered PDF, not an editable journal template. The local system does not contain the PDF's embedded Charis SIL font, so the editable draft uses Times New Roman. Exact publisher typesetting must be applied after a target journal and its official template are chosen.

## Supporting QA files

- `qa/manuscript_contact_sheet.png`
- `qa/supplement_contact_sheet.png`
- `qa/PACKAGE_VALIDATION.json`

Status: **PASS**
