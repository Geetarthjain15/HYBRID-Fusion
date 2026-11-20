# Template fidelity report

Base article: Napa KK, Murugan S, Murugan JS, Jayanthi A. *BRAIN-META: a reproducible CNN–vision transformer meta-ensemble pipeline for explainable brain tumor classification.* MethodsX 2026;16:103769. https://doi.org/10.1016/j.mex.2025.103769

---

## 1. Source availability — read this first

The base article was supplied **as a conversation attachment, not as a file on disk**. A filesystem-wide search of the project for the PDF returned nothing.

Two consequences follow, and neither can be worked around in this environment:

| Required by the brief | Status | Reason |
|---|---|---|
| Record the base PDF's SHA-256 hash | **NOT AVAILABLE** | No PDF file exists on disk to hash. A digest of text I received in conversation would not be a digest of the published file and would be misleading, so none is given. |
| Record page count | **18 pages** | Counted from the supplied document content, which is paginated 1–18 and ends with the reference list. |
| Inspect PDF font metadata and embedded font names | **NOT POSSIBLE** | Requires the binary. No PDF parser is installed (`PyMuPDF`/`fitz`, `pdfminer`, `pypdf` all absent) and there is no network access to add one. |
| Measure page dimensions and margins from the PDF | **NOT POSSIBLE** | Same reason. Values below are inferred from the article's text structure and from Elsevier's standard single-column research-article layout, and are labelled as inferences. |

Everything in section 3 that is marked *inferred* is a reasoned approximation from the text I could read, not a measurement. I have not described any of it as an exact match.

---

## 2. Section topology recovered from the article text

The base article follows the MethodsX structure. This was recoverable with confidence because the headings appear in the supplied text.

| # | Base-article section | Present in AGMF-Net manuscript | Mapping note |
|---|---|---|---|
| 1 | Title, authors, affiliations | yes | authors/affiliations left as explicit placeholders |
| 2 | Highlights | yes | five bullets, matching the base article's format |
| 3 | Graphical abstract | yes | proposed as Figures 2 + 3; no composite image was fabricated |
| 4 | Keywords | yes | |
| 5 | Abstract | yes | longer than the base article's, because the evaluation design carries part of the contribution |
| 6 | Specifications table | yes | same field names as the base article, including "Name and reference of original method" |
| 7 | Background | yes | |
| 8 | Method details | yes | contains architecture motivation, architecture, dataset, protocol, statistics |
| 8a | Related works | **merged into Background** | see deviation D1 |
| 8b | Problem statement and research contributions | **merged into Background and Highlights** | see deviation D1 |
| 9 | Method validation | yes | contains internal, cross-source, ablation, gate, calibration, explainability |
| 9a | Electronic health records analysis | **renamed** "Dataset and duplicate control" | see deviation D2 |
| 9b | Deep learning models | yes, as "AGMF-Net architecture" | |
| 9c | Explainable artificial intelligence | yes, as "Explainability demonstrator" | |
| 9d | Computation time analysis | **present but unpopulated** | see deviation D3 |
| 9e | Statistical approach for model comparisons | yes, as "Statistical analysis" | |
| 9f | Model evaluation | yes, distributed across the results subsections | |
| 10 | Model explanation / Grad-CAM | yes | scope explicitly narrowed; see the manuscript |
| 11 | Developing eXplainable UI | **omitted** | see deviation D4 |
| 12 | Theoretical and methodological implications | yes | |
| 13 | Impact on future research | **merged into Conclusion and future work** | |
| 14 | Limitations | yes, as "Assumptions, limitations and threats to validity" | expanded with a status column |
| 15 | Ethics statements | yes, placeholder | |
| 16 | Related research article | **omitted** | base article states "None"; nothing to carry over |
| 17 | CRediT author statement | yes, placeholder | |
| 18 | Declaration of generative AI | yes, placeholder | |
| 19 | Declaration of competing interest | yes, placeholder | |
| 20 | Acknowledgement | yes, placeholder | |
| 21 | Data availability | yes, placeholder | |
| 22 | References | yes | numbered, first-appearance order |

### Deviations, with reasons

**D1 — Related works and Problem statement folded into Background.** The base article devotes roughly a page to a survey of 2015–2025 hybrid CNN–ViT work. Reproducing a comparable survey would require citing 40+ papers I cannot verify in an offline environment, and the brief forbids citing work not inspected sufficiently to support the claim. I wrote a shorter, specific Background that motivates the two problems this paper actually addresses, and cited only sources I could verify. This is a deliberate reduction in scope, not an oversight. **If the target venue requires a full related-work section, it must be written by the authors with verified citations.**

**D2 — "Electronic health records analysis" renamed.** The base article uses this heading for what is a dataset description; no electronic health records are involved in either study. Retaining the heading would be inaccurate, so the section was renamed to describe its contents.

**D3 — Computation time analysis present but unpopulated.** The base article reports per-model training and testing seconds. Our training timings were not retained in a form verifiable from the artifact package. Rather than estimate them, the section carries an explicit `[AUTHOR INPUT REQUIRED]` and states why. Parameter counts, which *are* verified, are given.

**D4 — "Developing eXplainable UI" omitted from the manuscript.** A Streamlit demonstrator exists in the project, but describing its interface as a contribution would add a section whose claims I could not evidence at the level the rest of the paper is held to, and the brief directs that the application be described as a research demonstrator rather than a clinical tool. The Grad-CAM++ demonstrator is described in the Method validation section instead. Authors who want a UI section should add it with screenshots they can vouch for.

---

## 3. Visual specification

| Property | Base article (inferred) | AGMF-Net DOCX (actual) | Match |
|---|---|---|---|
| Page size | A4 portrait | A4 portrait, 11906 × 16838 twips | assumed match |
| Orientation | portrait | portrait | match |
| Margins | ~2.0 cm, Elsevier single-column standard | 2.0 cm all round (1134 twips) | approximation |
| Body layout | single column for the body; first page has a two-block Highlights / Graphical-abstract and Article-info / Abstract arrangement | single column throughout | **deviation** — see D5 |
| Body font | a Times-family serif (Elsevier house style) | Times New Roman | **substitution** — see section 4 |
| Body size | ~9–10 pt | 10 pt | approximation |
| Line spacing | single | single | match |
| Heading hierarchy | unnumbered, bold, size-graded | unnumbered, bold, 14 / 12 / 10.5 / 10 pt | match in kind |
| Table style | horizontal rules only, no vertical rules, shaded header absent | horizontal rules only; light shaded header row | minor deviation, noted |
| Table caption | above the table ("Table 1 / caption") | above the table | match |
| Figure caption | below the figure ("Fig. 1. caption") | below the figure, centred, 8.5 pt | match in kind |
| Citation style | numbered, square brackets, first-appearance order | numbered, first-appearance order | match |
| Page numbers | footer, centred | **not set** | deviation, see D6 |
| Running header | journal name and citation | **not set** | deviation, see D6 |

**D5 — first-page two-block layout not reproduced.** The base article's opening page places Highlights beside a Graphical abstract, and Article info beside the Abstract. Reproducing that needs either a two-column section break with balanced frames or a table-based layout, both of which render inconsistently across Word versions. The manuscript presents those four blocks sequentially instead. The content is complete and in the correct order; only the side-by-side arrangement differs. Publishers typeset from the submitted text, so this does not affect submission.

**D6 — no running header or page numbers.** These are supplied by the publisher's template at typesetting. Adding an approximation of Elsevier's header would put a journal name and a fabricated citation line on a manuscript that has no accepted venue, which would be worse than omitting it.

---

## 4. Font decision

The brief permits using the base article's exact font only if its identity is verifiable from the PDF, it is available in this environment, and the output actually renders with it. **None of the three could be established**, because the PDF binary is not present and no font-inspection library is installed.

Accordingly:

- **Font used:** Times New Roman for body and headings; Consolas for inline code.
- **Basis:** Elsevier research articles are typeset in a Times-family serif. Times New Roman is metrically compatible with that family and is present on the target platform.
- **Status:** this is a **documented substitution**, not a verified match. No claim of exact font matching is made anywhere in this package.

---

## 5. Word file construction

`python-docx` is not installed and there is no network access to add it. The DOCX is therefore written as raw Office Open XML inside a zip container by `src/publication/make_paper_docx.py` — the same file format, with no third-party dependency.

Validation performed on the emitted file:

| Check | Result |
|---|---|
| Zip integrity (`testzip`) | OK |
| `word/document.xml` parses as XML | OK, 180,417 bytes |
| `word/styles.xml` parses as XML | OK |
| `word/_rels/document.xml.rels` parses as XML | OK |
| `[Content_Types].xml` parses as XML | OK |
| Paragraphs emitted | 336 |
| Tables emitted | 5 |
| Figures embedded | 8, each with a matching media part and relationship |

---

## 6. What a human must still check

Because no page renderer is available here (see `VISUAL_QA_REPORT.md`), the following can only be confirmed by opening the file in Word:

1. Table column widths at A4 — the wide comparison table (Table 3) is the most likely to need adjustment.
2. Figure scaling and whether any figure lands awkwardly across a page break.
3. Whether headings orphan at page bottoms.
4. Whether the equation approximations in the architecture section read acceptably; they are rendered as italic plain text, not as Word equation objects.
5. Font rendering, if the target machine lacks Times New Roman.
