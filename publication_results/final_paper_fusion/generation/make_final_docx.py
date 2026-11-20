"""
Build FUSION_NETWORK_MANUSCRIPT.docx from FUSION_NETWORK_MANUSCRIPT.md.

python-docx is not installed in this environment and there is no network access
to add it, so the file is written as raw Office Open XML inside a zip container.
That is the same format python-docx emits; it simply keeps the dependency at
zero. The writer covers exactly what this manuscript needs - headings,
paragraphs, bold and italic runs, inline code, bullet and numbered lists,
block quotes, pipe tables with a header row, horizontal rules and embedded PNG
figures - and nothing else.

Layout follows the inspected base article as closely as this environment allows;
every deviation is recorded in TEMPLATE_FIDELITY_REPORT.md rather than being
presented as a match.

    env_win/Scripts/python.exe src/publication/make_paper_docx.py
"""

from __future__ import annotations

import os
import re
import struct
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "publication_results", "final_paper_fusion")
FIGDIR = os.path.join(OUT, "figures")

# ---- page geometry, in twentieths of a point (twips) ------------------------
# A4 portrait with 2.0 cm side margins, matching the single-column body of the
# inspected article as closely as measurement from rendered text permits.
PAGE_W, PAGE_H = 11906, 16838          # A4
MARGIN = 1134                          # 2.0 cm
CONTENT_W_EMU = int((PAGE_W - 2 * MARGIN) / 20 * 12700)   # twips -> EMU

BODY_FONT = "Times New Roman"          # metrically compatible substitute; see fidelity report
SANS_FONT = "Arial"
BODY_SZ = 20                           # half-points -> 10 pt
SMALL_SZ = 17                          # 8.5 pt


def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(26)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG: %s" % path)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


# ---- inline formatting -----------------------------------------------------

TOKEN = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`|\$\$?[^$]+?\$\$?)", re.S)


def runs(text: str, sz=BODY_SZ, base_font=BODY_FONT, color=None):
    """Turn a line of markdown into a sequence of w:r elements."""
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        bold = italic = False
        font = base_font
        body = part
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            bold, body = True, part[2:-2]
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            italic, body = True, part[1:-1]
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            font, body = "Consolas", part[1:-1]
        elif part.startswith("$"):
            italic = True
            body = part.strip("$").strip()
            body = (body.replace("\\mathrm", "").replace("\\mathbb{R}", "R")
                        .replace("\\times", "x").replace("\\in", " in ")
                        .replace("\\big", "").replace("\\Big", "")
                        .replace("\\left", "").replace("\\right", "")
                        .replace("\\;", " ").replace("\\,", " ")
                        .replace("\\qquad", "    ").replace("\\lambda", "lambda")
                        .replace("\\phi", "phi").replace("\\mathbf", "")
                        .replace("\\mathcal", "").replace("\\hat", "")
                        .replace("\\text", "").replace("_{", "_").replace("^{", "^")
                        .replace("{", "").replace("}", ""))
        rpr = ['<w:rFonts w:ascii="%s" w:hAnsi="%s"/>' % (font, font),
               '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (sz, sz)]
        if bold:
            rpr.append("<w:b/>")
        if italic:
            rpr.append("<w:i/>")
        if color:
            rpr.append('<w:color w:val="%s"/>' % color)
        for piece in re.split(r"(https?://[^\s]+)", body):
            if not piece:
                continue
            if piece.startswith(("http://", "https://")):
                url = piece.rstrip(".,;)")
                trail = piece[len(url):]
                link_rpr = list(rpr) + ['<w:color w:val="0563C1"/>', '<w:u w:val="single"/>']
                out.append('<w:fldSimple w:instr="HYPERLINK &quot;%s&quot;">'
                           '<w:r><w:rPr>%s</w:rPr><w:t>%s</w:t></w:r></w:fldSimple>'
                           % (esc(url), "".join(link_rpr), esc(url)))
                if trail:
                    out.append("<w:r><w:rPr>%s</w:rPr><w:t>%s</w:t></w:r>"
                               % ("".join(rpr), esc(trail)))
            else:
                out.append("<w:r><w:rPr>%s</w:rPr><w:t xml:space=\"preserve\">%s</w:t></w:r>"
                           % ("".join(rpr), esc(piece)))
    return "".join(out) or "<w:r></w:r>"


def para(text="", sz=BODY_SZ, bold=False, align=None, before=0, after=100,
         ind=0, font=BODY_FONT, color=None, line=240, keep_next=False):
    ppr = ['<w:spacing w:before="%d" w:after="%d" w:line="%d" w:lineRule="auto"/>'
           % (before, after, line)]
    if align:
        ppr.append('<w:jc w:val="%s"/>' % align)
    if ind:
        ppr.append('<w:ind w:left="%d"/>' % ind)
    if keep_next:
        ppr.append("<w:keepNext/>")
    body = ("<w:r><w:rPr><w:rFonts w:ascii=\"%s\" w:hAnsi=\"%s\"/><w:b/>"
            "<w:sz w:val=\"%d\"/><w:szCs w:val=\"%d\"/>%s</w:rPr>"
            "<w:t xml:space=\"preserve\">%s</w:t></w:r>"
            % (font, font, sz, sz,
               '<w:color w:val="%s"/>' % color if color else "", esc(text))
            ) if bold else runs(text, sz=sz, base_font=font, color=color)
    return "<w:p><w:pPr>%s</w:pPr>%s</w:p>" % ("".join(ppr), body)


def heading(text, level):
    sizes = {1: 28, 2: 24, 3: 21, 4: 20}
    return para(text, sz=sizes.get(level, 20), bold=True,
                before=260 if level <= 2 else 200, after=110,
                font=BODY_FONT, keep_next=True)


def hrule():
    return ('<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="6" w:space="1" '
            'w:color="BFBFBF"/></w:pBdr><w:spacing w:before="140" w:after="140"/>'
            '</w:pPr></w:p>')


def bullet(text, num=False, sz=BODY_SZ):
    mark = "" if num else "•  "
    return para(mark + text, sz=sz, ind=340, after=60)


def table(rows, widths=None):
    """rows[0] is the header. Widths are relative integers."""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    total = PAGE_W - 2 * MARGIN
    if not widths:
        widths = [1] * ncol
    wsum = float(sum(widths))
    cw = [int(total * w / wsum) for w in widths]

    def cell(txt, header, last_col):
        shade = '<w:shd w:val="clear" w:fill="EFF3F6"/>' if header else ""
        al = "right" if (not header and last_col and
                         re.fullmatch(r"[-+()%.,0-9 /±−]+", txt.strip() or "x")) else "left"
        p = para(txt, sz=SMALL_SZ, bold=header, align=al if al != "left" else None, after=0,
                 before=0, line=220)
        return ("<w:tc><w:tcPr><w:tcW w:w=\"%d\" w:type=\"dxa\"/>%s"
                "<w:tcMar><w:top w:w=\"50\" w:type=\"dxa\"/><w:bottom w:w=\"50\" w:type=\"dxa\"/>"
                "<w:left w:w=\"80\" w:type=\"dxa\"/><w:right w:w=\"80\" w:type=\"dxa\"/></w:tcMar>"
                "</w:tcPr>%s</w:tc>")

    out = ['<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/>'
           '<w:tblBorders>'
           '<w:top w:val="single" w:sz="8" w:color="7A7A7A"/>'
           '<w:bottom w:val="single" w:sz="8" w:color="7A7A7A"/>'
           '<w:insideH w:val="single" w:sz="2" w:color="D6DBDF"/>'
           '</w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr>' % total]
    for i, r in enumerate(rows):
        hdr = i == 0
        out.append('<w:tr>%s' % ('<w:trPr><w:tblHeader/></w:trPr>' if hdr else ""))
        for j, c in enumerate(r):
            shade = '<w:shd w:val="clear" w:fill="EFF3F6"/>' if hdr else ""
            al = None
            if not hdr and j > 0 and re.fullmatch(r"[-+()%.,0-9 /±−]*", c.strip()):
                al = "right"
            p = para(c, sz=SMALL_SZ, bold=hdr, align=al, after=0, before=0, line=220)
            out.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s'
                       '<w:tcMar><w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/>'
                       '<w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar>'
                       '</w:tcPr>%s</w:tc>' % (cw[j], shade, p))
        out.append("</w:tr>")
    out.append("</w:tbl>")
    out.append(para("", after=90))
    return "".join(out)


def image_par(rid, path, max_w_frac=1.0):
    w, h = png_size(path)
    cw = int(CONTENT_W_EMU * max_w_frac)
    ch = int(cw * h / w)
    return ('<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="140" w:after="60"/>'
            '<w:keepNext/></w:pPr><w:r><w:drawing>'
            '<wp:inline distT="0" distB="0" distL="0" distR="0">'
            '<wp:extent cx="%d" cy="%d"/><wp:docPr id="%d" name="Figure %d"/>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:nvPicPr><pic:cNvPr id="%d" name="Figure %d"/><pic:cNvPicPr/></pic:nvPicPr>'
            '<pic:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
            % (cw, ch, 100, 100, 100, 100, rid, cw, ch))


# ---- markdown -> docx ------------------------------------------------------

FIGURE_ANCHORS = [
    ("## Method details", None),
]

FIG_FOR_SECTION = {
    "### Multi-scale EfficientNet-B0–ViT-B/16 fusion architecture": ("figure_01_architecture",
                                  "Figure 1. Multi-scale EfficientNet-B0–ViT-B/16 feature-fusion path. Both branches are projected to a "
                                  "common 512-dimensional width before the gate combines them. Layer "
                                  "widths are read from the verified checkpoints."),
    "### Dataset and duplicate control": ("figure_02_evaluation_map",
                                          "Figure 2. Which images each reported number is computed on. "
                                          "Duplicate cleaning operates on image similarity and does not "
                                          "establish a patient-independent split."),
    "### Internal performance": ("figure_03_regime_collapse",
                                 "Figure 3. Accuracy and macro-F1 by evaluation regime. Open circles are "
                                 "the three individual seeds, the diamond is their mean and the bar is one "
                                 "sample standard deviation across seeds, which is not a confidence interval."),
    "### Cross-source performance": ("figure_04_confusion_matrices",
                                     "Figure 4. Confusion matrices summed over three seeds, row-normalised "
                                     "to recall with the raw count printed beneath each percentage."),
    "### Component ablation": ("figure_05_ablation_forest",
                               "Figure 5. Observed accuracy difference for each ablation against the full "
                               "model on the same 848 images. Thin lines are paired image-level 95% "
                               "bootstrap intervals, one per seed; the diamond is the mean observed "
                               "difference. Holm correction is across all 21 seed-specific tests."),
    "### Cross-source ablation": ("figure_09_cross_source_ablation_accuracy",
                                  "Figure 9. Cross-source accuracy of the full network and seven protocol-matched variants on the same 373 duplicate-disjoint images. Bars show the mean over three seeds and points show individual seeds."),
    "### Gate behaviour": ("figure_07_gate_behaviour",
                           "Figure 7. Distribution of CNN gate mass over the duplicate-clean test set. The "
                           "spread is the evidence that the gate is input-dependent; gate mass measures "
                           "branch utilisation and is not causal attribution."),
    "### Calibration": ("figure_06_calibration",
                        "Figure 6. Reliability curves for both regimes with bin populations beneath. "
                        "These comparisons are descriptive and exploratory only."),
    "### Explainability demonstrator": ("figure_11_gradcam_audit",
                                         "Figure 11. Deterministically selected Grad-CAM++ audit examples, one per true class. Maps show sensitivity through the final spatial EfficientNet stage only and are not validated tumour localisations."),
}
EXTRA_AFTER = {
    "### Cross-source performance": ("figure_08_class_failure",
                                     "Figure 8. The cross-source failure is class-specific. No-tumour recall "
                                     "is unchanged while pituitary recall falls to 13.81%; 260 of the 333 "
                                     "pituitary predictions are assigned to no-tumour. These are repeated "
                                     "predictions of the same 111 images."),
    "### Cross-source ablation": ("figure_10_pituitary_recall_by_ablation",
                                   "Figure 10. Pituitary recall by model variant on the duplicate-disjoint cross-source subset. None of the tested variants resolves the class-specific failure."),
}


def build():
    md = open(os.path.join(OUT, "FUSION_NETWORK_MANUSCRIPT.md"), encoding="utf-8").read()
    lines = md.split("\n")

    body, images = [], []          # images: list of (rid, filename)
    rid_n = [0]

    def add_figure(stem, caption):
        png = os.path.join(FIGDIR, stem + ".png")
        if not os.path.exists(png):
            return
        rid_n[0] += 1
        rid = "rId%d" % (100 + rid_n[0])
        images.append((rid, stem + ".png", png))
        body.append(image_par(rid, png))
        body.append(para(caption, sz=SMALL_SZ, align="center", after=180))

    i, cur_h3 = 0, None
    while i < len(lines):
        ln = lines[i].rstrip()

        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = lines[i].strip()
                if not re.match(r"^\|[\s:|-]+\|$", r):
                    cells = [c.strip() for c in r.strip("|").split("|")]
                    rows.append(cells)
                i += 1
            ncol = max(len(r) for r in rows)
            w = [2] + [1] * (ncol - 1) if ncol > 2 else [1] * ncol
            body.append(table(rows, w))
            continue

        if ln.startswith("# "):
            body.append(para(ln[2:], sz=32, bold=True, before=0, after=180))
        elif ln.startswith("## "):
            body.append(heading(ln[3:], 2)); cur_h3 = None
        elif ln.startswith("### "):
            cur_h3 = ln
            body.append(heading(ln[4:], 3))
            if cur_h3 in FIG_FOR_SECTION:
                pass   # figure is placed after the section's prose, below
        elif ln.startswith("#### "):
            body.append(heading(ln[5:], 4))
        elif ln.strip() == "---":
            body.append(hrule())
        elif ln.startswith("> "):
            body.append(para(ln[2:], sz=SMALL_SZ, ind=280, before=80, after=120,
                             color="4A4A4A"))
        elif re.match(r"^\s*[-*]\s+", ln):
            body.append(bullet(re.sub(r"^\s*[-*]\s+", "", ln)))
        elif re.match(r"^\s*\d+\.\s+", ln):
            body.append(bullet(ln.strip(), num=True))
        elif ln.startswith("$$"):
            eq = ln.strip("$ ")
            j = i
            while not eq and j + 1 < len(lines):
                j += 1
                eq = lines[j].strip("$ ")
            body.append(para("$" + eq + "$", align="center", before=80, after=100))
            i = j
        elif ln.strip() == "":
            pass
        else:
            body.append(para(ln))

        # place the section figure when the section's text ends
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if cur_h3 and (nxt.startswith("### ") or nxt.startswith("## ")):
            if cur_h3 in FIG_FOR_SECTION:
                add_figure(*FIG_FOR_SECTION[cur_h3])
            if cur_h3 in EXTRA_AFTER:
                add_figure(*EXTRA_AFTER[cur_h3])
            cur_h3 = None
        i += 1

    sect = ('<w:sectPr><w:pgSz w:w="%d" w:h="%d"/>'
            '<w:pgMar w:top="%d" w:right="%d" w:bottom="%d" w:left="%d" '
            'w:header="708" w:footer="708" w:gutter="0"/>'
            '<w:docGrid w:linePitch="360"/></w:sectPr>'
            % (PAGE_W, PAGE_H, MARGIN, MARGIN, MARGIN, MARGIN))

    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
           'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
           'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
           '<w:body>%s%s</w:body></w:document>' % ("".join(body), sect))

    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/'
            '2006/relationships/styles" Target="styles.xml"/>']
    for rid, name, _p in images:
        rels.append('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/officeDocument/'
                    '2006/relationships/image" Target="media/%s"/>' % (rid, name))
    rels.append("</Relationships>")

    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:docDefaults><w:rPrDefault><w:rPr>'
              '<w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
              '<w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr></w:rPrDefault>'
              '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="240" w:lineRule="auto"/>'
              '</w:pPr></w:pPrDefault></w:docDefaults>'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
              '<w:name w:val="Normal"/><w:qFormat/></w:style></w:styles>'
              % (BODY_FONT, BODY_FONT, BODY_FONT, BODY_FONT, BODY_SZ, BODY_SZ))

    ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
          '<Default Extension="xml" ContentType="application/xml"/>',
          '<Default Extension="png" ContentType="image/png"/>',
          '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
          'officedocument.wordprocessingml.document.main+xml"/>',
          '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-'
          'officedocument.wordprocessingml.styles+xml"/>',
          '</Types>']

    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
                 '2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')

    path = os.path.join(OUT, "FUSION_NETWORK_MANUSCRIPT.docx")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", doc)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/_rels/document.xml.rels", "".join(rels))
        for _rid, name, p in images:
            z.write(p, "word/media/%s" % name)
    print("  wrote %s (%.0f KB, %d figures embedded)"
          % (os.path.relpath(path, ROOT), os.path.getsize(path) / 1024, len(images)))
    return path


if __name__ == "__main__":
    build()
    sys.exit(0)
