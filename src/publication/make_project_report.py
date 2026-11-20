"""
B.Tech mid-semester project report (Word).

Scope is exactly the four sections requested: Introduction, Literature Survey,
Problem Statement and Methodology, and References. Sections (d) Work done till
date and (e) Future work required by the evaluation guidelines are NOT included
here, because they were not requested; the note at the end of the run says so.

Formatting follows the usual departmental convention rather than a journal
style: A4, 1 inch margins, Times New Roman 12 pt body, 1.5 line spacing,
justified paragraphs, numbered headings, page numbers in the footer, and
references numbered by order of first appearance with working hyperlinks.

python-docx is not installed and there is no network access here, so the file is
written as raw Office Open XML in a zip container.

    env_win/Scripts/python.exe src/publication/make_project_report.py
"""

from __future__ import annotations

import os
import re
import struct
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "publication_results", "final_paper_claude")
FIG = os.path.join(OUT, "figures")

PAGE_W, PAGE_H = 11906, 16838      # A4 in twips
MARGIN = 1440                      # 1 inch
CONTENT_EMU = int((PAGE_W - 2 * MARGIN) / 20 * 12700)
FONT = "Times New Roman"
SZ = 24          # half-points -> 12 pt
SZ_SMALL = 20    # 10 pt
LINE = 360       # 1.5 line spacing


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def png_size(p):
    with open(p, "rb") as f:
        h = f.read(26)
    return struct.unpack(">II", h[16:24])


# --------------------------------------------------------------- runs -------

TOK = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*)", re.S)


def rtext(t, sz=SZ, bold=False, italic=False, font=FONT, color=None, sub=False, sup=False):
    rpr = ['<w:rFonts w:ascii="%s" w:hAnsi="%s"/>' % (font, font),
           '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (sz, sz)]
    if bold:
        rpr.append("<w:b/>")
    if italic:
        rpr.append("<w:i/>")
    if color:
        rpr.append('<w:color w:val="%s"/>' % color)
    if sub:
        rpr.append('<w:vertAlign w:val="subscript"/>')
    if sup:
        rpr.append('<w:vertAlign w:val="superscript"/>')
    return ('<w:r><w:rPr>%s</w:rPr><w:t xml:space="preserve">%s</w:t></w:r>'
            % ("".join(rpr), esc(t)))


def inline(text, sz=SZ):
    out = []
    for part in TOK.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            out.append(rtext(part[2:-2], sz=sz, bold=True))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            out.append(rtext(part[1:-1], sz=sz, italic=True))
        else:
            out.append(rtext(part, sz=sz))
    return "".join(out)


def p(text="", sz=SZ, bold=False, align="both", before=0, after=120, ind=0,
      first=0, line=LINE, keep=False, color=None):
    ppr = ['<w:spacing w:before="%d" w:after="%d" w:line="%d" w:lineRule="auto"/>'
           % (before, after, line), '<w:jc w:val="%s"/>' % align]
    if ind or first:
        ppr.append('<w:ind w:left="%d" w:firstLine="%d"/>' % (ind, first))
    if keep:
        ppr.append("<w:keepNext/>")
    body = rtext(text, sz=sz, bold=True, color=color) if bold else inline(text, sz=sz)
    return "<w:p><w:pPr>%s</w:pPr>%s</w:p>" % ("".join(ppr), body)


def h(text, level=1):
    sizes = {1: 28, 2: 26, 3: 24}
    return p(text, sz=sizes.get(level, 24), bold=True, align="left",
             before=280 if level == 1 else 220, after=140, keep=True)


def bullet(text, ind=720):
    return p("•  " + text, ind=ind, after=80, align="both")


def numbered(n, text, ind=720):
    return p("%s  %s" % (n, text), ind=ind, after=80, align="both")


def pagebreak():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def figure(rid, path, caption, frac=0.94):
    w, hh = png_size(path)
    cw = int(CONTENT_EMU * frac)
    ch = int(cw * hh / w)
    img = ('<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="200" w:after="80"/>'
           '<w:keepNext/></w:pPr><w:r><w:drawing>'
           '<wp:inline distT="0" distB="0" distL="0" distR="0">'
           '<wp:extent cx="%d" cy="%d"/><wp:docPr id="7" name="pic"/>'
           '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
           '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
           '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
           '<pic:nvPicPr><pic:cNvPr id="7" name="pic"/><pic:cNvPicPr/></pic:nvPicPr>'
           '<pic:blipFill><a:blip r:embed="%s"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
           '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
           '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
           '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
           % (cw, ch, rid, cw, ch))
    cap = p(caption, sz=SZ_SMALL, align="center", after=220, line=240)
    return img + cap


def table(rows, widths=None, sz=SZ_SMALL):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    total = PAGE_W - 2 * MARGIN
    widths = widths or [1] * ncol
    cw = [int(total * w / float(sum(widths))) for w in widths]
    out = ['<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:tblBorders>'
           '<w:top w:val="single" w:sz="8" w:color="000000"/>'
           '<w:bottom w:val="single" w:sz="8" w:color="000000"/>'
           '<w:left w:val="single" w:sz="4" w:color="808080"/>'
           '<w:right w:val="single" w:sz="4" w:color="808080"/>'
           '<w:insideH w:val="single" w:sz="4" w:color="808080"/>'
           '<w:insideV w:val="single" w:sz="4" w:color="808080"/>'
           '</w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr>' % total]
    for i, r in enumerate(rows):
        hdr = i == 0
        out.append('<w:tr>%s' % ('<w:trPr><w:tblHeader/></w:trPr>' if hdr else ""))
        for j, c in enumerate(r):
            shade = '<w:shd w:val="clear" w:fill="E8E8E8"/>' if hdr else ""
            al = "center" if hdr else ("center" if j > 0 and re.fullmatch(
                r"[-+()%.,0-9 /±−a-z ]*", c.strip()) and any(
                ch.isdigit() for ch in c) else "left")
            out.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/>%s'
                       '<w:vAlign w:val="center"/>'
                       '<w:tcMar><w:top w:w="70" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/>'
                       '<w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/>'
                       '</w:tcMar></w:tcPr>%s</w:tc>'
                       % (cw[j], shade,
                          p(c, sz=sz, bold=hdr, align=al, after=0, line=240)))
        out.append("</w:tr>")
    out.append("</w:tbl>")
    out.append(p("", after=120))
    return "".join(out)


# ----------------------------------------------------------- references -----
# Ordered by first appearance in the text. Only sources that were actually
# inspected in this project, or whose bibliographic details are printed in the
# reference list of the base article that was read in full.

REFS = [
    ("Napa KK, Murugan S, Murugan JS, Jayanthi A. BRAIN-META: a reproducible CNN-vision "
     "transformer meta-ensemble pipeline for explainable brain tumor classification. "
     "MethodsX. 2026;16:103769.", "https://doi.org/10.1016/j.mex.2025.103769"),
    ("Ali S, Li J, Pei Y, Khurram R, Rehman KU, Mahmood T. A comprehensive survey on brain "
     "tumor diagnosis using deep learning and emerging hybrid techniques with multi-modal "
     "MR image. Archives of Computational Methods in Engineering. 2022;29(7):4871-4896.",
     "https://doi.org/10.1007/s11831-022-09758-z"),
    ("Hossain S, Chakrabarty A, Gadekallu TR, Alazab M, Piran MJ. Vision transformers, "
     "ensemble model, and transfer learning leveraging explainable AI for brain tumor "
     "detection and classification. IEEE Journal of Biomedical and Health Informatics. "
     "2023;28(3):1261-1272.", "https://doi.org/10.1109/JBHI.2023.3266614"),
    ("Goceri E. An efficient network with CNN and transformer blocks for glioma grading and "
     "brain tumor classification from MRIs. Expert Systems with Applications. 2025;268:126290.",
     "https://doi.org/10.1016/j.eswa.2024.126290"),
    ("Apostolopoulos ID, Aznaouridis S, Tzani M. An attention-based deep convolutional neural "
     "network for brain tumor and disorder classification and grading in magnetic resonance "
     "imaging. Information. 2023;14(3):174.", "https://doi.org/10.3390/info14030174"),
    ("Demiroglu U. Tumor detection by classification of brain MRI images using the vision "
     "transformers. Adiyaman University Journal of Science. 2024;14(2):140-156.",
     "https://doi.org/10.37094/adyujsci.1572289"),
    ("Dixon J, Akinniyi O, Abdelhamid A, Saleh GA, Rahman MM, Khalifa F. A hybrid "
     "learning-architecture for improved brain tumor recognition. Algorithms. 2024;17(6):221.",
     "https://doi.org/10.3390/a17060221"),
    ("Hussain T, Shouno H, Hussain A, Hussain D, Ismail M, Mir TH, Akhy SA. EFFResNet-ViT: a "
     "fusion-based convolutional and vision transformer model for explainable medical image "
     "classification. IEEE Access. 2025.", "https://doi.org/10.1109/ACCESS.2025.3554184"),
    ("Tariq A, Iqbal MM, Iqbal MJ, Ahmad I. Transforming brain tumor detection: empowering "
     "multi-class classification with vision transformers and EfficientNetV2. IEEE Access. 2025.",
     "https://doi.org/10.1109/ACCESS.2025.3555638"),
    ("Kanadath A, Jothi JAA, Urolagin S. CViTS-Net: a CNN-ViT network with skip connections "
     "for histopathology image classification. IEEE Access. 2024.",
     "https://doi.org/10.1109/ACCESS.2024.3448302"),
    ("Tabatabaei S, Rezaee K, Zhu M. Attention transformer mechanism and fusion-based deep "
     "learning architecture for MRI brain tumor classification system. Biomedical Signal "
     "Processing and Control. 2023;86:105119.", "https://doi.org/10.1016/j.bspc.2023.105119"),
    ("Benzorgat N, Xia K, Benzorgat MNE. Enhancing brain tumor MRI classification with an "
     "ensemble of deep learning models and transformer integration. PeerJ Computer Science. "
     "2024;10:e2425.", "https://doi.org/10.7717/peerj-cs.2425"),
    ("Chattopadhay A, Sarkar A, Howlader P, Balasubramanian VN. Grad-CAM++: generalized "
     "gradient-based visual explanations for deep convolutional networks. IEEE Winter "
     "Conference on Applications of Computer Vision (WACV); 2018:839-847.",
     "https://doi.org/10.1109/WACV.2018.00097"),
    ("Islam MA, Mridha MF, Safran M, Alfarhood S, Kabir MM. Revolutionizing brain tumor "
     "detection using explainable AI in MRI images. NMR in Biomedicine. 2025;38(3):e70001.",
     "https://doi.org/10.1002/nbm.70001"),
    ("Nickparvar M. Brain tumor MRI dataset. Kaggle; 2022.",
     "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"),
    ("McNemar Q. Note on the sampling error of the difference between correlated proportions "
     "or percentages. Psychometrika. 1947;12(2):153-157.",
     "https://doi.org/10.1007/BF02295996"),
    ("Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal "
     "of Statistics. 1979;6(2):65-70.", "https://www.jstor.org/stable/4615733"),
]

HYPER = []   # (rid, url)


def ref_link(idx):
    """A clickable [n] citation is unnecessary; the reference list carries the links."""
    return "[%d]" % idx


def hyperlink_par(n, text, url):
    """Numbered reference with the URL as a clickable hyperlink."""
    rid = "rIdH%d" % n
    HYPER.append((rid, url))
    lead = rtext("[%d]  " % n, sz=SZ_SMALL)
    body = rtext(text + "  ", sz=SZ_SMALL)
    link = ('<w:hyperlink r:id="%s"><w:r><w:rPr>'
            '<w:rFonts w:ascii="%s" w:hAnsi="%s"/><w:sz w:val="%d"/><w:szCs w:val="%d"/>'
            '<w:color w:val="0563C1"/><w:u w:val="single"/></w:rPr>'
            '<w:t xml:space="preserve">%s</w:t></w:r></w:hyperlink>'
            % (rid, FONT, FONT, SZ_SMALL, SZ_SMALL, esc(url)))
    ppr = ('<w:spacing w:before="0" w:after="140" w:line="280" w:lineRule="auto"/>'
           '<w:jc w:val="both"/><w:ind w:left="600" w:hanging="600"/>')
    return "<w:p><w:pPr>%s</w:pPr>%s%s%s</w:p>" % (ppr, lead, body, link)
BODY = r'''

# ============================================================== TITLE PAGE ===
doc.append(p("MID-SEMESTER PROJECT REPORT", sz=32, bold=True, align="center",
             before=1400, after=200, line=240))
doc.append(p("B. Tech. Project", sz=26, align="center", after=700, line=240))
doc.append(p("AGMF-Net: An Adaptive Gated Multi-scale CNN-Transformer "
             "Fusion Network for Brain MRI Classification with Duplicate-Aware "
             "and Cross-Source Evaluation",
             sz=30, bold=True, align="center", after=900, line=300))
doc.append(p("Submitted by", sz=24, align="center", after=140, line=240))
doc.append(p("[STUDENT NAME(S) AND ROLL NUMBER(S)]", sz=24, bold=True,
             align="center", after=700, line=240))
doc.append(p("Under the supervision of", sz=24, align="center", after=140, line=240))
doc.append(p("[SUPERVISOR NAME AND DESIGNATION]", sz=24, bold=True,
             align="center", after=700, line=240))
doc.append(p("[DEPARTMENT]", sz=24, align="center", after=100, line=240))
doc.append(p("[INSTITUTE NAME]", sz=24, align="center", after=100, line=240))
doc.append(p("[ACADEMIC SESSION]", sz=24, align="center", after=600, line=240))
doc.append(p("Signature of Student(s): ____________________          "
             "Signature of Supervisor: ____________________",
             sz=22, align="center", after=0, line=240))
doc.append(pagebreak())

# ============================================================ 1. INTRODUCTION
doc.append(h("1. Introduction", 1))

doc.append(p(
    "Tumours of the central nervous system are among the more difficult conditions to "
    "assess from imaging alone. They vary widely in shape, size, location and signal "
    "characteristics, and two lesions of different histological type can look similar on a "
    "single slice while two examples of the same type can look quite different. Magnetic "
    "resonance imaging is the modality of choice for this assessment because it resolves "
    "soft tissue well and requires no ionising radiation, and the volume of MRI examinations "
    "now performed has grown to the point where reporting workload is itself a constraint on "
    "turnaround. Any tool that could reliably pre-sort studies, or flag those that need "
    "attention sooner, would therefore be useful.",
    first=420))

doc.append(p(
    "Interest has consequently turned to automated classification of tumour type directly "
    "from 2D structural slices. Convolutional neural networks became the standard approach "
    "because convolution encodes assumptions that suit medical images: locality and "
    "translation equivariance make the network efficient at picking up texture, edges and "
    "boundary irregularity. That same locality is also a limitation. A convolutional kernel "
    "sees a restricted neighbourhood, and stacking layers widens that neighbourhood only "
    "gradually, so relationships between a lesion and distant anatomical landmarks are "
    "captured indirectly at best.",
    first=420))

doc.append(p(
    "Vision transformers were introduced into medical imaging largely to address this. "
    "Self-attention places no locality restriction on which parts of an image may influence "
    "one another, so a transformer can relate a lesion to midline structures or ventricular "
    "geometry elsewhere in the same slice. Because the two families carry genuinely "
    "different inductive biases, combining them is a reasonable design idea rather than "
    "merely an additive one, and a substantial literature now reports hybrid "
    "convolutional-transformer classifiers for brain MRI [1], [2], [3], [4].",
    first=420))

doc.append(p(
    "What that literature reports far less consistently is how the hybrid is actually "
    "combined, and whether the reported accuracy would survive contact with data from "
    "somewhere else. Many hybrids concatenate a convolutional feature vector and a "
    "transformer feature vector and pass the result to a classifier. Concatenation fixes the "
    "relative influence of the two branches at training time and applies it identically to "
    "every image, even though the balance that suits a large, well-defined mass is unlikely "
    "to be the balance that suits a small lesion whose context matters more. Worse, when the "
    "two vectors have different widths, the wider branch acquires more influence on the "
    "downstream linear layer for no reason other than its width.",
    first=420))

doc.append(p(
    "Two further problems concern evaluation rather than architecture, and in practice they "
    "matter more. The first is near-duplicate leakage. Public brain MRI collections are "
    "typically assembled from several upstream sources and redistributed after resizing, "
    "re-encoding or augmentation, so the same underlying slice can appear more than once in "
    "a pooled set, sometimes on both sides of a train and test partition. A model that has "
    "memorised such an image is then scored as though it had generalised to it. Stratified "
    "splitting does not prevent this, because stratification balances class frequency and "
    "not image identity. The second is that a held-out split drawn from a single collection "
    "measures interpolation within one acquisition and post-processing regime only. It says "
    "nothing about behaviour when the scanner, sequence or export pipeline changes, which is "
    "precisely the situation a deployed classifier would face.",
    first=420))

doc.append(p(
    "This project addresses both levels. On the architectural side it develops **AGMF-Net "
    "(Adaptive Gated Multi-scale Fusion Network)**, a single feature-level hybrid in which "
    "convolutional features pooled from three network depths and a transformer class-token "
    "representation are first projected to a common width, and then combined by a small "
    "learned gate whose two outputs sum to one and are computed separately for every input "
    "image. Auxiliary classifiers attached to each branch during training keep both pathways "
    "discriminative, so neither can become a passenger. On the evaluation side the project "
    "treats the measurement protocol as part of the work rather than as a reporting "
    "formality: the image pool is audited for near-duplicates and repartitioned before any "
    "result is quoted, every configuration is trained under three seeds, each architectural "
    "component is removed in a controlled ablation and tested with paired statistics under "
    "multiplicity correction, and the model is additionally evaluated on a set of images "
    "from a second collection that share no near-duplicate with anything seen in training.",
    first=420))

doc.append(p(
    "The findings obtained so far justify that emphasis. On the duplicate-cleaned internal "
    "test set the model reaches 98.27 per cent accuracy, but on 373 duplicate-disjoint "
    "images drawn from a second source the same model reaches 69.53 per cent, and the "
    "degradation is concentrated almost entirely in one class. A single headline accuracy "
    "figure would not have been false, but it would have conveyed a misleading impression of "
    "how the model behaves outside the collection it was trained on.",
    first=420))

doc.append(h("1.1 Objectives of the project", 2))
doc.append(p("The project was undertaken with the following objectives.", after=100))
for i, t in enumerate([
    "To design a convolutional-transformer hybrid in which the contribution of each branch "
    "is determined per image by a learned gate, rather than being fixed by concatenation.",
    "To combine convolutional features from more than one spatial scale with the global "
    "transformer representation, so that information discarded by using only the terminal "
    "convolutional output is retained.",
    "To construct an evaluation protocol that controls for near-duplicate images between "
    "training and test partitions, and to quantify the effect of that control.",
    "To measure the contribution of every architectural component through controlled, "
    "protocol-matched ablations analysed with paired statistical tests and multiplicity "
    "correction, rather than by comparing single accuracy figures.",
    "To assess behaviour under a change of image source using a duplicate-disjoint external "
    "subset, and to report class-specific failure modes rather than aggregate accuracy alone.",
    "To report calibration alongside accuracy, so that the reliability of the model's "
    "confidence values is documented rather than assumed.",
], 1):
    doc.append(numbered("%d." % i, t))

doc.append(h("1.2 Organisation of this report", 2))
doc.append(p(
    "Section 2 reviews related work on convolutional, transformer and hybrid models for "
    "brain MRI classification, and identifies the gap this project addresses. Section 3 "
    "states the problem formally and describes the methodology, covering the dataset and "
    "duplicate audit, the AGMF-Net architecture, the training protocol, the evaluation "
    "protocol and the statistical procedure. Section 4 lists the references cited.",
    first=420))
doc.append(pagebreak())

# ========================================================= 2. LITERATURE =====
doc.append(h("2. Literature Survey", 1))

doc.append(p(
    "The literature relevant to this project falls into four groups: convolutional baselines, "
    "transformer-based models, hybrid and ensemble approaches, and the comparatively small "
    "body of work that examines evaluation practice itself. Unless stated otherwise, the "
    "performance figures quoted below are those reported by the authors on their own "
    "partitions, and are therefore not directly comparable with one another or with the "
    "results of this project.",
    first=420))

doc.append(h("2.1 Convolutional approaches", 2))
doc.append(p(
    "Convolutional networks remain a strong baseline for this task. Apostolopoulos et al. "
    "[5] added an attention mechanism to a convolutional backbone for tumour and disorder "
    "classification and grading, reporting 95.6 per cent accuracy. Survey work by Ali et al. "
    "[2] covers the broader landscape of deep learning for brain tumour diagnosis and notes "
    "the recurring reliance on single-collection evaluation. The general position that "
    "emerges is that a well-trained convolutional model with sensible preprocessing is "
    "difficult to beat by a large margin on these public benchmarks, which is itself an "
    "important observation when interpreting small reported improvements.",
    first=420))

doc.append(h("2.2 Transformer-based approaches", 2))
doc.append(p(
    "Transformer models have been applied to the same task with mixed results. Demiroglu [6] "
    "evaluated a standalone vision transformer and reported 81.9 per cent accuracy, "
    "noticeably below contemporary convolutional baselines, and attributed the gap to the "
    "absence of feature fusion. Dixon et al. [7] reported 92.7 per cent with a hybrid "
    "learning architecture, and Hussain et al. [8] described EFFResNet-ViT, a fusion-based "
    "convolutional and transformer model aimed at explainable classification, reporting 92.5 "
    "per cent. Tariq et al. [9] combined transformers with EfficientNetV2 in an ensemble and "
    "reported 96.0 per cent. Taken together these results suggest that transformers used in "
    "isolation on datasets of this size tend to underperform, and that most of their benefit "
    "in this domain has been realised through combination with convolutional features.",
    first=420))

doc.append(h("2.3 Hybrid and ensemble architectures", 2))
doc.append(p(
    "The hybrid line of work is the most directly relevant. Hossain et al. [3] combined "
    "vision transformers, ensembling and transfer learning with explainable AI components, "
    "reporting 96.9 per cent. Kanadath et al. [10] proposed CViTS-Net, a convolutional and "
    "transformer network with skip connections, reporting 95.0 per cent. Tabatabaei et al. "
    "[11] used an attention-transformer mechanism within a fusion-based architecture, and "
    "Benzorgat et al. [12] reported gains from ensembling several deep models with "
    "transformer integration. Goceri [4] proposed a network combining convolutional and "
    "transformer blocks for glioma grading and tumour classification.",
    first=420))

doc.append(p(
    "The most closely related work, and the one taken as the reference point for this "
    "project, is BRAIN-META by Napa et al. [1]. That study trains ten separate "
    "convolutional-transformer base learners, each pairing a different convolutional "
    "backbone with a transformer head, then concatenates their softmax outputs into a "
    "stacked feature vector and trains two meta-learners, logistic regression and XGBoost, "
    "on that vector. The XGBoost meta-learner reaches 97.10 per cent accuracy and 96.94 per "
    "cent F1 on a test partition of 1,311 images, above the best individual base learner at "
    "88.33 per cent. Grad-CAM heatmaps are provided from the convolutional components for "
    "interpretability, and the system is deployed as a Streamlit application.",
    first=420))

doc.append(p(
    "Two design characteristics of BRAIN-META are worth isolating, because AGMF-Net differs "
    "from it in both. First, the fusion happens at the level of predictions: the base "
    "learners never share features, and the meta-learner sees only their class "
    "probabilities. Second, the improvement comes from ensembling breadth, with ten models "
    "trained and retained, including several whose individual accuracy is reported as low as "
    "30.7 per cent and which are kept deliberately for architectural diversity. That design "
    "achieves its result at a correspondingly large training and inference cost.",
    first=420))

doc.append(h("2.4 Explainability and evaluation practice", 2))
doc.append(p(
    "Grad-CAM and its successors are the usual choice for visual interpretability in this "
    "domain. Grad-CAM++ [13] generalises the original weighting scheme and is used in the "
    "present project. Islam et al. [14] discuss explainable AI for brain tumour detection "
    "from MRI more broadly. A limitation common to all of this work, and one that applies "
    "equally to the present project, is that the maps are produced from convolutional "
    "activations and are rarely validated against expert annotation, so they indicate where "
    "a convolutional pathway is locally sensitive rather than confirming that a lesion has "
    "been correctly localised.",
    first=420))

doc.append(p(
    "Comparatively few studies in this area examine the evaluation protocol itself. Near-"
    "duplicate contamination between partitions of pooled public collections is seldom "
    "audited, and where a dataset is redistributed after re-encoding the risk is "
    "substantial. Evaluation on data from a source not represented in training is similarly "
    "uncommon; most reported figures are single-split, single-collection and single-run. "
    "Statistical treatment is usually limited to point accuracy without paired testing, "
    "multiplicity correction or any uncertainty estimate, which makes small reported "
    "improvements difficult to interpret.",
    first=420))

doc.append(h("2.5 Gap identified", 2))
doc.append(p(
    "Three gaps follow from the above, and together they define the scope of this project.",
    after=100))
for t in [
    "**Fusion is usually fixed rather than adaptive.** Where convolutional and transformer "
    "features are combined at the feature level, the combination is generally a "
    "concatenation whose branch balance is identical for every image, and is often between "
    "vectors of unequal width.",
    "**Component contributions are asserted rather than measured.** Architectural elements "
    "are presented as beneficial on the basis of a single overall accuracy figure, without "
    "controlled removal, paired testing or correction for the number of comparisons made.",
    "**Duplicate contamination and source shift are rarely tested.** Reported accuracy is "
    "typically measured on one collection with no audit for repeated images and no "
    "evaluation on data from a different source, so the figure describes interpolation "
    "within a single regime.",
]:
    doc.append(bullet(t))
doc.append(pagebreak())

# ================================== 3. PROBLEM STATEMENT AND METHODOLOGY =====
doc.append(h("3. Problem Statement and Methodology", 1))

doc.append(h("3.1 Problem statement", 2))
doc.append(p(
    "Given a 2D structural brain MRI slice, the task is to assign it to one of four classes: "
    "glioma, meningioma, no tumour, or pituitary tumour. The problem this project addresses "
    "is not simply to obtain a high accuracy on a held-out partition of a public collection, "
    "since published models already report figures above 97 per cent on this task. It is "
    "twofold.",
    first=420))
doc.append(p(
    "First, to design a convolutional-transformer fusion in which the relative contribution "
    "of the two branches is determined by the input rather than fixed at training time, and "
    "in which convolutional information from more than one spatial scale is retained.",
    first=420))
doc.append(p(
    "Second, to establish whether the resulting accuracy is a property of the model or a "
    "property of the evaluation protocol. Concretely: how much of the reported accuracy "
    "survives the removal of near-duplicate images between partitions; which architectural "
    "components can be shown to contribute once their removal is tested with paired "
    "statistics and multiplicity correction; and how far the accuracy transfers to images "
    "from a source not represented in training.",
    first=420))

doc.append(h("3.2 Dataset and duplicate control", 2))
doc.append(p(
    "The work uses a publicly available brain tumour MRI collection [15] comprising 7,023 "
    "two-dimensional images across the four classes. The files are image exports and carry "
    "no DICOM headers and no patient, study or series identifiers. This has a consequence "
    "that constrains the whole project and is stated here rather than in a footnote: because "
    "no grouping key exists, no partition used in this work can be verified as patient-"
    "independent, and no such claim is made anywhere in this report.",
    first=420))
doc.append(p(
    "Because pooled public collections commonly contain re-encoded copies of the same slice, "
    "the pool was audited for near-duplicates before any partition was formed. Each image "
    "was reduced to a 256-bit difference hash, and image pairs within a Hamming distance of "
    "8 were treated as near-duplicates. Connected components of this relation were formed, "
    "yielding 5,648 clusters from the 7,023 images, and one representative was retained from "
    "each cluster. This removed 1,375 redundant images. The retained images were then "
    "repartitioned with a fixed split seed into 4,800 training and 848 test images. Table 1 "
    "summarises the partition.",
    first=420))

doc.append(p("Table 1. Duplicate-clean partition of the image pool.",
             sz=SZ_SMALL, align="left", after=80))
doc.append(table([
    ["Quantity", "Value"],
    ["Pooled images", "7,023"],
    ["Perceptual hash", "256-bit difference hash"],
    ["Near-duplicate threshold", "Hamming distance <= 8"],
    ["Similarity clusters formed", "5,648"],
    ["Redundant images removed", "1,375"],
    ["Training images after cleaning", "4,800"],
    ["Test images after cleaning", "848"],
    ["Test class counts", "glioma 243, meningioma 228, no tumour 116, pituitary 261"],
], widths=[2, 3]))

doc.append(p(
    "It should be understood precisely what this procedure does. It controls redundancy at "
    "the level of the image. It does not identify patients, and two genuinely different "
    "slices acquired from the same patient will not be linked by it. Duplicate cleaning "
    "therefore removes one identifiable leakage pathway without eliminating leakage in "
    "general.",
    first=420))

doc.append(p(
    "A second, separate collection of 499 images was reserved as an external set. Applying "
    "the same hash criterion against the training pool showed that 126 of these images have "
    "a near-duplicate in training and 373 do not. Only the 373 duplicate-disjoint images are "
    "treated as a genuine source-shift test; the remaining 126 are never presented as unseen "
    "data. External labels were not used at any point for training, validation, early "
    "stopping, threshold selection or model selection. Figure 1 shows which images each "
    "reported number is computed on.",
    first=420))

doc.append(FIG_EVAL)
doc.append(p(
    "Figure 1. Evaluation protocol, showing the derivation of each partition from the pooled "
    "image set and which images each reported result is computed on.",
    sz=SZ_SMALL, align="center", after=220, line=240))

doc.append(h("3.3 AGMF-Net architecture", 2))
doc.append(p(
    "The network takes a 224 x 224 x 3 input and passes it to two parallel branches. Figure 2 "
    "shows the forward path.",
    first=420))
doc.append(p(
    "The convolutional branch uses an EfficientNet-B0 backbone. Rather than taking only the "
    "terminal output, activations are tapped at three depths: stage 3 with 40 channels, "
    "stage 5 with 112 channels, and the final stage with 1,280 channels. Each tap is reduced "
    "by global average pooling over its spatial dimensions and the three pooled vectors are "
    "concatenated, giving 1,432 dimensions. A linear layer maps this to 512 dimensions, "
    "followed by layer normalisation and a GELU non-linearity.",
    first=420))
doc.append(p(
    "The transformer branch uses a ViT-B/16 backbone. Its 768-dimensional class-token "
    "embedding is projected to 512 dimensions through a linear layer, layer normalisation "
    "and GELU. Projecting both branches to the same width is deliberate and is one of the "
    "specific corrections this design makes to plain concatenation: it removes the width "
    "asymmetry that would otherwise give the wider branch more influence for reasons "
    "unrelated to the information it carries.",
    first=420))
doc.append(p(
    "Writing c and v for the two projected 512-dimensional vectors, the gate receives their "
    "concatenation and produces two scalars through a linear layer of 1,024 to 256 "
    "dimensions, a GELU non-linearity, a linear layer of 256 to 2 dimensions, and a softmax. "
    "Because of the softmax the two gate values sum to one by construction. The fused "
    "representation is formed by scaling each branch by twice its gate value and "
    "concatenating the results, giving 1,024 dimensions. The factor of two keeps the "
    "expected scale comparable to plain concatenation, and has the property that when both "
    "gate values equal 0.5 the gated and ungated forms coincide exactly. Classification "
    "applies layer normalisation, dropout of 0.2, and a linear map to four logits.",
    first=420))
doc.append(p(
    "During training two auxiliary linear classifiers, each mapping 512 dimensions to four "
    "classes, are attached to the two projected branch representations. The training "
    "objective is the cross-entropy loss on the fused prediction plus 0.3 times the sum of "
    "the two auxiliary cross-entropy losses, with label smoothing of 0.1 applied throughout. "
    "The purpose is to require each branch to remain discriminative on its own, so that the "
    "head cannot learn to rely on one pathway while the other degenerates. The auxiliary "
    "heads are discarded at inference.",
    first=420))
doc.append(p(
    "The complete network has 91,211,918 parameters, of which the gate accounts for 262,914. "
    "Two properties of the gate should be stated carefully because they are easy to "
    "overstate. It is a learned module whose output varies with the input, which was "
    "verified empirically. It is not a fixed or manually chosen ratio. Equally, the gate "
    "value measures how much of each branch's representation enters the fused vector, which "
    "is branch utilisation, and does not by itself establish that a branch is causally "
    "responsible for a prediction.",
    first=420))

doc.append(FIG_ARCH)
doc.append(p(
    "Figure 2. AGMF-Net forward path. Both branches are projected to a common 512-dimensional "
    "width before the learned gate combines them. Layer widths shown were verified against "
    "the trained model.",
    sz=SZ_SMALL, align="center", after=220, line=240))

doc.append(h("3.4 Training protocol", 2))
doc.append(p(
    "Images are resized to 224 x 224 and normalised with a channel mean and standard "
    "deviation of 0.5. Training augmentation consists of random resized cropping with scale "
    "0.8 to 1.0, horizontal flipping, rotation of up to 15 degrees, and brightness and "
    "contrast jitter of 0.2. Optimisation uses AdamW. The remaining settings are listed in "
    "Table 2. Every configuration in the project was trained under three seeds, 42, 43 and "
    "44, and all reported variation is the sample standard deviation across those three "
    "runs.",
    first=420))

doc.append(p("Table 2. Training configuration.", sz=SZ_SMALL, align="left", after=80))
doc.append(table([
    ["Setting", "Value", "Setting", "Value"],
    ["Input size", "224 x 224", "Batch size", "16"],
    ["Optimiser", "AdamW", "Label smoothing", "0.1"],
    ["Base learning rate", "1e-4", "Maximum epochs", "15"],
    ["Backbone LR multiplier", "0.1", "Early-stopping patience", "4"],
    ["Weight decay", "0.05", "Validation fraction", "0.15"],
    ["Auxiliary loss weight", "0.3", "Training seeds", "42, 43, 44"],
], widths=[3, 2, 3, 2]))

doc.append(h("3.5 Evaluation protocol", 2))
doc.append(p(
    "The model is evaluated under three settings, shown in Figure 1. The duplicate-clean "
    "held-out split of 848 images is the primary internal result. The original held-out "
    "split of 1,052 images, formed without the duplicate audit, is retained for comparison "
    "so that the effect of duplicate cleaning is visible. The 373 duplicate-disjoint images "
    "form the source-shift test. Results from these settings are computed on different image "
    "sets and are never averaged together or presented as alternative measurements of the "
    "same quantity.",
    first=420))
doc.append(p(
    "Metrics are recomputed from the per-image prediction files rather than read from "
    "training logs. For each model and seed the project computes accuracy, macro precision, "
    "macro recall, macro F1, balanced accuracy, per-class precision, recall, specificity and "
    "F1, the confusion matrix, negative log-likelihood, the multiclass Brier score, expected "
    "calibration error over 15 equal-width bins, one-vs-rest ROC-AUC and one-vs-rest average "
    "precision. Where a quantity is mathematically undefined for a class it is reported as "
    "unavailable rather than as zero.",
    first=420))

doc.append(h("3.6 Ablation design", 2))
doc.append(p(
    "To determine which components contribute, seven variants were trained under an "
    "identical protocol, each altering exactly one element of the design: the gate removed, "
    "the multi-scale taps removed, the auxiliary losses removed, the exponential moving "
    "average removed, test-time augmentation removed, the transformer branch removed, and "
    "the convolutional branch removed. Each variant is compared against the full model at "
    "the same seed on the same 848 images, matched by exact filename, with the requirement "
    "that the true label agree with the split manifest for every image.",
    first=420))

doc.append(h("3.7 Statistical procedure", 2))
doc.append(p(
    "Seven variants across three seeds give 21 paired comparisons. For each, the 2 x 2 table "
    "of joint correctness is formed and an exact two-sided McNemar test [16] is applied to "
    "the discordant pairs, using the binomial form rather than the chi-square approximation "
    "because the discordant counts are small. The Holm step-down procedure [17] is applied "
    "across all 21 tests as a single family. Correcting across seven while selecting or "
    "pooling one value per variant was rejected, since that would be a smaller and more "
    "permissive family obtained by discarding evidence.",
    first=420))
doc.append(p(
    "Uncertainty on each paired difference is estimated by a bootstrap over images, "
    "stratified by true class, with 10,000 iterations at a fixed seed. A single resampling "
    "index set is applied to both models in each iteration so that the pairing is preserved. "
    "The resampling unit is an image; because no patient identifiers exist, no patient-level "
    "uncertainty can be estimated and none is reported. The seed standard deviation and the "
    "bootstrap interval measure different quantities and are reported separately, never "
    "combined. Finally, the 2,544 predictions arising from 848 images scored under three "
    "seeds are repeated measurements of the same images and are never treated as 2,544 "
    "independent observations.",
    first=420))

doc.append(h("3.8 Interpretability component", 2))
doc.append(p(
    "Grad-CAM++ [13] maps are generated from the final convolutional stage of the "
    "EfficientNet branch. Example images are selected deterministically as the first "
    "filename-ordered test image of each class, fixed before any heatmap is inspected, so "
    "that the figure is not a selection of visually appealing cases. The scope of this "
    "component is limited and is stated as such: the maps reflect the sensitivity of the "
    "score to one branch's final convolutional feature map, do not represent the transformer "
    "pathway, and have not been compared against expert annotation because no verified "
    "lesion masks exist for this dataset. They are a demonstrator, not validated "
    "localisation.",
    first=420))
doc.append(pagebreak())

# ============================================================= REFERENCES ====
doc.append(h("4. References", 1))
for i, (txt, url) in enumerate(REFS, 1):
    doc.append(hyperlink_par(i, txt, url))
'''


# ------------------------------------------------------------- assemble -----

def build():
    global FIG_ARCH, FIG_EVAL
    doc = []

    imgs = []
    arch = os.path.join(FIG, "figure_01_architecture.png")
    evalm = os.path.join(FIG, "figure_02_evaluation_map.png")
    for rid, path in (("rIdImg1", arch), ("rIdImg2", evalm)):
        if not os.path.exists(path):
            raise SystemExit("missing figure: %s" % path)
        imgs.append((rid, os.path.basename(path), path))
    FIG_ARCH = figure("rIdImg1", arch, "", frac=0.97)
    FIG_EVAL = figure("rIdImg2", evalm, "", frac=0.97)
    # figure() appends an empty caption paragraph; strip it, captions are explicit
    FIG_ARCH = FIG_ARCH.split("<w:p><w:pPr><w:spacing w:before=\"0\"")[0]
    FIG_EVAL = FIG_EVAL.split("<w:p><w:pPr><w:spacing w:before=\"0\"")[0]

    g = dict(globals())
    g.update(dict(doc=doc, FIG_ARCH=FIG_ARCH, FIG_EVAL=FIG_EVAL))
    exec(BODY, g)

    footer = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
              '<w:r><w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s"/><w:sz w:val="20"/></w:rPr>'
              '<w:fldChar w:fldCharType="begin"/></w:r>'
              '<w:r><w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s"/><w:sz w:val="20"/></w:rPr>'
              '<w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
              '<w:r><w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s"/><w:sz w:val="20"/></w:rPr>'
              '<w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>'
              % (FONT, FONT, FONT, FONT, FONT, FONT))

    sect = ('<w:sectPr><w:footerReference w:type="default" r:id="rIdFtr"/>'
            '<w:pgSz w:w="%d" w:h="%d"/>'
            '<w:pgMar w:top="%d" w:right="%d" w:bottom="%d" w:left="%d" '
            'w:header="708" w:footer="708" w:gutter="0"/>'
            '<w:docGrid w:linePitch="360"/></w:sectPr>'
            % (PAGE_W, PAGE_H, MARGIN, MARGIN, MARGIN, MARGIN))

    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document '
                'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
                'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                '<w:body>%s%s</w:body></w:document>' % ("".join(doc), sect))

    rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/styles" Target="styles.xml"/>',
            '<Relationship Id="rIdFtr" Type="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships/footer" Target="footer1.xml"/>']
    for rid, name, _ in imgs:
        rels.append('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/'
                    'officeDocument/2006/relationships/image" Target="media/%s"/>' % (rid, name))
    for rid, url in HYPER:
        rels.append('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/'
                    'officeDocument/2006/relationships/hyperlink" Target="%s" '
                    'TargetMode="External"/>' % (rid, esc(url)))
    rels.append("</Relationships>")

    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              '<w:docDefaults><w:rPrDefault><w:rPr>'
              '<w:rFonts w:ascii="%s" w:hAnsi="%s" w:eastAsia="%s" w:cs="%s"/>'
              '<w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr></w:rPrDefault>'
              '<w:pPrDefault><w:pPr>'
              '<w:spacing w:after="120" w:line="%d" w:lineRule="auto"/>'
              '</w:pPr></w:pPrDefault></w:docDefaults>'
              '<w:style w:type="paragraph" w:default="1" w:styleId="Normal">'
              '<w:name w:val="Normal"/><w:qFormat/></w:style></w:styles>'
              % (FONT, FONT, FONT, FONT, SZ, SZ, LINE))

    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.'
          'relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
          '<Default Extension="png" ContentType="image/png"/>'
          '<Override PartName="/word/document.xml" ContentType="application/vnd.'
          'openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
          '<Override PartName="/word/styles.xml" ContentType="application/vnd.'
          'openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
          '<Override PartName="/word/footer1.xml" ContentType="application/vnd.'
          'openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')

    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
                 'relationships"><Relationship Id="rId1" Type="http://schemas.'
                 'openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                 'Target="word/document.xml"/></Relationships>')

    out = os.path.join(OUT, "AGMF_NET_PROJECT_REPORT.docx")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles)
        z.writestr("word/footer1.xml", footer)
        z.writestr("word/_rels/document.xml.rels", "".join(rels))
        for _rid, name, path in imgs:
            z.write(path, "word/media/%s" % name)
    print("  wrote %s (%.0f KB)" % (os.path.relpath(out, ROOT), os.path.getsize(out) / 1024))
    print("  figures embedded: %d | hyperlinks: %d | references: %d"
          % (len(imgs), len(HYPER), len(REFS)))
    return out


if __name__ == "__main__":
    build()
    sys.exit(0)
