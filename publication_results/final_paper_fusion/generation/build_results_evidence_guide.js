const fs = require('fs');
const path = require('path');
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun,
  LevelFormat, PageBreak, PageNumber, Packer, Paragraph, ShadingType,
  Table, TableCell, TableRow, TextRun, WidthType,
} = require('docx');

const root = path.resolve(process.argv[2] || '.');
const outDir = path.join(root, 'publication_results', 'final_paper_fusion');
const outPath = path.join(outDir, 'RESULTS_AND_EVIDENCE_PRESENTATION_GUIDE.docx');
const figureDir = path.join(outDir, 'figures');

const ink = '111827';
const muted = '4B5563';
const blue = '1F4E78';
const blueFill = 'EAF2F8';
const greenFill = 'EAF6EF';
const amberFill = 'FFF4D6';
const redFill = 'FDECEC';
const rule = 'CBD5E1';

function run(text, options = {}) {
  return new TextRun({
    text,
    font: 'Arial',
    size: options.size || 20,
    bold: options.bold || false,
    italics: options.italics || false,
    color: options.color || ink,
  });
}

function p(text, options = {}) {
  const children = Array.isArray(text) ? text : [run(text, options)];
  return new Paragraph({
    children,
    alignment: options.alignment,
    spacing: { before: options.before || 0, after: options.after ?? 120, line: options.line || 290 },
    keepNext: options.keepNext || false,
    pageBreakBefore: options.pageBreakBefore || false,
  });
}

function heading(text, level = 1, pageBreakBefore = false) {
  return new Paragraph({
    text,
    heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
    pageBreakBefore,
    keepNext: true,
    spacing: { before: level === 1 ? 260 : 180, after: 100 },
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    text,
    numbering: { reference: 'bullets', level },
    spacing: { after: 70, line: 270 },
  });
}

function callout(title, text, fill = blueFill) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    margins: { top: 130, bottom: 130, left: 170, right: 170 },
    rows: [new TableRow({
      children: [new TableCell({
        width: { size: 9360, type: WidthType.DXA },
        shading: { fill, type: ShadingType.CLEAR },
        borders: {
          top: { style: BorderStyle.SINGLE, size: 4, color: rule },
          bottom: { style: BorderStyle.SINGLE, size: 4, color: rule },
          left: { style: BorderStyle.SINGLE, size: 4, color: rule },
          right: { style: BorderStyle.SINGLE, size: 4, color: rule },
        },
        children: [
          p([run(title, { bold: true, color: blue }), run(`  ${text}`)], { after: 0 }),
        ],
      })],
    })],
  });
}

function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const makeCell = (text, width, header = false) => new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: 'DCE6F1', type: ShadingType.CLEAR } : undefined,
    margins: { top: 75, bottom: 75, left: 85, right: 85 },
    children: [p(String(text), { after: 0, line: 240, bold: header, size: header ? 18 : 17 })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((x, i) => makeCell(x, widths[i], true)) }),
      ...rows.map(r => new TableRow({ children: r.map((x, i) => makeCell(x, widths[i])) })),
    ],
  });
}

function figure(filename, caption, width = 620) {
  const file = path.join(figureDir, filename);
  if (!fs.existsSync(file)) return [p(`[Figure unavailable: ${filename}]`)];
  const data = fs.readFileSync(file);
  const heightMap = {
    'figure_01_architecture.png': 306,
    'figure_03_regime_collapse.png': 345,
    'figure_05_ablation_forest.png': 365,
    'figure_08_class_failure.png': 350,
  };
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [new ImageRun({
        data,
        type: 'png',
        transformation: { width, height: heightMap[filename] || Math.round(width * 0.56) },
        altText: { title: caption, description: caption, name: caption },
      })],
    }),
    p(caption, { alignment: AlignmentType.CENTER, italics: true, color: muted, size: 17, after: 150 }),
  ];
}

const children = [];

children.push(new Paragraph({
  text: 'Results and Evidence Presentation Guide',
  heading: HeadingLevel.TITLE,
  alignment: AlignmentType.CENTER,
  spacing: { after: 140 },
}));
children.push(p('Multi Scale EfficientNet B0 and ViT B16 Fusion Network', {
  alignment: AlignmentType.CENTER, bold: true, size: 23, color: blue, after: 70,
}));
children.push(p('For the publication manuscript and B Tech progress report', {
  alignment: AlignmentType.CENTER, italics: true, color: muted, after: 220,
}));
children.push(callout('Main conclusion', 'The work is strongest when it presents both sides of the evidence: 98.27 ± 0.07% duplicate-clean internal accuracy and the fall to 69.53 ± 1.98% on 373 duplicate-disjoint cross-source images. The fusion architecture is the proposed method; the duplicate-aware, multi-seed and cross-source evaluation is the main evidence that makes the study defensible.', greenFill));

children.push(heading('1 What the reader must understand', 1));
children.push(bullet('The proposed model is a single feature-level fusion network, not a stacking ensemble.'));
children.push(bullet('EfficientNet-B0 contributes pooled features from three scales; ViT-B/16 contributes its 768-dimensional class token.'));
children.push(bullet('Both branches are projected to 512 dimensions before gated fusion and four-class classification.'));
children.push(bullet('The primary internal result is the duplicate-clean test result, not the fine-tuned CNN branch result and not the conventional split alone.'));
children.push(bullet('The external result demonstrates unresolved source-shift fragility, especially for the pituitary class.'));
children.push(bullet('BRAIN-META is a contextual reference only because the evaluation protocols and test images differ.'));

children.push(heading('2 Results that must appear', 1));
children.push(p('Use the following values exactly. Every ± value below is the sample standard deviation across seeds 42, 43 and 44. It is not a confidence interval.'));
children.push(table(
  ['Evaluation setting', 'Images', 'Accuracy', 'Macro F1', 'How to use it'],
  [
    ['Original held-out split', '1,052', '99.14 ± 0.10%', '99.12 ± 0.09%', 'Secondary conventional benchmark'],
    ['Duplicate-clean held-out split', '848', '98.27 ± 0.07%', '98.28 ± 0.04%', 'Primary internal result'],
    ['External collection, all images', '499', '76.49 ± 1.52%', '70.61 ± 2.15%', 'Descriptive only; includes 126 overlaps'],
    ['Duplicate-disjoint cross-source subset', '373', '69.53 ± 1.98%', '68.69 ± 2.21%', 'Primary source-shift result'],
  ],
  [2800, 850, 1500, 1500, 2710],
));
children.push(p('Required interpretation', { bold: true, color: blue, before: 130, after: 60 }));
children.push(bullet('The 28.74 percentage-point fall from duplicate-clean internal accuracy to duplicate-disjoint cross-source accuracy is the central robustness finding.'));
children.push(bullet('Do not subtract BRAIN-META 97.10% from your results and call the difference an improvement. That would compare different protocols.'));
children.push(...figure('figure_03_regime_collapse.png', 'Recommended main result figure. Individual seeds remain visible, and the error bars are seed standard deviations rather than confidence intervals.'));

children.push(heading('3 Dataset and split evidence', 1, true));
children.push(table(
  ['Evidence item', 'Verified value', 'Why it matters'],
  [
    ['Pooled images', '7,023', 'Defines the full audited pool'],
    ['Perceptual-hash clusters', '5,648', 'Groups visually redundant images'],
    ['Redundant images removed', '1,375', 'Reduces identifiable image-level duplication'],
    ['Duplicate-clean split', '4,800 train and 848 test', 'Primary internal protocol'],
    ['Hash rule', '256-bit dHash, Hamming distance ≤ 8', 'Operational duplicate definition'],
    ['External collection', '499 total; 126 overlapping; 373 disjoint', 'Separates overlap from unseen-source stress test'],
    ['Patient grouping', 'Not available', 'Patient-level independence cannot be verified'],
  ],
  [2700, 2300, 4360],
));
children.push(callout('Required limitation', 'Duplicate cleaning controls image redundancy only. It does not prove patient-independent splitting because patient, study and series identifiers are unavailable.', amberFill));
children.push(p('Evidence sources', { bold: true, color: blue, before: 140, after: 60 }));
children.push(bullet('dataset_clean/split_report.json'));
children.push(bullet('results/cross_source.json'));
children.push(bullet('publication_results/patient_audit/patient_group_audit.json'));

children.push(heading('4 Architecture evidence', 1));
children.push(table(
  ['Component', 'Verified specification'],
  [
    ['CNN branch', 'EfficientNet-B0 taps at 40, 112 and 1,280 channels'],
    ['Multi-scale vector', '40 + 112 + 1,280 = 1,432 dimensions'],
    ['CNN projection', '1,432 → 512, LayerNorm and GELU'],
    ['Transformer branch', 'ViT-B/16 class token, 768 dimensions'],
    ['ViT projection', '768 → 512, LayerNorm and GELU'],
    ['Fusion gate', '1,024 → 256 → 2 logits, softmax; 262,914 parameters'],
    ['Fused vector', '[2g_c c ; 2g_v v], 1,024 dimensions'],
    ['Classifier', 'LayerNorm, dropout 0.2, linear 1,024 → 4'],
    ['Auxiliary supervision', 'One 512 → 4 head per branch; weight 0.3 during training'],
    ['Total parameters', '91,211,918'],
  ],
  [2900, 6460],
));
children.push(...figure('figure_01_architecture.png', 'Recommended architecture figure. Present feature-level fusion as the core design; do not present adaptive gating as the sole novelty.'));
children.push(callout('Verified checkpoint evidence', 'All three recovered checkpoints loaded with 578 of 578 state-dictionary keys, no missing or unexpected tensors, and reproduced all 848 clean-test predicted labels. Maximum probability difference was 4.8 × 10⁻⁶.', greenFill));

children.push(heading('5 Internal ablation evidence', 1, true));
children.push(table(
  ['Variant', 'Accuracy', 'Difference', 'Holm significant', 'Conclusion'],
  [
    ['Full fusion network', '98.27 ± 0.07%', 'Reference', '—', 'Reference'],
    ['CNN only, ViT removed', '95.68 ± 0.34%', '−2.59 pp', '3 of 3 seeds', 'Supported loss'],
    ['No auxiliary loss', '97.72 ± 0.48%', '−0.55 pp', '0 of 3', 'Suggestive, inconclusive'],
    ['ViT only, CNN removed', '98.03 ± 0.14%', '−0.24 pp', '0 of 3', 'Suggestive, inconclusive'],
    ['No TTA', '98.19 ± 0.25%', '−0.08 pp', '0 of 3', 'No detectable benefit'],
    ['No multi-scale taps', '98.23 ± 0.20%', '−0.04 pp', '0 of 3', 'No detectable benefit'],
    ['No gate', '98.27 ± 0.18%', '0.00 pp', '0 of 3', 'No detectable benefit'],
    ['No EMA', '98.31 ± 0.30%', '+0.04 pp', '0 of 3', 'No detectable benefit'],
  ],
  [2300, 1350, 1200, 1500, 3010],
));
children.push(p('Statistical evidence to state', { bold: true, color: blue, before: 130, after: 60 }));
children.push(bullet('Seven variants × three seeds produced 21 paired comparisons.'));
children.push(bullet('Each variant was matched to the full model at the same seed on the same 848 images.'));
children.push(bullet('Exact two-sided McNemar tests were used, with Holm correction across all 21 tests.'));
children.push(bullet('Paired image-level bootstrap intervals used 10,000 stratified resamples. These are not patient-level intervals.'));
children.push(bullet('Only the CNN-only variant remained significant at all three seeds after correction.'));
children.push(...figure('figure_05_ablation_forest.png', 'Recommended ablation figure. The observed effect and paired bootstrap interval are shown for every seed.'));
children.push(callout('Interpretation rule', 'No significant difference does not mean equivalence, zero effect or an unnecessary component. Write “no detectable accuracy benefit under this protocol.”', amberFill));

children.push(heading('6 Cross-source evidence and class failure', 1, true));
children.push(table(
  ['Class', 'Support', 'Mean recall', 'Required interpretation'],
  [
    ['Glioma', '52', '98.08%', 'Largely preserved'],
    ['Meningioma', '122', '86.07%', 'Moderate degradation'],
    ['No tumour', '88', '100.00%', 'Preserved on this subset'],
    ['Pituitary', '111', '13.81%', 'Severe unresolved failure'],
  ],
  [1800, 1200, 1600, 4760],
));
children.push(callout('Critical class-level result', 'Across three seeds, 111 pituitary images generated 333 repeated predictions; 260 of 333 predictions, or 78.1%, were assigned to no tumour. The 333 predictions are not 333 independent cases.', redFill));
children.push(p('Cross-source ablation result', { bold: true, color: blue, before: 140, after: 60 }));
children.push(bullet('Full network: 69.53 ± 1.98% accuracy.'));
children.push(bullet('CNN-only: 44.50 ± 3.09% accuracy, a −25.02 pp difference; significant at all three seeds after Holm correction.'));
children.push(bullet('All other cross-source ablation differences had zero of three Holm-significant seed comparisons.'));
children.push(bullet('No-auxiliary-loss reached 70.15 ± 1.57%, but this is a final-test observation and must not be used retrospectively for model selection.'));
children.push(...figure('figure_08_class_failure.png', 'Recommended class-failure figure. It prevents aggregate accuracy from hiding the pituitary-to-no-tumour error pattern.'));

children.push(heading('7 Calibration and confidence evidence', 1));
children.push(table(
  ['Metric', 'Duplicate-clean', 'Cross-source', 'Interpretation'],
  [
    ['Balanced accuracy', '0.9820', '0.7449', 'Large decision-performance loss'],
    ['Macro ROC-AUC', '0.9995', '0.9849', 'Ranking degrades less than accuracy'],
    ['Macro average precision', '0.9985', '0.9664', 'High ranking quality persists'],
    ['Negative log-likelihood', '0.1188', '0.7493', 'Substantially worse under shift'],
    ['Brier score', '0.0335', '0.4192', 'Substantially worse under shift'],
    ['ECE, 15 bins', '0.0752', '0.1618', 'Confidence becomes less reliable'],
  ],
  [2400, 1700, 1700, 3560],
));
children.push(callout('Required wording', 'The softmax score is the model output, not the probability that a medical diagnosis is correct. Calibration comparisons are descriptive because no paired calibration test was prespecified.', amberFill));

children.push(heading('8 Grad CAM++ evidence', 1));
children.push(p('Present Grad-CAM++ as a verified implementation and limited CNN-path sensitivity visualisation, not as validated tumour localisation.'));
children.push(table(
  ['True class', 'Selected probability', 'Deletion AUC CAM / random', 'Insertion AUC CAM / random', 'Flip correlation'],
  [
    ['Glioma', '0.9362', '0.3981 / 0.7949', '0.7645 / 0.6324', '0.9778'],
    ['Meningioma', '0.9288', '0.3045 / 0.6960', '0.8529 / 0.7129', '0.9361'],
    ['No tumour', '0.9413', '0.5165 / 0.4972', '0.7311 / 0.6700', '0.9411'],
    ['Pituitary', '0.9143', '0.1753 / 0.2656', '0.5811 / 0.3589', '0.9821'],
  ],
  [1600, 1500, 2200, 2200, 1860],
));
children.push(bullet('Examples were selected deterministically before inspecting the maps.'));
children.push(bullet('The target is the fused-model pre-softmax class logit; the layer is the final spatial EfficientNet stage.'));
children.push(bullet('No verified lesion masks, bounding boxes or clinician annotations are available.'));
children.push(bullet('A red region on a no-tumour image means strongest support for the no-tumour score after within-image normalisation; it does not indicate a tumour.'));

children.push(heading('9 Comparison with BRAIN META', 1, true));
children.push(table(
  ['Characteristic', 'BRAIN-META', 'Proposed work'],
  [
    ['Design', 'Ten CNN–ViT base learners plus logistic-regression and XGBoost meta-learners', 'Single EfficientNet-B0–ViT-B/16 feature-fusion network'],
    ['Fusion', 'Prediction-level softmax stacking', 'Projected feature-level fusion'],
    ['Preprocessing', 'Resize, normalise and CLAHE', 'Resize and normalise with mean/std 0.5; no CLAHE'],
    ['Test set', '1,311 images', '1,052 original; 848 duplicate-clean; 373 cross-source'],
    ['Duplicate control', 'Not reported in inspected paper', 'dHash audit; 1,375 redundant images removed'],
    ['Reported accuracy', '97.10%', '99.14 ± 0.10% original; 98.27 ± 0.07% duplicate-clean'],
    ['Reported F1', '96.94%', '99.12 ± 0.09% original; 98.28 ± 0.04% duplicate-clean'],
    ['Comparability', 'Different reported protocol', 'Contextual only; no controlled ranking'],
  ],
  [2200, 3400, 3760],
));
children.push(callout('Do not claim', 'Do not state that the proposed model statistically outperforms BRAIN-META. Its per-image predictions were unavailable and it was not reproduced on the identical split.', redFill));

children.push(heading('10 Assumptions and limitations', 1));
children.push(p('Addressed or partly addressed', { bold: true, color: blue, after: 60 }));
children.push(bullet('Near-duplicate image leakage was reduced through perceptual-hash clustering and repartitioning.'));
children.push(bullet('Run-to-run variation was partly addressed through three independently trained seeds.'));
children.push(bullet('Component attribution was addressed with protocol-matched ablations, paired tests and multiplicity correction.'));
children.push(bullet('Accuracy-only reporting was expanded with class-wise, ranking and calibration metrics.'));
children.push(bullet('Source-shift blindness was exposed through the 373-image duplicate-disjoint evaluation; it was not solved.'));
children.push(p('Not addressed', { bold: true, color: blue, before: 130, after: 60 }));
children.push(bullet('Patient, study and series independence.'));
children.push(bullet('Scanner, sequence, institution and demographic stratification.'));
children.push(bullet('Three-dimensional study-level modelling.'));
children.push(bullet('Prospective, multi-institutional or reader-study validation.'));
children.push(bullet('Quantitative tumour localisation against verified annotations.'));
children.push(bullet('Fairness, clinical utility and deployment safety.'));
children.push(bullet('A controlled same-split reproduction of BRAIN-META.'));

children.push(heading('11 What to show in each deliverable', 1, true));
children.push(table(
  ['Evidence item', 'Progress report', 'Publication paper', 'Supplement'],
  [
    ['Architecture and training protocol', 'Summary plus architecture figure', 'Full method and Figure 1', 'Exact layer and hyperparameter tables'],
    ['Original split result', 'Brief benchmark', 'Secondary result', 'Per-seed metrics'],
    ['Duplicate-clean result', 'Headline result', 'Primary internal result', 'Full metric panel'],
    ['Cross-source result', 'Headline limitation', 'Primary robustness result', 'Per-seed and per-class tables'],
    ['Internal ablation', 'Key conclusion', 'Full table and forest plot', 'All 21 paired tests and bootstrap intervals'],
    ['Cross-source ablation', 'CNN-only failure', 'Full summary table', 'All paired tests'],
    ['Calibration', 'ECE comparison', 'Reliability plot and probabilistic metrics', 'Bin definitions and values'],
    ['Grad-CAM++', 'One example per class', 'Audit figure with limitations', 'Deletion, insertion and stability values'],
    ['BRAIN-META', 'Contextual comparison', 'Protocol-aware comparison table', 'Citation and template record'],
    ['Limitations', 'Explicit list', 'Dedicated limitations section', 'Evidence and residual-risk table'],
  ],
  [2600, 2100, 2400, 2260],
));

children.push(heading('12 Exact files to use', 1));
children.push(table(
  ['Purpose', 'Verified project file'],
  [
    ['Headline metrics', 'publication_results/metrics/fusion_v2_metrics.json'],
    ['Duplicate audit', 'dataset_clean/split_report.json'],
    ['External overlap manifest', 'results/cross_source.json'],
    ['Patient metadata audit', 'publication_results/patient_audit/patient_group_audit.json'],
    ['Checkpoint recovery', 'publication_results/ablations/checkpoint_recovery/import_manifest.json'],
    ['Internal ablation statistics', 'publication_results/ablations/final_analysis/final_ablation_analysis.json'],
    ['Cross-source ablations', 'publication_results/ablations/cross_source/cross_source_ablation_analysis.json'],
    ['Grad-CAM++ audit', 'publication_results/xai/XAI_AUDIT.json'],
    ['Claim traceability', 'publication_results/final_paper_fusion/EVIDENCE_TRACEABILITY.csv'],
    ['Ready figures', 'publication_results/final_paper_fusion/figures/'],
    ['Ready manuscript', 'publication_results/final_paper_fusion/FUSION_NETWORK_MANUSCRIPT.docx'],
  ],
  [3000, 6360],
));

children.push(heading('13 Final wording guardrails', 1));
children.push(table(
  ['Use this wording', 'Avoid this wording'],
  [
    ['Duplicate-clean image-level split', 'Leakage-free or patient-independent split'],
    ['No detectable accuracy benefit under this protocol', 'The component is useless or equivalent'],
    ['Duplicate-disjoint cross-source stress test', 'Universal external validation'],
    ['Contextual comparison with BRAIN-META', 'Statistically better than BRAIN-META'],
    ['CNN-path Grad-CAM++ sensitivity map', 'Verified tumour localisation or model reasoning'],
    ['Softmax confidence score', 'Probability that the diagnosis is correct'],
    ['Research demonstrator', 'Clinically ready diagnostic system'],
  ],
  [4680, 4680],
));

const doc = new Document({
  creator: 'Fusion network research project',
  title: 'Results and Evidence Presentation Guide',
  description: 'Verified evidence checklist for the publication manuscript and progress report',
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 220 } } } }],
    }],
  },
  styles: {
    default: { document: { run: { font: 'Arial', size: 20, color: ink }, paragraph: { spacing: { line: 290 } } } },
    paragraphStyles: [
      { id: 'Title', name: 'Title', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: 'Arial', size: 34, bold: true, color: '000000' },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 180 }, outlineLevel: 0 } },
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: 'Arial', size: 26, bold: true, color: '000000' },
        paragraph: { spacing: { before: 260, after: 100 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { font: 'Arial', size: 22, bold: true, color: '000000' },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1008, right: 1008, bottom: 1008, left: 1008 } },
    },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [run('Results and evidence guide  |  Page ', { size: 16, color: muted }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: muted })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log(outPath);
});
