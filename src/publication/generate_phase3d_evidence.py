"""Build manuscript evidence matrices without changing models or results."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def csvout(p,rows):
 with open(p,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',required=True);ap.add_argument('--base-paper',required=True);ap.add_argument('--out-root',default='publication_results/manuscript_evidence');a=ap.parse_args()
 root=Path(a.project_root).resolve();out=(root/a.out_root);out.mkdir(parents=True,exist_ok=True);pdf=Path(a.base_paper).resolve()
 p1=root/'publication_results/metrics/fusion_v2_metrics.json';p3=root/'publication_results/ablations/final_analysis/final_ablation_analysis.json';pa=root/'publication_results/patient_audit/patient_group_audit.json';p3c=root/'publication_results/ablations/publication_artifacts/artifact_manifest.json'
 data=json.loads(p1.read_text());abl=json.loads(p3.read_text());patient=json.loads(pa.read_text());art=json.loads(p3c.read_text())
 if abl.get('acceptance_status')!='PASS' or art.get('acceptance_status')!='PASS': raise SystemExit('prior phase not accepted')
 def acc(setting): return data['settings'][setting]['across_seeds']['accuracy']['mean']*100,data['settings'][setting]['across_seeds']['accuracy']['sd_across_seeds']*100
 def mf1(setting): return data['settings'][setting]['across_seeds']['macro_f1']['mean']*100,data['settings'][setting]['across_seeds']['macro_f1']['sd_across_seeds']*100
 oa,os=acc('original');of,ofs=mf1('original');ca,cs=acc('clean');cf,cfs=mf1('clean');xa,xs=acc('cross_source_disjoint')
 comparison=[
 {'model':'FusionV2','protocol':'conventional/original held-out image split','n_images':1052,'accuracy_pct':f'{oa:.2f} ± {os:.2f}','f1_pct':f'{of:.2f} ± {ofs:.2f}','uncertainty':'sample SD across seeds 42/43/44','comparability':'contextual only versus BRAIN-META','source':'fusion_v2_metrics.json: settings.original'},
 {'model':'FusionV2','protocol':'duplicate-clean held-out image split','n_images':848,'accuracy_pct':f'{ca:.2f} ± {cs:.2f}','f1_pct':f'{cf:.2f} ± {cfs:.2f}','uncertainty':'sample SD across seeds 42/43/44','comparability':'primary internal protocol; not identical to BRAIN-META','source':'fusion_v2_metrics.json: settings.clean'},
 {'model':'FusionV2','protocol':'duplicate-disjoint cross-source image subset','n_images':373,'accuracy_pct':f'{xa:.2f} ± {xs:.2f}','f1_pct':'available in machine-readable metrics; not used for base-paper headline comparison','uncertainty':'sample SD across seeds 42/43/44','comparability':'different source and subset; not comparable to BRAIN-META test split','source':'fusion_v2_metrics.json: settings.cross_source_disjoint'},
 {'model':'BRAIN-META (XGBoost meta-learner)','protocol':'paper-reported 80/20 split; 5712 training and 1311 testing images','n_images':1311,'accuracy_pct':'97.10','f1_pct':'96.94','uncertainty':'not reported with multi-seed uncertainty in cited result','comparability':'contextual only; different preprocessing, split and ten-model stacking pipeline','source':'base PDF pp. 5 and 8; DOI 10.1016/j.mex.2025.103769'}]
 csvout(out/'base_paper_comparison.csv',comparison)
 assumptions=[
 {'assumption':'Image labels are correct','status':'untreated','evidence':'no independent relabelling or clinician adjudication available','remaining_consequence':'label error may bias all metrics'},
 {'assumption':'Near-duplicate clusters identify the targeted image-level duplication','status':'treated partially','evidence':'duplicate-clean split and split manifest','remaining_consequence':'does not establish patient independence'},
 {'assumption':'Held-out images represent independent patients','status':'not established','evidence':'patient audit found no defensible patient/study key','remaining_consequence':'same-patient leakage cannot be excluded'},
 {'assumption':'Seeds 42–44 characterize training variability','status':'treated partially','evidence':'three independently trained seeds','remaining_consequence':'n=3 gives an imprecise estimate of seed spread'},
 {'assumption':'Clean-test distribution represents deployment institutions/scanners','status':'untreated','evidence':'single benchmark family; acquisition provenance unavailable','remaining_consequence':'no multi-institutional generalization claim'},
 {'assumption':'Saved probabilities, filenames and labels align','status':'treated','evidence':'Phase 3A/addendum canonical fingerprint and Phase 3B validation','remaining_consequence':'no identified alignment failure'},
 {'assumption':'Gate mass measures causal branch importance','status':'rejected assumption','evidence':'gate utilization separated from controlled ablation evidence','remaining_consequence':'gate plots remain descriptive only'}]
 csvout(out/'assumptions.csv',assumptions)
 limitations=[
 {'limitation':'Patient identity unavailable','treatment':'metadata and filename audit','remaining_risk':'patient-level independence cannot be verified','allowed_wording':'held-out images on a duplicate-clean partition','prohibited_wording':'patient-independent split'},
 {'limitation':'Cross-source degradation','treatment':'373-image duplicate-disjoint evaluation','remaining_risk':'69.53 ± 1.98% accuracy and class-specific failure','allowed_wording':'substantial cross-source degradation was observed','prohibited_wording':'robust across hospitals or scanners'},
 {'limitation':'Only three training seeds','treatment':'report seed values and sample SD','remaining_risk':'uncertain run-to-run variability','allowed_wording':'mean ± seed SD for seeds 42–44','prohibited_wording':'seed SD is a confidence interval'},
 {'limitation':'Base-paper protocols differ','treatment':'separate rows and contextual label','remaining_risk':'reported values do not estimate controlled model difference','allowed_wording':'reported-value contextual comparison','prohibited_wording':'FusionV2 statistically outperforms BRAIN-META'},
 {'limitation':'Ablation multiplicity','treatment':'Holm correction across 21 seed-specific tests','remaining_risk':'limited power for small component effects','allowed_wording':'no detectable accuracy benefit under this protocol','prohibited_wording':'component is useless or equivalent'},
 {'limitation':'Calibration analysis exploratory','treatment':'descriptive NLL/Brier/ECE only','remaining_risk':'no prespecified paired calibration inference','allowed_wording':'numerically lower/higher in these runs','prohibited_wording':'better calibrated'},
 {'limitation':'No clinical validation','treatment':'explicit scope restriction','remaining_risk':'clinical utility and safety unknown','allowed_wording':'research benchmark result','prohibited_wording':'clinically ready'}]
 csvout(out/'limitations.csv',limitations)
 conc={x['variant']:x for x in abl['component_conclusions']}
 components=[]
 for v,label in [('no_gate','learned gate'),('no_multiscale','multi-scale taps'),('no_auxiliary_loss','auxiliary losses'),('no_ema','EMA'),('cnn_branch_only','ViT branch contribution'),('vit_branch_only','CNN branch contribution'),('no_tta','TTA')]:
  c=conc[v];components.append({'component':label,'ablation':v,'observed_delta_accuracy_pp':f"{c['observed_mean_accuracy_difference_pp']:+.4f}",'verdict':c['verdict'],'holm_significant_seeds':c['n_holm_significant_of_3'],'permitted_interpretation':('removing the ViT branch caused consistent accuracy loss' if v=='cnn_branch_only' else 'use the Phase 3B verdict verbatim; do not infer equivalence or zero effect')})
 csvout(out/'contribution_matrix.csv',components)
 claims=[
 {'claim':'FusionV2 achieved 99.14 ± 0.10% on the conventional split','status':'allowed','evidence':'Phase 1 original setting','required_caveat':'conventional/original image split'},
 {'claim':'FusionV2 achieved 98.27 ± 0.07% on the duplicate-clean split','status':'allowed','evidence':'Phase 1 clean setting','required_caveat':'held-out images, not verified held-out patients'},
 {'claim':'Removing the ViT branch reduced accuracy by about 2.59 pp','status':'allowed','evidence':'Phase 3B; Holm-significant at all three seeds','required_caveat':'duplicate-clean protocol only'},
 {'claim':'Auxiliary-loss and CNN-branch contributions are suggestive','status':'allowed with caution','evidence':'adverse direction across all seeds; corrected evidence insufficient','required_caveat':'suggestive but inconclusive'},
 {'claim':'Gate, multi-scale taps, EMA and TTA have zero value','status':'prohibited','evidence':'nonsignificant ablations do not prove equivalence','required_caveat':'say no detectable accuracy benefit under this protocol'},
 {'claim':'FusionV2 outperforms BRAIN-META statistically','status':'prohibited','evidence':'no aligned predictions or identical protocol','required_caveat':'contextual reported-value comparison only'},
 {'claim':'FusionV2 is clinically ready or universally generalizable','status':'prohibited','evidence':'no clinical or multi-institutional validation','required_caveat':'research benchmark only'},
 {'claim':'Patient-level independence was established','status':'prohibited','evidence':'no patient/study identifiers','required_caveat':'patient-level independence cannot be verified'},
 {'claim':'Gate mass provides anatomical or causal explanation','status':'prohibited','evidence':'gate mass is branch utilization telemetry','required_caveat':'descriptive utilization only'}]
 csvout(out/'claims_matrix.csv',claims)
 report=['# Phase 3D manuscript evidence','',f'Base paper: Kumar Napa et al., *MethodsX* 16 (2026) 103769, DOI 10.1016/j.mex.2025.103769. PDF SHA-256: `{sha(pdf)}`.','', '## Contextual comparison','', '| Model/evaluation | n | Accuracy | F1 | Comparability |','|---|---:|---:|---:|---|']
 for r in comparison: report.append(f"| {r['model']} — {r['protocol']} | {r['n_images']} | {r['accuracy_pct']} | {r['f1_pct']} | {r['comparability']} |")
 report += ['','The BRAIN-META values are verified from the supplied paper: its algorithm states an 80/20 split on PDF page 5; the dataset section reports 5,712 training and 1,311 testing images on page 5; and the XGBoost result reports accuracy 0.9710 and F1 0.9694 on page 8. This is not a controlled head-to-head comparison.','', '## What this study does not establish','', '- Patient-independent evaluation.','- Clinical readiness or diagnostic safety.','- Generalization across hospitals, scanners, populations, or acquisition protocols.','- Statistical superiority over BRAIN-META.','- Equivalence or zero utility for components with nonsignificant ablations.','- Causal or anatomical explanation from gate mass.','', '## Evidence-controlled component conclusions','']
 for r in components: report.append(f"- **{r['component']}**: {r['verdict']} (observed Δ accuracy {r['observed_delta_accuracy_pp']} pp).")
 report += ['','See the accompanying CSV matrices for manuscript-ready assumptions, limitations, contribution interpretations, and allowed/prohibited claims.','']
 (out/'MANUSCRIPT_EVIDENCE.md').write_text('\n'.join(report),encoding='utf-8')
 manifest={'phase':'3D','acceptance_status':'PASS','training_performed':False,'model_modified':False,'base_paper':{'path':str(pdf),'sha256':sha(pdf),'doi':'10.1016/j.mex.2025.103769','verified_pages':[1,5,8]},'inputs':{str(x.relative_to(root)).replace('\\','/'):sha(x) for x in (p1,p3,pa,p3c)},'outputs':[]}
 for p in sorted(out.iterdir()):
  if p.is_file() and p.name!='evidence_manifest.json':manifest['outputs'].append({'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
 (out/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Phase 3D: PASS')
if __name__=='__main__':main()
