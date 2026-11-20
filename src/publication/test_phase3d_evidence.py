import csv,json,sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--project-root')+1]).resolve();out=root/'publication_results/manuscript_evidence'
tests=[]
def ck(n,x):tests.append((n,bool(x)));print('PASS' if x else 'FAIL',n)
cmp=list(csv.DictReader((out/'base_paper_comparison.csv').open(encoding='utf-8'))); claims=list(csv.DictReader((out/'claims_matrix.csv').open(encoding='utf-8'))); comps=list(csv.DictReader((out/'contribution_matrix.csv').open(encoding='utf-8')))
ck('four comparison rows',len(cmp)==4);ck('base accuracy verified',cmp[-1]['accuracy_pct']=='97.10');ck('base F1 verified',cmp[-1]['f1_pct']=='96.94');ck('clean accuracy separated',cmp[1]['accuracy_pct'].startswith('98.27'));ck('cross-source disclosed',cmp[2]['accuracy_pct'].startswith('69.53'));ck('all comparisons contextual',all('context' in r['comparability'].lower() or r['model']=='FusionV2' for r in cmp));ck('seven component rows',len(comps)==7);ck('ViT contribution supported',next(r for r in comps if r['ablation']=='cnn_branch_only')['verdict']=='supported by consistent accuracy evidence');ck('statistical superiority prohibited',next(r for r in claims if 'outperforms BRAIN-META' in r['claim'])['status']=='prohibited');ck('patient claim prohibited',next(r for r in claims if 'Patient-level' in r['claim'])['status']=='prohibited')
md=(out/'MANUSCRIPT_EVIDENCE.md').read_text(encoding='utf-8').lower();ck('no clinical readiness claim','clinically ready.' not in md);ck('explicit non-comparability','not a controlled head-to-head' in md);ck('PDF pages cited','pdf page 5' in md and 'page 8' in md)
limit_rows=list(csv.DictReader((out/'limitations.csv').open(encoding='utf-8')))
structured_prohibition=(next(r for r in claims if 'outperforms BRAIN-META' in r['claim'])['status']=='prohibited'
 and any('statistically outperforms' in r['prohibited_wording'].lower() for r in limit_rows))
ck('no unsupported superiority',structured_prohibition and 'fusionv2 statistically outperforms' not in md)
failed=[n for n,x in tests if not x];(out/'TEST_REPORT.md').write_text('# Phase 3D test report\n\n'+'\n'.join(f"- {'PASS' if x else 'FAIL'} — {n}" for n,x in tests)+f"\n\n**Result: {len(tests)-len(failed)}/{len(tests)} passed.**\n",encoding='utf-8');print(f'{len(tests)-len(failed)}/{len(tests)} passed');raise SystemExit(bool(failed))
