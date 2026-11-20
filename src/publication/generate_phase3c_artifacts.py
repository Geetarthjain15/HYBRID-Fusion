"""Generate Phase 3C publication tables and figures from accepted Phase 3B data.

This script is deliberately analysis-only: it reads immutable predictions and
Phase 3B results, performs no training or model selection, and writes only to a
new publication-artifact directory.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os, platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["full_fusion_v2", "no_gate", "no_multiscale", "no_auxiliary_loss",
            "no_ema", "cnn_branch_only", "vit_branch_only", "no_tta"]
LABELS = ["Glioma", "Meningioma", "No tumor", "Pituitary"]
DISPLAY = {"full_fusion_v2":"Full FusionV2", "no_gate":"No gate",
           "no_multiscale":"No multi-scale", "no_auxiliary_loss":"No auxiliary loss",
           "no_ema":"No EMA", "cnn_branch_only":"CNN only",
           "vit_branch_only":"ViT only", "no_tta":"No TTA"}

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def write_csv(path, rows, fields):
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def savefig(fig, base: Path):
    fig.savefig(str(base)+".png",dpi=300,facecolor="white")
    fig.savefig(str(base)+".pdf",facecolor="white")
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",required=True)
    ap.add_argument("--out-root",default="publication_results/ablations/publication_artifacts")
    ap.add_argument("--visual-inspected",action="store_true")
    a=ap.parse_args(); root=Path(a.project_root).resolve(); out=(root/a.out_root).resolve()
    source=root/"publication_results/ablations/final_analysis/final_ablation_analysis.json"
    if not source.exists(): raise SystemExit("missing Phase 3B JSON")
    d=json.loads(source.read_text(encoding="utf-8"))
    if d.get("acceptance_status")!="PASS": raise SystemExit("Phase 3B did not PASS")
    if out.exists() and any(out.iterdir()):
        allowed={"tables","figures","PHASE3C_ARTIFACTS.md","artifact_manifest.json","VISUAL_INSPECTION.md","TEST_REPORT.md"}
        if any(p.name not in allowed for p in out.iterdir()): raise SystemExit("unexpected existing output; refusing overwrite")
    (out/"tables").mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.titlesize":11,"axes.labelsize":10,"figure.dpi":120})

    # Tables are generated directly from Phase 3B machine-readable evidence.
    ab=[]
    conclusions={x["variant"]:x for x in d["component_conclusions"]}
    for v in VARIANTS:
        s=d["metrics_summary"][v]; c=conclusions.get(v,{})
        tests=[x for x in d["paired_tests"] if x["variant"]==v]
        ab.append({"model":v,"accuracy_mean_pct":100*s["accuracy"]["mean"],
                   "accuracy_seed_sd_pct":100*s["accuracy"]["sd"],
                   "macro_f1_mean_pct":100*s["macro_f1"]["mean"],
                   "macro_f1_seed_sd_pct":100*s["macro_f1"]["sd"],
                   "observed_delta_accuracy_pp":c.get("observed_mean_accuracy_difference_pp",0.0),
                   "holm_significant_seeds":c.get("n_holm_significant_of_3","reference"),
                   "verdict":c.get("verdict","reference")})
    write_csv(out/"tables/final_ablation_table.csv",ab,list(ab[0]))

    prov=[]
    for v in VARIANTS:
        for seed in (42,43,44):
            key=f"{v}|{seed}"; m=d["metrics_by_seed"][key]
            prov.append({"model":v,"seed":seed,"split":"duplicate-clean","n_images":m["n_images"],
                         "canonical_fingerprint":d["all_runs_share_canonical_fingerprint"],
                         "source_record":key})
    write_csv(out/"tables/experiment_provenance.csv",prov,list(prov[0]))
    # Preserve Phase 3B per-class values verbatim as the publication table source.
    pc_src=root/"publication_results/ablations/final_analysis/per_class_metrics.csv"
    (out/"tables/per_class_metrics.csv").write_bytes(pc_src.read_bytes())

    # Figure 1: accuracy effects with seed points and seed SD (not CI).
    means=np.array([100*d["metrics_summary"][v]["accuracy"]["mean"] for v in VARIANTS])
    sds=np.array([100*d["metrics_summary"][v]["accuracy"]["sd"] for v in VARIANTS])
    fig,ax=plt.subplots(figsize=(8.2,4.8)); y=np.arange(len(VARIANTS))
    colors=["#1f4e79"]+["#5b9bd5"]*7
    ax.errorbar(means,y,xerr=sds,fmt="none",ecolor="#444",capsize=3,lw=1.2,zorder=1)
    ax.scatter(means,y,c=colors,s=48,zorder=3)
    for i,v in enumerate(VARIANTS):
        vals=[100*d["metrics_by_seed"][f"{v}|{s}"]["accuracy"] for s in (42,43,44)]
        ax.scatter(vals,[i-.13,i,i+.13],s=15,color="#222",alpha=.65,zorder=4)
    ax.set_yticks(y, [DISPLAY[v] for v in VARIANTS]); ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%) — point: seed mean; bar: seed SD; dots: individual seeds")
    ax.set_title("Duplicate-clean ablation performance (n=848 images per seed)")
    ax.grid(axis="x",alpha=.25); ax.set_xlim(min(means)-.8,100.05)
    fig.tight_layout(); savefig(fig,out/"figures/ablation_accuracy")

    # Figure 2: summed reference confusion matrix across seeds, explicitly repeated predictions.
    cm=sum(np.array(d["metrics_by_seed"][f"full_fusion_v2|{s}"]["confusion_matrix"]["counts"]) for s in (42,43,44))
    fig,ax=plt.subplots(figsize=(5.8,5.0)); im=ax.imshow(cm,cmap="Blues")
    for i in range(4):
        for j in range(4): ax.text(j,i,str(int(cm[i,j])),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xticks(range(4),LABELS,rotation=30,ha="right"); ax.set_yticks(range(4),LABELS)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("FusionV2 confusion matrix (three seeds)")
    fig.colorbar(im,ax=ax,label="Prediction count")
    fig.text(.5,.015,"Summed over seeds: each of 848 images appears three times (not independent cases).",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1)); savefig(fig,out/"figures/reference_confusion")

    chosen=["full_fusion_v2","cnn_branch_only","vit_branch_only","no_auxiliary_loss"]
    fig,axs=plt.subplots(1,4,figsize=(13.2,3.7),sharex=True,sharey=True)
    for ax,v in zip(axs,chosen):
        raw=sum(np.array(d["metrics_by_seed"][f"{v}|{s}"]["confusion_matrix"]["counts"],float) for s in (42,43,44))
        norm=raw/raw.sum(axis=1,keepdims=True); im=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues")
        for i in range(4):
            for j in range(4): ax.text(j,i,f"{100*norm[i,j]:.1f}",ha="center",va="center",fontsize=7,color="white" if norm[i,j]>.55 else "black")
        ax.set_title(DISPLAY[v]); ax.set_xticks(range(4),["Gl","Me","NT","Pi"]); ax.set_yticks(range(4),["Gl","Me","NT","Pi"])
    axs[0].set_ylabel("True class"); fig.supxlabel("Predicted class (row-normalized %, summed over seeds)")
    fig.suptitle("Class-wise error patterns for reference and informative ablations",y=.99)
    fig.subplots_adjust(left=.06,right=.99,bottom=.18,top=.84,wspace=.18); savefig(fig,out/"figures/ablation_confusion_comparison")

    variants=VARIANTS[1:]; losses=[]; gains=[]
    for v in variants:
        ts=sorted((x for x in d["paired_tests"] if x["variant"]==v),key=lambda x:x["seed"])
        losses.append([x["table"]["reference_correct_ablation_wrong"] for x in ts])
        gains.append([x["table"]["reference_wrong_ablation_correct"] for x in ts])
    fig,axs=plt.subplots(1,2,figsize=(9.6,4.7),sharey=True); vmax=max(np.array(losses).max(),np.array(gains).max())
    for ax,arr,title in zip(axs,(losses,gains),("Reference correct / ablation wrong","Reference wrong / ablation correct")):
        im=ax.imshow(arr,aspect="auto",cmap="Oranges",vmin=0,vmax=vmax)
        for i in range(7):
            for j in range(3): ax.text(j,i,str(arr[i][j]),ha="center",va="center",fontsize=8)
        ax.set_xticks(range(3),["42","43","44"]); ax.set_title(title,fontsize=9); ax.set_xlabel("Seed")
    axs[0].set_yticks(range(7),[DISPLAY[v] for v in variants]); fig.colorbar(im,ax=axs.ravel().tolist(),label="Images",fraction=.035,pad=.03)
    fig.suptitle("Paired prediction disagreements versus full FusionV2 (n=848 per seed)")
    fig.subplots_adjust(left=.22,right=.9,bottom=.12,top=.84,wspace=.18); savefig(fig,out/"figures/paired_disagreements")

    # Figure 3: reference reliability, seed curves plus pooled bin counts.
    fig,(ax,bx)=plt.subplots(2,1,figsize=(6.6,6.4),height_ratios=[3,1],sharex=True)
    counts=np.zeros(15,dtype=int)
    for seed,col in zip((42,43,44),("#1f77b4","#ff7f0e","#2ca02c")):
        bins=d["metrics_by_seed"][f"full_fusion_v2|{seed}"]["ece_bins"]
        x=[]; yy=[]
        for i,b in enumerate(bins):
            counts[i]+=b["n"]
            if b["n"]: x.append(b["mean_confidence"]); yy.append(b["accuracy"])
        ax.plot(x,yy,"o-",ms=4,lw=1.2,label=f"Seed {seed}",color=col)
    ax.plot([0,1],[0,1],"--",color="#666",lw=1,label="Perfect calibration")
    ax.set_ylabel("Observed accuracy"); ax.set_title("FusionV2 reliability on duplicate-clean test set")
    ax.legend(loc="upper left",fontsize=8); ax.grid(alpha=.2)
    centers=(np.arange(15)+.5)/15; bx.bar(centers,counts,width=.058,color="#7f8fa6")
    bx.set_xlabel("Predicted confidence"); bx.set_ylabel("Count\n(3 seeds)"); bx.set_xlim(0,1)
    fig.text(.5,.01,"ECE uses 15 equal-width bins; curves are seed-specific and counts repeat the same 848 images across seeds.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1)); savefig(fig,out/"figures/reference_reliability")

    # Figure 4: gate mass, descriptive only, by true class and seed.
    gate_rows=[]
    for seed in (42,43,44):
        p=root/f"runs_v2_clean/fusion_v2_seed{seed}/gates_test.csv"
        with p.open(encoding="utf-8") as f:
            for r in csv.DictReader(f): gate_rows.append((seed,r["true_label"],float(r["gate_cnn"])))
    fig,ax=plt.subplots(figsize=(8.0,4.8)); positions=[]; data=[]; colors=[]; ticks=[]
    for ci,cl in enumerate([x.lower().replace(" ","") for x in LABELS]):
        ticks.append(ci*4+2)
        for si,seed in enumerate((42,43,44)):
            positions.append(ci*4+si+1); data.append([g for s,c,g in gate_rows if s==seed and c==cl]); colors.append(["#4c78a8","#f58518","#54a24b"][si])
    bp=ax.boxplot(data,positions=positions,widths=.7,patch_artist=True,showfliers=False)
    for patch,c in zip(bp["boxes"],colors): patch.set_facecolor(c); patch.set_alpha(.75)
    ax.set_xticks(ticks,LABELS); ax.set_ylabel("CNN gate mass")
    ax.set_title("Learned FusionV2 gate utilization by class and seed (descriptive only)")
    handles=[plt.Line2D([0],[0],color=c,lw=6,label=f"Seed {s}") for c,s in zip(["#4c78a8","#f58518","#54a24b"],(42,43,44))]
    ax.legend(handles=handles,loc="best",fontsize=8); ax.grid(axis="y",alpha=.2)
    fig.text(.5,.01,"Gate mass indicates branch weighting, not causal contribution, anatomical attention, or clinical explanation.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1)); savefig(fig,out/"figures/gate_mass_by_class")

    md=["# Phase 3C publication artifacts","",
        "All values and figures are generated from the accepted Phase 3B machine-readable evidence. No training, tuning, or model modification occurred.","",
        "## Primary ablation table","", "| Model | Accuracy mean ± seed SD (%) | Macro-F1 mean ± seed SD (%) | Observed Δ accuracy (pp) | Holm-significant seeds | Interpretation |",
        "|---|---:|---:|---:|---:|---|"]
    for r in ab:
        md.append(f"| {DISPLAY[r['model']]} | {r['accuracy_mean_pct']:.4f} ± {r['accuracy_seed_sd_pct']:.4f} | {r['macro_f1_mean_pct']:.4f} ± {r['macro_f1_seed_sd_pct']:.4f} | {r['observed_delta_accuracy_pp']:+.4f} | {r['holm_significant_seeds']} | {r['verdict']} |")
    md += ["","Seed SD describes spread across three training seeds; it is not a confidence interval. Observed differences are not labelled important or negligible because no prespecified margin exists.","",
           "## Figures","","- `ablation_accuracy`: seed means, seed SD, and individual seed results.","- `reference_confusion`: summed prediction counts across seeds; repeated images are explicitly labelled.","- `ablation_confusion_comparison`: normalized class-error patterns for informative controls.","- `paired_disagreements`: paired reference wins and losses by seed.","- `reference_reliability`: seed-specific calibration curves with bin counts.","- `gate_mass_by_class`: descriptive gate utilization only.","",
           "## Interpretation guardrails","","Calibration and gate comparisons are descriptive. Gate mass indicates utilization, not causal contribution, anatomical attention, or clinical explanation. Non-significance is not equivalence. These results concern image-level evaluation on one duplicate-clean benchmark partition and do not establish patient independence, clinical readiness, or cross-institutional generalization.",""]
    (out/"PHASE3C_ARTIFACTS.md").write_text("\n".join(md),encoding="utf-8")
    visual="PASS — all six PNG figures visually inspected for legibility, clipping, overlap, labels, and misleading axes." if a.visual_inspected else "PENDING — manual visual inspection required before Phase 3C acceptance."
    (out/"VISUAL_INSPECTION.md").write_text("# Visual inspection\n\n"+visual+"\n",encoding="utf-8")
    files=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in {"artifact_manifest.json","TEST_REPORT.md"}: files.append({"path":str(p.relative_to(root)).replace("\\","/"),"bytes":p.stat().st_size,"sha256":sha256(p)})
    manifest={"phase":"3C","source":str(source.relative_to(root)).replace("\\","/"),"source_sha256":sha256(source),
              "phase3b_acceptance_status":"PASS","training_performed":False,"model_modified":False,
              "n_images_per_seed":848,"seeds":[42,43,44],"visual_inspection":visual,"files":files,
              "environment":{"python":platform.python_version(),"matplotlib":matplotlib.__version__},
              "acceptance_status":"PASS" if a.visual_inspected else "PENDING_VISUAL_INSPECTION"}
    (out/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Phase 3C: {manifest['acceptance_status']} | {len(files)} artifacts")

if __name__=="__main__": main()
