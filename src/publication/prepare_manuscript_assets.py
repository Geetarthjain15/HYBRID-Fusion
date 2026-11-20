"""Prepare non-destructive manuscript figures and an evidence manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fusion_v2_gradcam import (CLASS_NAMES, gradcam_plus_plus, load_verified_model,
                               overlay_heatmap, predict, prepare_image)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_both(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture(out: Path):
    fig, ax = plt.subplots(figsize=(13, 8)); ax.set_xlim(0, 13); ax.set_ylim(0, 10); ax.axis("off")
    def box(x, y, w, h, title, detail, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",
                                    facecolor=color,edgecolor="#334155",linewidth=1.4))
        ax.text(x+w/2,y+h*.64,title,ha="center",va="center",fontsize=11,fontweight="bold")
        ax.text(x+w/2,y+h*.28,detail,ha="center",va="center",fontsize=8.7,color="#334155")
    def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=13,color="#64748b",lw=1.4))
    box(4.8,8.8,3.4,.8,"MRI image","Resize 224×224; normalize mean/std 0.5","#e2e8f0")
    box(.5,7.0,5.2,1.0,"EfficientNet-B0 CNN","Fully fine-tuned; discriminative backbone LR","#dbeafe")
    box(7.3,7.0,5.2,1.0,"ViT-B/16","CLS representation; fully fine-tuned","#ede9fe")
    box(.5,5.3,5.2,1.0,"Multi-scale pooling","Stages 3, 5, final: 40 + 112 + 1280 = 1432","#dbeafe")
    box(7.3,5.3,5.2,1.0,"Global representation","CLS token: 768 dimensions","#ede9fe")
    box(.5,3.7,5.2,1.0,"CNN projection","Linear 1432→512 · LayerNorm · GELU","#bfdbfe")
    box(7.3,3.7,5.2,1.0,"ViT projection","Linear 768→512 · LayerNorm · GELU","#ddd6fe")
    box(4.3,2.1,4.4,1.0,"Adaptive two-way gate","Input-dependent softmax [CNN, ViT]","#fef3c7")
    box(4.3,.5,4.4,1.0,"Fusion classifier","Weighted concat 1024 · LayerNorm · Dropout · 4 classes","#dcfce7")
    for a,b in [((6.5,8.8),(3.1,8)),((6.5,8.8),(9.9,8)),((3.1,7),(3.1,6.3)),((9.9,7),(9.9,6.3)),
                ((3.1,5.3),(3.1,4.7)),((9.9,5.3),(9.9,4.7)),((3.1,3.7),(5.2,3.1)),((9.9,3.7),(7.8,3.1)),
                ((6.5,2.1),(6.5,1.5))]: arrow(a,b)
    ax.text(.55,3.25,"Auxiliary CNN head (training only)",fontsize=8.5,color="#1d4ed8")
    ax.text(9.15,3.25,"Auxiliary ViT head (training only)",fontsize=8.5,color="#6d28d9",ha="center")
    ax.set_title("AGMF-Net (implementation identifier: FusionV2)",fontsize=16,fontweight="bold",pad=12)
    save_both(fig,out/"figure_1_agmf_net_architecture")


def qualitative_gradcam(root: Path, out: Path):
    pred_path=root/"runs_v2_clean/fusion_v2_seed42/preds_test.csv"
    rows=list(csv.DictReader(pred_path.open(encoding="utf-8")))
    chosen=[next(r for r in rows if r["true_label"]==c) for c in CLASS_NAMES]
    files={p.name:p for p in (root/"dataset").rglob("*") if p.is_file()}
    model,device,ck_hash=load_verified_model(root,42,"cpu")
    fig,axes=plt.subplots(2,4,figsize=(12,6.2))
    provenance=[]
    for j,row in enumerate(chosen):
        path=files[row["filename"]]
        display,tensor,rgb=prepare_image(Image.open(path))
        result=predict(model,tensor,device,True)
        heat=gradcam_plus_plus(model,tensor,device,result.predicted_index)
        overlay=overlay_heatmap(rgb,heat)
        axes[0,j].imshow(display); axes[0,j].axis("off"); axes[0,j].set_title(row["true_label"].title())
        axes[1,j].imshow(overlay); axes[1,j].axis("off"); axes[1,j].set_title(f"Pred: {result.predicted_class}")
        provenance.append({"class":row["true_label"],"filename":row["filename"],
                           "predicted":result.predicted_class,"checkpoint_sha256":ck_hash})
    fig.suptitle("Deterministically selected CNN-path Grad-CAM++ examples (seed 42)",fontweight="bold")
    fig.text(.5,.015,"Top: input. Bottom: CNN-path map for the fused predicted class. Qualitative only; no localization ground truth.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94)); save_both(fig,out/"figure_6_gradcam_examples")
    return provenance


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",default="."); ap.add_argument("--out-root",default="publication_results/manuscript_draft/figures")
    a=ap.parse_args(); root=Path(a.project_root).resolve(); out=(root/a.out_root).resolve(); out.mkdir(parents=True,exist_ok=True)
    architecture(out)
    copies={
      "figure_2_ablation_accuracy":root/"publication_results/ablations/publication_artifacts/figures/ablation_accuracy",
      "figure_3_clean_confusion":root/"publication_results/metrics/figures/confusion_clean",
      "figure_4_clean_reliability":root/"publication_results/metrics/figures/reliability_clean",
      "figure_5_gate_mass":root/"publication_results/ablations/publication_artifacts/figures/gate_mass_by_class",
      "figure_7_cross_source_confusion":root/"publication_results/metrics/figures/confusion_cross_source_disjoint"}
    for name,src in copies.items():
        for ext in (".png",".pdf"): shutil.copy2(src.with_suffix(ext),out/(name+ext))
    gradcam=qualitative_gradcam(root,out)
    files=[]
    for p in sorted(out.iterdir()):
        if p.is_file(): files.append({"file":p.name,"bytes":p.stat().st_size,"sha256":digest(p)})
    manifest={"purpose":"AGMF-Net manuscript assets","source_model":"FusionV2","files":files,
              "gradcam_examples":gradcam,
              "warning":"Grad-CAM++ examples are qualitative CNN-path visualizations without localization ground truth."}
    (out/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PASS","n_files":len(files),"out":str(out)},indent=2))

if __name__=="__main__": main()
