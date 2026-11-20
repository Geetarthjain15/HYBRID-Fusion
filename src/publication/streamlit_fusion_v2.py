"""Local research UI for verified FusionV2 and CNN-path Grad-CAM++."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from fusion_v2_gradcam import (
    CLASS_NAMES, DemoValidationError, gradcam_plus_plus, load_verified_model,
    overlay_heatmap, predict, prepare_image,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "publication_results/metrics/fusion_v2_metrics.json"

st.set_page_config(page_title="FusionV2 MRI Research Demo", page_icon="🧠", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1180px; padding-top: 1.5rem;}
  [data-testid="stMetric"] {background:#111827; border:1px solid #273449;
    border-radius:12px; padding:14px 16px;}
  .result-card {background:linear-gradient(135deg,#111827,#172033); border:1px solid #334155;
    border-radius:16px; padding:22px; margin:8px 0 18px 0;}
  .result-label {color:#94a3b8; font-size:.86rem; text-transform:uppercase; letter-spacing:.08em;}
  .result-class {font-size:2.25rem; font-weight:750; margin:.2rem 0; color:#f8fafc;}
  .result-score {display:inline-block; background:#064e3b; color:#6ee7b7;
    border-radius:999px; padding:5px 11px; font-weight:650;}
</style>
""", unsafe_allow_html=True)


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def uncertainty_summary(probabilities: np.ndarray) -> tuple[float, float, float]:
    """Return top score, top-two margin and entropy normalized to [0, 1]."""
    p = np.asarray(probabilities, dtype=np.float64)
    ordered = np.sort(p)[::-1]
    entropy = -float(np.sum(np.where(p > 0, p * np.log(p), 0.0))) / math.log(len(p))
    return float(ordered[0]), float(ordered[0] - ordered[1]), entropy


@st.cache_resource(show_spinner="Verifying and loading FusionV2 checkpoint…")
def cached_model(seed: int):
    return load_verified_model(PROJECT_ROOT, seed)


@st.cache_data
def load_metrics(path: str, modified_ns: int):
    del modified_ns
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("acceptance_status") != "PASS":
        raise DemoValidationError("Publication metrics do not carry PASS acceptance status")
    return data


def mean_sd(metric: dict, percent: bool = True) -> str:
    factor = 100 if percent else 1
    return f"{metric['mean'] * factor:.2f} ± {metric['sd_across_seeds'] * factor:.2f}"


def aggregate_metrics_table(metrics: dict) -> pd.DataFrame:
    labels = {"original": "Original/conventional", "clean": "Duplicate-clean",
              "cross_source_disjoint": "Cross-source disjoint"}
    rows = []
    for key, label in labels.items():
        setting = metrics["settings"][key]
        a = setting["across_seeds"]
        rows.append({"Evaluation": label, "Images": setting["n_images"],
                     "Accuracy %": mean_sd(a["accuracy"]),
                     "Macro-F1 %": mean_sd(a["macro_f1"]),
                     "Balanced accuracy %": mean_sd(a["balanced_accuracy"]),
                     "Macro ROC-AUC": mean_sd(a["macro_roc_auc_ovr"], False),
                     "Macro AP": mean_sd(a["macro_average_precision_ovr"], False),
                     "NLL": f"{a['nll']['mean']:.4f}",
                     "Brier": f"{a['brier']['mean']:.4f}",
                     "ECE": f"{a['ece']['mean']:.4f}"})
    return pd.DataFrame(rows)


def per_class_table(metrics: dict, setting_key: str) -> pd.DataFrame:
    rows = []
    for item in metrics["settings"][setting_key]["per_class_across_seeds"]:
        rows.append({"Class": item["class"], "Support": item["support"],
                     "Precision %": item["precision"]["mean"] * 100,
                     "Recall/sensitivity %": item["recall_sensitivity"]["mean"] * 100,
                     "Specificity %": item["specificity"]["mean"] * 100,
                     "F1 %": item["f1"]["mean"] * 100,
                     "ROC-AUC": item["roc_auc_ovr"]["mean"],
                     "Average precision": item["average_precision_ovr"]["mean"]})
    return pd.DataFrame(rows)


st.title("🧠 FusionV2 brain MRI classifier")
st.caption("Verified four-class research demonstration with CNN-path Grad-CAM++")

try:
    metrics = load_metrics(str(METRICS_PATH), METRICS_PATH.stat().st_mtime_ns)
except (OSError, ValueError, KeyError, DemoValidationError) as exc:
    st.error(f"Verified metrics could not be loaded safely: {exc}")
    st.stop()

with st.sidebar:
    st.header("Inference settings")
    seed = st.selectbox("Verified checkpoint", (42, 43, 44), index=0,
                        format_func=lambda x: f"FusionV2 — seed {x}")
    use_tta = st.checkbox("Horizontal-flip TTA", value=True,
                          help="Enabled in the verified FusionV2 evaluation protocol.")
    st.divider()
    st.markdown("**Accepted image files**")
    st.caption("JPG, JPEG, PNG, BMP, TIFF. DICOM and NIfTI are not accepted by this interface.")
    st.markdown("**Class order**")
    st.caption(" · ".join(CLASS_NAMES))
    st.divider()
    st.caption("The selected seed changes the verified checkpoint, not the model architecture.")

prediction_tab, metrics_tab = st.tabs(["Scan prediction", "Verified model metrics"])

with prediction_tab:
    uploaded = st.file_uploader("Upload one brain MRI image",
        type=("png", "jpg", "jpeg", "bmp", "tif", "tiff"),
        help="The image is processed locally and is not written to the dataset.")
    if uploaded is None:
        st.info("Upload a supported image to run the verified FusionV2 checkpoint.")
    else:
        raw = uploaded.getvalue()
        upload_id = file_sha256(raw)
        try:
            image = Image.open(uploaded); image.load()
            display, tensor, display_float = prepare_image(image)
            model, device, checkpoint_hash = cached_model(seed)
            result = predict(model, tensor, device, use_tta=use_tta)
            top_score, margin, entropy = uncertainty_summary(result.probabilities)
        except (UnidentifiedImageError, OSError) as exc:
            st.error(f"The uploaded file could not be decoded as an image: {exc}"); st.stop()
        except (DemoValidationError, RuntimeError, ValueError) as exc:
            st.error(f"Validation or inference failed safely: {exc}"); st.stop()

        left, right = st.columns([1.05, 1.25], gap="large")
        with left:
            st.image(display, caption="Uploaded scan (displayed after 224×224 resize)",
                     use_container_width=True)
        with right:
            st.markdown(f'<div class="result-card"><div class="result-label">Predicted class</div>'
                f'<div class="result-class">{result.predicted_class.upper()}</div>'
                f'<span class="result-score">Model score {top_score * 100:.2f}%</span></div>',
                unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Top-class score", f"{top_score * 100:.2f}%")
            c2.metric("Top-two margin", f"{margin * 100:.2f} pp")
            c3.metric("Normalized entropy", f"{entropy:.3f}", help="0 = concentrated; 1 = uniform")

        probability_df = pd.DataFrame({"Class": [c.title() if c != "notumor" else "No tumor" for c in CLASS_NAMES],
            "Probability": result.probabilities, "Probability %": result.probabilities * 100
        }).sort_values("Probability", ascending=False)
        st.subheader("All class probabilities")
        pcol, gcol = st.columns([1.35, 0.65])
        with pcol:
            st.bar_chart(probability_df.set_index("Class")["Probability"], horizontal=True)
            st.dataframe(probability_df[["Class", "Probability %"]].style.format({"Probability %": "{:.2f}"}),
                         hide_index=True, use_container_width=True)
        with gcol:
            st.markdown("**Dynamic fusion gate**")
            gate_df = pd.DataFrame({"Branch": ["CNN", "ViT"],
                                    "Gate mass": [result.gate_cnn, result.gate_vit]})
            st.bar_chart(gate_df.set_index("Branch"), horizontal=True)
            st.caption(f"CNN {result.gate_cnn * 100:.2f}% · ViT {result.gate_vit * 100:.2f}%. "
                       "Gate mass describes branch utilization, not causal importance.")

        state_key = f"heatmap_{upload_id}_{seed}"
        if st.button("Generate Grad-CAM++ heatmap", type="primary", use_container_width=True):
            st.session_state[state_key] = True
        if st.session_state.get(state_key, False):
            try:
                with st.spinner("Computing CNN-path Grad-CAM++…"):
                    heatmap = gradcam_plus_plus(model, tensor, device, result.predicted_index)
                    overlay = overlay_heatmap(display_float, heatmap)
                h1, h2, h3 = st.columns(3)
                h1.image(display, caption="Input", use_container_width=True)
                h2.image(heatmap, caption="CNN-path Grad-CAM++", clamp=True, use_container_width=True)
                h3.image(overlay, caption="Heatmap overlay", use_container_width=True)
                st.info("The heatmap is derived from the final spatial EfficientNet stage for the "
                        "selected fused class score. It does not spatially explain the ViT branch or "
                        "constitute verified tumor localization.")
            except (DemoValidationError, RuntimeError, ValueError) as exc:
                st.error(f"Heatmap generation failed safely: {exc}")

        report = {"model": "FusionV2", "checkpoint_seed": seed, "tta": use_tta,
            "predicted_class": result.predicted_class, "top_class_model_score": top_score,
            "top_two_margin": margin, "normalized_entropy": entropy,
            "probabilities": dict(zip(CLASS_NAMES, map(float, result.probabilities))),
            "gate_mass": {"cnn": result.gate_cnn, "vit": result.gate_vit},
            "checkpoint_sha256": checkpoint_hash, "uploaded_file_sha256": upload_id}
        st.download_button("Download this inference record (JSON)", json.dumps(report, indent=2),
                           "fusionv2_inference.json", "application/json")
        st.caption(f"Verified checkpoint SHA-256: `{checkpoint_hash}`")

with metrics_tab:
    st.subheader("Verified evaluation results")
    st.caption("Generated from stored predictions for seeds 42, 43 and 44; ± is sample SD across seeds.")
    st.dataframe(aggregate_metrics_table(metrics), hide_index=True, use_container_width=True)
    setting_label = st.selectbox("Per-class metric set", ("clean", "original", "cross_source_disjoint"),
        format_func={"clean": "Duplicate-clean (primary internal)", "original": "Original/conventional",
                     "cross_source_disjoint": "Cross-source disjoint"}.get)
    st.dataframe(per_class_table(metrics, setting_label).style.format({"Precision %": "{:.2f}",
        "Recall/sensitivity %": "{:.2f}", "Specificity %": "{:.2f}", "F1 %": "{:.2f}",
        "ROC-AUC": "{:.4f}", "Average precision": "{:.4f}"}), hide_index=True,
        use_container_width=True)
    selected = metrics["settings"]["clean"]["per_seed"][str(seed)]
    st.subheader(f"Selected checkpoint — seed {seed}, duplicate-clean set")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Accuracy", f"{selected['accuracy'] * 100:.2f}%")
    s2.metric("Macro-F1", f"{selected['macro_f1'] * 100:.2f}%")
    s3.metric("Brier score", f"{selected['brier']:.4f}")
    s4.metric("ECE", f"{selected['ece']:.4f}")
    cm = selected["confusion_matrix"]
    st.dataframe(pd.DataFrame(cm["counts"], index=cm["labels"], columns=cm["labels"]),
                 use_container_width=True)
    st.caption("Confusion matrix: rows are true classes; columns are predicted classes.")
