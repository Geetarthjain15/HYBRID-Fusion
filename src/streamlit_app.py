import os
import json
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:
    from transformers.models.auto import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from hybrid_model import HybridBrainTumorModel

st.set_page_config(page_title="Brain Tumor AI", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIT_PATH = os.path.join(BASE_DIR, "vit_brain_tumor_5class.pth") if os.path.exists(os.path.join(BASE_DIR, "vit_brain_tumor_5class.pth")) else "src/vit_brain_tumor_5class.pth"
HYBRID_PATH = os.path.join(BASE_DIR, "hybrid_brain_tumor.pth") if os.path.exists(os.path.join(BASE_DIR, "hybrid_brain_tumor.pth")) else "src/hybrid_brain_tumor.pth"
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
LABELS = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}

def get_vit():
    if not os.path.exists(VIT_PATH):
        st.error(f"Missing ViT weights at {VIT_PATH}. Please provide the file.")
        return None
    model = AutoModelForImageClassification.from_pretrained(
        "google/vit-base-patch16-224", num_labels=4, ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(VIT_PATH, map_location='cpu'))
    model.eval()
    return model

def get_hybrid():
    if not os.path.exists(HYBRID_PATH):
        st.error(f"Missing Hybrid weights at {HYBRID_PATH}. Please provide the file.")
        return None
    if not os.path.exists(VIT_PATH):
        st.error(f"Missing base ViT weights at {VIT_PATH} required for hybrid model.")
        return None
    model = HybridBrainTumorModel(VIT_PATH)
    model.load_state_dict(torch.load(HYBRID_PATH, map_location='cpu'))
    model.eval()
    return model

def get_cam(model, img, target, is_hybrid):
    img_tensor = processor(images=img, return_tensors="pt").pixel_values
    
    if is_hybrid:
        target_layers = [model.cnn_features[-1]]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    else:
        class ViTWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x): return self.m(x).logits
            def reshape_transform(self, t):
                return t[:, 1:, :].reshape(t.size(0), 14, 14, t.size(2)).permute(0, 3, 1, 2)
        
        wrapped = ViTWrapper(model)
        target_layers = [wrapped.m.vit.encoder.layer[-1].layernorm_before]
        cam = GradCAMPlusPlus(model=wrapped, target_layers=target_layers, reshape_transform=wrapped.reshape_transform)
    
    mask = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(target)])[0]
    img_float = np.array(img.resize((224, 224))).astype(np.float32) / 255.0
    return show_cam_on_image(img_float, mask, use_rgb=True)

st.title(" Brain Tumor AI")

# --- Performance Reference ---
with st.sidebar:
    choice = st.radio("Select Model", ["Original ViT", "Hybrid CNN+ViT"])
    is_hybrid = choice == "Hybrid CNN+ViT"
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    if is_hybrid:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                m = json.load(f)
            per_class_rows = "\n".join(
                f"        | **{c.title()}** | {a:.2%} |" for c, a in m["per_class_test_acc"].items()
            )
            st.markdown(f"""
        | Metric | Value |
        |---|---|
        | **Training Acc** | {m['train_acc']:.2%} |
        | **Best Val Acc** | {m['best_val_acc']:.2%} |
        | **Test Acc** | {m['test_acc']:.2%} |
{per_class_rows}
            """)
        else:
            st.caption("No metrics file found. Run `train.py` to generate `src/metrics.json`.")
    else:
        # The shipped ViT checkpoint (vit_brain_tumor_5class.pth) scores at
        # chance on the held-out test set (26.1% acc, macro ROC-AUC 0.50) - it
        # predicts 'pituitary' for ~99% of inputs. Shown for reference only.
        st.warning("This checkpoint is NOT a trained baseline: 26.1% test accuracy "
                   "(chance = 25%). Use the Hybrid model.")
        st.markdown("""
        | Metric | Value |
        |---|---|
        | **Test Acc** | 26.05% |
        | **Macro ROC-AUC** | 0.503 |
        | **Backbone** | ViT-B/16 |
        """)
    st.markdown("---")

processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = get_hybrid() if is_hybrid else get_vit()

if model is None:
    st.warning("Model failed to load. Please check the errors above.")
    st.stop()

file = st.file_uploader("Upload MRI", type=["jpg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)
    
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.pixel_values) if is_hybrid else model(**inputs).logits
    
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    idx = torch.argmax(probs).item()
    
    st.metric("Prediction", LABELS[idx].upper(), f"{probs[idx]:.2%}")
    
    # --- Safety Thresholds ---
    if probs[idx] < 0.40:
        st.error("🚨 This image doesn't look like a standard MRI. Results may be invalid.")
    elif probs[idx] < 0.60:
        st.warning("⚠️ Low confidence. The AI is struggling to identify clear features.")

    if st.button("Show Heatmap"):
        heatmap = get_cam(model, img, idx, is_hybrid)
        st.image(heatmap, caption="AI Reasoning Area")
