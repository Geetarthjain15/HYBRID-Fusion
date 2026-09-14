# 🧠 Brain Tumor Detection AI - Quick Start Guide

This project is a deep learning web application built with **PyTorch**, **HuggingFace Transformers (ViT)**, **TorchVision (EfficientNet)**, and **Streamlit** to detect brain tumors from MRI scans and display explainable AI heatmaps (Grad-CAM++).

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [How to Start the Project](#how-to-start-the-project)
3. [Model Architectures](#model-architectures)
4. [Dataset & Weight Files](#dataset--weight-files)
5. [How to Train on Custom MRI Dataset](#how-to-train-on-custom-mri-dataset)

---

## 1. Project Overview

The system classifies MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

It includes two model choices:
1. **Original ViT (Vision Transformer):** Pure ViT-B/16 architecture.
2. **Hybrid CNN + ViT:** Dual-branch fusion model combining local CNN features (EfficientNet-B0) with global ViT self-attention.

---

## 2. How to Start the Project

### Option A: Using existing Virtual Environment (`env`)
Run the following command from the `vit2` folder:

```bash
cd vit2
PYTHONPATH=src ./env/bin/streamlit run src/streamlit_app.py
```

### Option B: Installing in a new environment
1. Activate your virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run Streamlit:
   ```bash
   PYTHONPATH=src streamlit run src/streamlit_app.py
   ```

4. Open your browser and navigate to:
   **http://localhost:8501**

---

## 3. Dataset & Weight Files Summary

### ❓ Do we have the raw MRI dataset in this project folder?
- **No.** Raw image dataset folders (`dataset/Training` and `dataset/Testing`) are **not** committed inside the repository to keep the repository lightweight.

### ❓ How are the model weight files created/trained?
- Pre-trained backbone weights (`vit_brain_tumor_5class.pth` and `hybrid_brain_tumor.pth`) are stored in `src/`.
- The application automatically initializes and loads these weights out-of-the-box so you can run predictions and test the interface immediately without needing to run a multi-hour training script first.

---

## 4. How to Train on Custom MRI Dataset

If you want to train or fine-tune the models on your own MRI dataset:

1. Create a `dataset` directory inside `vit2/`:
   ```
   vit2/dataset/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```

2. Run the training script:
   ```bash
   cd vit2
   PYTHONPATH=src ./env/bin/python src/train.py
   ```
3. The trained weights will automatically save to `src/hybrid_brain_tumor.pth` and will be loaded by the web app!

---

## 📁 File Structure

```
vit2/
├── requirements.txt         # Dependencies
├── START_GUIDE.md           # Instructions guide
├── HOW_TO_START.txt         # Quick command instructions
└── src/
    ├── streamlit_app.py     # Main Web App UI & Grad-CAM visualization
    ├── hybrid_model.py      # Hybrid CNN+ViT PyTorch architecture
    ├── train.py             # Model training script
    ├── vit_brain_tumor_5class.pth  # ViT model weights
    └── hybrid_brain_tumor.pth     # Hybrid model weights
```
