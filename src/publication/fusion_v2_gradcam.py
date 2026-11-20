"""Verified FusionV2 inference with a CNN-path Grad-CAM++ explanation.

The prediction is produced by the complete gated CNN-ViT model.  The heatmap
is computed from gradients of one fused output logit with respect to the final
spatial EfficientNet feature map.  It therefore visualises only the CNN spatial
pathway into that output; it is not a spatial explanation of the ViT branch or
proof of anatomical/clinical correctness.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.models import efficientnet_b0
from transformers import ViTConfig, ViTForImageClassification

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fusion_v2 import FINAL_CH, TAPS, VIT_DIM, FusionV2, tta_logits  # noqa: E402

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_SIZE = 224
NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(*NORM),
])


class OfflineFusionV2(FusionV2):
    """Architecture-identical FusionV2 construction without downloading base weights.

    Every parameter is immediately replaced by the strictly loaded verified
    checkpoint, so fetching ImageNet/Hugging Face initialization is unnecessary.
    """

    def __init__(self, num_classes: int = 4, proj_dim: int = 512,
                 multiscale: bool = True, gated: bool = True, dropout: float = 0.2):
        nn.Module.__init__(self)
        self.multiscale = multiscale
        self.gated = gated
        self.cnn_stages = efficientnet_b0(weights=None).features
        cnn_dim = FINAL_CH + (sum(TAPS.values()) if multiscale else 0)
        self.vit = ViTForImageClassification(ViTConfig(num_labels=num_classes))
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.vit_proj = nn.Sequential(
            nn.Linear(VIT_DIM, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.gate = nn.Sequential(
            nn.Linear(2 * proj_dim, 256), nn.GELU(), nn.Linear(256, 2))
        self.head = nn.Sequential(
            nn.LayerNorm(2 * proj_dim), nn.Dropout(dropout),
            nn.Linear(2 * proj_dim, num_classes))
        self.aux_cnn = nn.Linear(proj_dim, num_classes)
        self.aux_vit = nn.Linear(proj_dim, num_classes)


class DemoValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    class_names: Sequence[str]
    probabilities: np.ndarray
    predicted_index: int
    gate_cnn: float
    gate_vit: float

    @property
    def predicted_class(self) -> str:
        return self.class_names[self.predicted_index]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expected_checkpoint_hash(project_root: Path, seed: int) -> str:
    manifest_path = project_root / "publication_results/ablations/checkpoint_recovery/import_manifest.json"
    if not manifest_path.is_file():
        raise DemoValidationError(f"Checkpoint recovery manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("seeds", {}).get(str(seed))
    if not entry or entry.get("decision") != "ACCEPTED":
        raise DemoValidationError(f"Seed {seed} is not ACCEPTED in {manifest_path}")
    return entry["import"]["destination_sha256"]


def checkpoint_path(project_root: Path, seed: int) -> Path:
    return project_root / f"publication_runs/recovered_checkpoints/clean/full_fusion_v2/seed_{seed}/checkpoint.pth"


def load_verified_model(project_root: str | os.PathLike, seed: int = 42,
                        device: str | torch.device | None = None) -> tuple[FusionV2, torch.device, str]:
    root = Path(project_root).resolve()
    path = checkpoint_path(root, seed)
    if not path.is_file():
        raise DemoValidationError(f"Verified checkpoint missing: {path}")
    expected = expected_checkpoint_hash(root, seed)
    actual = sha256(path)
    if actual != expected:
        raise DemoValidationError(
            f"Checkpoint SHA-256 mismatch for seed {seed}: expected {expected}, got {actual}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = OfflineFusionV2(num_classes=len(CLASS_NAMES), multiscale=True, gated=True)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(dev).eval()
    return model, dev, actual


def prepare_image(image: Image.Image) -> tuple[Image.Image, torch.Tensor, np.ndarray]:
    rgb = image.convert("RGB")
    display = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    display_float = np.asarray(display, dtype=np.float32) / 255.0
    tensor = EVAL_TRANSFORM(rgb).unsqueeze(0)
    return display, tensor, display_float


@torch.no_grad()
def predict(model: FusionV2, tensor: torch.Tensor, device: torch.device,
            use_tta: bool = True) -> Prediction:
    x = tensor.to(device)
    probabilities = (tta_logits(model, x) if use_tta else model(x).softmax(-1))[0]
    _, _, _, gate = model(x, return_aux=True)
    if gate is None or gate.shape != (1, 2):
        raise DemoValidationError("Expected one learned two-branch gate row")
    probs = probabilities.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(probs).all() or abs(float(probs.sum()) - 1.0) > 1e-5:
        raise DemoValidationError("Invalid model probabilities")
    return Prediction(CLASS_NAMES, probs, int(probs.argmax()),
                      float(gate[0, 0]), float(gate[0, 1]))


def gradcam_plus_plus(model: FusionV2, tensor: torch.Tensor, device: torch.device,
                      target_index: int) -> np.ndarray:
    if target_index not in range(len(CLASS_NAMES)):
        raise ValueError(f"target_index must be 0..{len(CLASS_NAMES) - 1}")
    # Use the final spatial EfficientNet stage. The target remains the complete
    # fused model logit, so the learned gate and both downstream projections are
    # retained in the gradient path.
    x = tensor.to(device)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        with GradCAMPlusPlus(model=model, target_layers=[model.cnn_stages[-1]]) as cam:
            heatmap = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_index)])[0]
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(heatmap).all():
        raise DemoValidationError(f"Invalid Grad-CAM++ output: shape={heatmap.shape}")
    lo, hi = float(heatmap.min()), float(heatmap.max())
    if hi > lo:
        heatmap = (heatmap - lo) / (hi - lo)
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


def overlay_heatmap(display_float: np.ndarray, heatmap: np.ndarray,
                    image_weight: float = 0.55) -> np.ndarray:
    if display_float.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("display_float must be a 224x224 RGB image")
    if heatmap.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("heatmap must be 224x224")
    return show_cam_on_image(display_float, heatmap, use_rgb=True,
                             image_weight=image_weight)
