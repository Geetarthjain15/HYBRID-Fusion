"""Phase 6: deterministic faithfulness audit for FusionV2 CNN-path Grad-CAM++.

This script does not treat a heatmap as a tumour mask.  It evaluates whether
pixels ranked highly by the displayed map affect the selected fused class score
more than a deterministic random ordering, and whether the map is stable under
a horizontal flip.  Outputs are isolated beneath publication_results/xai/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fusion_v2_gradcam import (  # noqa: E402
    CLASS_NAMES,
    gradcam_plus_plus,
    load_verified_model,
    overlay_heatmap,
    predict,
    prepare_image,
)


@dataclass(frozen=True)
class ExampleResult:
    filename: str
    source_class: str
    predicted_class: str
    target_index: int
    target_probability: float
    deletion_auc_cam: float
    deletion_auc_random: float
    insertion_auc_cam: float
    insertion_auc_random: float
    flip_stability_correlation: float
    heatmap_mean: float
    heatmap_std: float
    heatmap_nonzero_fraction: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_auc(values: list[float]) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("AUC requires at least two finite values")
    x = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    return float(np.trapezoid(np.asarray(values, dtype=np.float64), x))


def coarse_order(heatmap: np.ndarray, grid: int = 14) -> np.ndarray:
    """Rank a fixed grid by mean activation; avoids pretending 7x7 CAM is pixel precise."""
    if heatmap.shape != (224, 224):
        raise ValueError("Expected a 224x224 heatmap")
    block = 224 // grid
    scores = heatmap.reshape(grid, block, grid, block).mean(axis=(1, 3))
    return np.argsort(scores.ravel())[::-1]


def block_mask(order: np.ndarray, fraction: float, grid: int = 14) -> torch.Tensor:
    count = int(round(float(fraction) * grid * grid))
    coarse = np.zeros(grid * grid, dtype=np.float32)
    coarse[order[:count]] = 1.0
    coarse = coarse.reshape(grid, grid)
    expanded = np.repeat(np.repeat(coarse, 224 // grid, axis=0), 224 // grid, axis=1)
    return torch.from_numpy(expanded).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def target_probability(model, tensor: torch.Tensor, device: torch.device, target: int) -> float:
    return float(model(tensor.to(device)).softmax(-1)[0, target].detach().cpu())


def perturbation_curves(model, tensor: torch.Tensor, device: torch.device, target: int,
                        heatmap: np.ndarray, steps: int, random_seed: int) -> dict:
    fractions = np.linspace(0.0, 1.0, steps + 1)
    cam_order = coarse_order(heatmap)
    rng = np.random.default_rng(random_seed)
    random_order = rng.permutation(cam_order.size)
    baseline = torch.zeros_like(tensor)
    curves = {"deletion_cam": [], "deletion_random": [],
              "insertion_cam": [], "insertion_random": []}
    for fraction in fractions:
        for label, order in (("cam", cam_order), ("random", random_order)):
            mask = block_mask(order, float(fraction)).to(tensor.dtype)
            deleted = tensor * (1.0 - mask) + baseline * mask
            inserted = baseline * (1.0 - mask) + tensor * mask
            curves[f"deletion_{label}"].append(
                target_probability(model, deleted, device, target))
            curves[f"insertion_{label}"].append(
                target_probability(model, inserted, device, target))
    curves["fractions"] = fractions.tolist()
    curves["auc"] = {name: normalized_auc(values) for name, values in curves.items()
                     if name != "fractions"}
    return curves


def finite_correlation(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if av.std() <= 1e-12 or bv.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def first_images(test_root: Path) -> list[tuple[str, Path]]:
    examples = []
    for class_name in CLASS_NAMES:
        folder = test_root / class_name
        candidates = sorted(p for p in folder.glob("*") if p.is_file())
        if not candidates:
            raise FileNotFoundError(f"No image available for class {class_name}: {folder}")
        examples.append((class_name, candidates[0]))
    return examples


def save_figure(path_base: Path, image: np.ndarray, heatmap: np.ndarray,
                overlay: np.ndarray, title: str, result: ExampleResult, curves: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    axes[0, 0].imshow(image); axes[0, 0].set_title("Input")
    map_plot = axes[0, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Support for: {result.predicted_class}")
    fig.colorbar(map_plot, ax=axes[0, 1], fraction=0.046, label="Per-image normalized activation")
    axes[0, 2].imshow(overlay); axes[0, 2].set_title("CNN-path overlay")
    for ax in axes[0]:
        ax.axis("off")
    x = curves["fractions"]
    axes[1, 0].plot(x, curves["deletion_cam"], label="Grad-CAM++ order")
    axes[1, 0].plot(x, curves["deletion_random"], label="Random order", linestyle="--")
    axes[1, 0].set(title="Deletion", xlabel="Fraction replaced", ylabel="Target probability")
    axes[1, 1].plot(x, curves["insertion_cam"], label="Grad-CAM++ order")
    axes[1, 1].plot(x, curves["insertion_random"], label="Random order", linestyle="--")
    axes[1, 1].set(title="Insertion", xlabel="Fraction restored", ylabel="Target probability")
    axes[1, 0].legend(); axes[1, 1].legend()
    axes[1, 2].axis("off")
    note = (f"Prediction: {result.predicted_class} ({result.target_probability * 100:.2f}%)\n"
            f"Deletion AUC: CAM {result.deletion_auc_cam:.3f}, random {result.deletion_auc_random:.3f}\n"
            f"Insertion AUC: CAM {result.insertion_auc_cam:.3f}, random {result.insertion_auc_random:.3f}\n"
            f"Flip-aligned map correlation: {result.flip_stability_correlation:.3f}\n\n"
            "Red is the maximum within this image's selected-class map.\n"
            "It is not a tumour label, probability, or validated boundary.")
    axes[1, 2].text(0.0, 0.95, note, va="top", fontsize=10, wrap=True)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42, choices=(42, 43, 44))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root, out = args.project_root.resolve(), args.out_root.resolve()
    output_json = out / "XAI_AUDIT.json"
    if output_json.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_json}; use --resume after review")
    if args.steps < 4:
        raise ValueError("--steps must be at least 4")
    test_root = root / "dataset/Testing"
    examples = first_images(test_root)
    out.mkdir(parents=True, exist_ok=True)
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    model, device, checkpoint_hash = load_verified_model(root, args.seed, device="cpu")
    results, curve_records = [], {}
    for index, (source_class, path) in enumerate(examples):
        with Image.open(path) as loaded:
            display, tensor, display_float = prepare_image(loaded)
        prediction = predict(model, tensor, device, use_tta=False)
        target = prediction.predicted_index
        heatmap = gradcam_plus_plus(model, tensor, device, target)
        curves = perturbation_curves(model, tensor, device, target, heatmap,
                                     args.steps, random_seed=20260925 + index)
        flipped_tensor = torch.flip(tensor, dims=[3])
        flipped_map = gradcam_plus_plus(model, flipped_tensor, device, target)
        aligned_flipped_map = np.fliplr(flipped_map)
        item = ExampleResult(
            filename=path.relative_to(root).as_posix(), source_class=source_class,
            predicted_class=prediction.predicted_class, target_index=target,
            target_probability=float(prediction.probabilities[target]),
            deletion_auc_cam=curves["auc"]["deletion_cam"],
            deletion_auc_random=curves["auc"]["deletion_random"],
            insertion_auc_cam=curves["auc"]["insertion_cam"],
            insertion_auc_random=curves["auc"]["insertion_random"],
            flip_stability_correlation=finite_correlation(heatmap, aligned_flipped_map),
            heatmap_mean=float(heatmap.mean()), heatmap_std=float(heatmap.std()),
            heatmap_nonzero_fraction=float((heatmap > 0).mean()),
        )
        results.append(item)
        curve_records[source_class] = curves
        overlay = overlay_heatmap(display_float, heatmap)
        save_figure(figures / f"{source_class}_faithfulness", np.asarray(display),
                    heatmap, overlay,
                    f"CNN-path Grad-CAM++ audit — source class {source_class}", item, curves)
    payload = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 6 - Quantitative explainability evaluation",
        "project_root": str(root), "checkpoint_seed": args.seed,
        "checkpoint_sha256": checkpoint_hash,
        "target_layer": "FusionV2.cnn_stages[-1]",
        "target_score": "selected fused-model pre-softmax logit",
        "preprocessing": {"size": [224, 224], "mean": [0.5] * 3, "std": [0.5] * 3},
        "perturbation": {"grid": "14x14", "baseline": "normalized zero (display mid-gray)",
                         "steps": args.steps, "random_seed_base": 20260925},
        "examples": [asdict(item) for item in results],
        "curves": curve_records,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "torch": torch.__version__, "device": str(device)},
        "limitations": [
            "No lesion masks, bounding boxes, or clinician annotations are available.",
            "The map explains the CNN spatial pathway into one fused logit, not the ViT pathway.",
            "The displayed map is normalized independently per image; red means the within-map maximum.",
            "No-tumor maps visualize positive support for the no-tumor class, not a detected lesion.",
            "Faithfulness on four deterministic examples is a diagnostic, not clinical validation.",
        ],
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = ["# FusionV2 Grad-CAM++ faithfulness audit", "", "## Result", "",
          "The implementation targets the selected fused-model logit and the final spatial EfficientNet stage. "
          "A red region on a no-tumor prediction is expected because every nonconstant map is normalized "
          "independently: red is the maximum support for the selected **no-tumor** class within that image. "
          "It is not evidence that a tumor exists at that location.", "", "## Deterministic examples", "",
          "| Source class | Prediction | Probability | Deletion AUC CAM/random | Insertion AUC CAM/random | Flip correlation |",
          "|---|---|---:|---:|---:|---:|"]
    for item in results:
        md.append(f"| {item.source_class} | {item.predicted_class} | {item.target_probability:.4f} | "
                  f"{item.deletion_auc_cam:.4f}/{item.deletion_auc_random:.4f} | "
                  f"{item.insertion_auc_cam:.4f}/{item.insertion_auc_random:.4f} | "
                  f"{item.flip_stability_correlation:.4f} |")
    md += ["", "## Interpretation", "",
           "Deletion and insertion compare the Grad-CAM++ ranking with a fixed random control. Lower deletion "
           "AUC and higher insertion AUC are directionally consistent with a more faithful ranking, but four "
           "examples do not establish general validity.", "", "## Limits", "",
           "- Localization accuracy: **not available** because no verified lesion masks or boxes exist.",
           "- The map is CNN-path sensitivity, not a complete explanation of the gated CNN-ViT model.",
           "- The map does not explain horizontal-flip TTA probability averaging.",
           "- Per-image min-max normalization prevents color intensity from being compared across images.",
           "- No-tumor must be labelled as selected-class support, not as a tumor heatmap."]
    (out / "HEATMAP_VALIDATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "XAI_LIMITATIONS.md").write_text(
        "# XAI limitations\n\n"
        "No verified tumor masks, bounding boxes, or clinician annotations are available. Therefore, tumor "
        "localization accuracy cannot be computed and is reported as **not available**. Grad-CAM++ is a "
        "class-conditioned sensitivity map from the final CNN feature stage. It neither spatially decomposes "
        "the ViT branch nor proves clinically correct reasoning. A red maximum on a no-tumor image represents "
        "the strongest within-image support for the no-tumor score after per-image normalization; it does not "
        "represent tumor tissue.\n", encoding="utf-8")
    print(f"PASS: wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
